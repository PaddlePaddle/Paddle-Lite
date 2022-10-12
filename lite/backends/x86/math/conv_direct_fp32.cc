/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
#include "lite/backends/x86/math/avx/conv_utils.h"
#include "lite/core/context.h"
#ifdef __AVX__
#include <immintrin.h>
#else
#include <emmintrin.h>
#endif
#include "lite/backends/x86/math/conv_direct_fp32.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

#define GET_OFF(field) offsetof(jit_param, field)

struct jit_param {
  const float* in_row_addr;
  const float* kernel_addr;
  float* out_row_addr;
  size_t oc;
  size_t ic;
  size_t wh;
};

conv_direct::conv_direct() : JitCode(8192, Xbyak::AutoGrow) {}

void conv_direct::generate_code(int ic,
                                int ih,
                                int iw,
                                int oc,
                                int oc_expand,
                                int oh,
                                int ow,
                                int ph,
                                int pw,
                                int ww,
                                int wh,
                                int stridew) {
#ifdef __AVX__
  constexpr int BLOCK = 8;
#else
  constexpr int BLOCK = 4;
#endif

  constexpr int oc_block = 4 * BLOCK;
  const int oc_loop_n =
      oc_expand / oc_block;  // deal with every 32 output channels
  const int oc_remain = oc_expand % oc_block;
  int temp;
  constexpr int ow_bulk = 3;

  constexpr int ic_block = 8;
  const int ic_loop_n = ic / ic_block;  // we deal with every 8 input channels
  const int ic_remain = ic % ic_block;

  int new_ow = 0;  // 0 or 3 or 6....
  bool right = false;
  if (pw == 0) {
    new_ow = ow / ow_bulk * ow_bulk;
    new_ow -= ow_bulk;
  } else if (pw > 0) {
    // whether padding is required for the rightet output
    right = stridew * (ow - 1) + ww - iw - pw > 0;
    if (right)
      new_ow = ow % ow_bulk ? ow / ow_bulk * ow_bulk : ow - ow_bulk;
    else
      new_ow = ow / ow_bulk * ow_bulk;
    new_ow -= ow_bulk;
  }

  // The number of channels of convolution kernel
  // and ic are always the same !
  int wc = ic;
  int ihw = ih * iw;
  int wchw = wc * wh * ww;
  int whwB = wh * ww * BLOCK;
  int ohw = oh * ow;

  preCode();  // save regs that must be saved
  using reg64_t = const Xbyak::Reg64;
  reg64_t ow_bulk_i_xb = rax;
  reg64_t in_row_addr_xb = r8;
  mov(in_row_addr_xb, ptr[param1 + GET_OFF(in_row_addr)]);
  reg64_t kernel_addr_xb = r9;
  mov(kernel_addr_xb, ptr[param1 + GET_OFF(kernel_addr)]);
  reg64_t out_row_addr_xb = r10;
  mov(out_row_addr_xb, ptr[param1 + GET_OFF(out_row_addr)]);
  reg64_t oc_xb = r11;
  mov(oc_xb, ptr[param1 + GET_OFF(oc)]);
  reg64_t wh_xb = r12;
  mov(wh_xb, ptr[param1 + GET_OFF(wh)]);
  reg64_t ic_xb = r13;
  mov(ic_xb, ptr[param1 + GET_OFF(ic)]);
  reg64_t wh_i_xb = r14;

  reg64_t aux_in_row_addr_xb = r15;
  reg64_t aux_kernel_addr_xb = rdx;

  // Take out oc_group/BLOCK * bulk results at a time
  // bulk are usually ow_bulk=3
  auto load = [=, &temp](int oc_group, int bulk) {
    for (int oc_gi = 0; oc_gi < oc_group; oc_gi += BLOCK) {
      for (int j = 0; j < bulk; j++) {
        Vmm res(oc_gi / BLOCK * bulk + j);
        temp = (oc_gi * ohw + j * BLOCK) * sizeof(float);
        vmovups(res, ptr[out_row_addr_xb + temp]);
      }
    }
  };

  // store oc_group/BLOCK * bulk results at a time
  // bulk are usually ow_bulk(3)
  auto store = [=, &temp](int oc_group, int bulk) {
    // Store output is required here! 12 outputs!
    for (int oc_gi = 0; oc_gi < oc_group; oc_gi += BLOCK) {
      for (int j = 0; j < bulk; j++) {
        Vmm res(oc_gi / BLOCK * bulk + j);
        temp = (oc_gi * ohw + j * BLOCK) * sizeof(float);
        vmovups(ptr[out_row_addr_xb + temp], res);
      }
    }
  };

  // one_line means we only compute one line kernel and one line input
  // every input row consists of [l_pad's data, real data, r_pad's data]
  auto fmadd_one_line = [=, &temp](
      int oc_group, int ic_group, int l_pad, int r_pad, int bulk) {
    // kernel has ww columns, but I only take one number at a time
    for (int ww_i = 0; ww_i < ww; ww_i++) {
      for (int ic_i = 0; ic_i < ic_group; ic_i++) {  // in channel
        // get three input data for most cases because bulk is usually 3
        for (int j = 0; j < bulk; j++) {
          // no need to fetch this input
          if (ww_i + stridew * j < l_pad) continue;
          if (ww_i + stridew * j >= stridew * (bulk - 1) + ww - r_pad) continue;

          Vmm input = Vmm(oc_group / BLOCK * bulk + j);
          temp = (ww_i - l_pad + stridew * j + ic_i * ihw) * sizeof(float);
          vbroadcastss(input, ptr[aux_in_row_addr_xb + temp]);
        }

        for (int oc_gi = 0; oc_gi < oc_group; oc_gi += BLOCK) {
          //  fetch one float number in kernel
          Vmm kernel = Vmm(15);
          temp = (oc_gi * wchw + ic_i * whwB + ww_i * BLOCK) * sizeof(float);
          vmovups(kernel, ptr[aux_kernel_addr_xb + temp]);

          for (int j = 0; j < bulk; j++) {
            // no need to fetch this input
            if (ww_i + stridew * j < l_pad) continue;
            if (ww_i + stridew * j >= stridew * (bulk - 1) + ww - r_pad)
              continue;

            Vmm input = Vmm(oc_group / BLOCK * bulk + j);
            Vmm res(oc_gi / BLOCK * bulk + j);
            vfmadd231ps(res, kernel, input);
          }
        }
      }
    }
  };

  // compute output whole line
  // three parts: [left, middle, right]
  auto cal_out_whole_line = [=](int oc_group, int ic_group) {
    int ow_bulk_i = ow / ow_bulk;

    auto cal_bulk = [=](
        int oc_group, int ic_group, int l_pad, int r_pad, int bulk) {
      load(oc_group, bulk);

      mov(aux_in_row_addr_xb, in_row_addr_xb);
      mov(aux_kernel_addr_xb, kernel_addr_xb);

      Xbyak::Label wh_loop;
      mov(wh_i_xb, wh_xb);

      L(wh_loop);
      {
        fmadd_one_line(oc_group, ic_group, l_pad, r_pad, bulk);
        add(aux_in_row_addr_xb, iw * sizeof(float));
        add(aux_kernel_addr_xb, ww * BLOCK * sizeof(float));
        dec(wh_i_xb);
        cmp(wh_i_xb, 0);
        jg(wh_loop, T_NEAR);  // T_NEAR is required if the size between jmp and
                              // label is > 127 byte
      }

      store(oc_group, bulk);
    };

    // entry !
    // left
    // we need check if there is a left part
    // besided, we need check if the first ow_bulk output is
    // associated with right boundry
    int ow_rpad = 0;
    int first_ow_bulk_rpad = (ow_bulk - 1) * stridew + ww - pw - iw;
    if (first_ow_bulk_rpad > 0)  // means ow <= 3
      ow_rpad = (ow - 1) * stridew + ww - pw - iw;

    mov(ow_bulk_i_xb, 0);
    if (pw > 0 || ow <= ow_bulk) {
      int temp_rpad = 0;
      if (ow_rpad > 0) temp_rpad = ow_rpad;
      cal_bulk(oc_group, ic_group, pw, temp_rpad, ow <= ow_bulk ? ow : ow_bulk);

      inc(ow_bulk_i_xb);
      add(in_row_addr_xb, (stridew * ow_bulk - pw) * sizeof(float));
      add(out_row_addr_xb, ow_bulk * BLOCK * sizeof(float));
      ow_bulk_i--;
    }

    // judge whether there is an middle part
    if (ow_bulk_i > 0 && !(right && ow == 2 * ow_bulk)) {
      // middle !
      Xbyak::Label ow_loop;
      L(ow_loop);
      {
        cal_bulk(oc_group, ic_group, 0, 0, ow_bulk);

        add(in_row_addr_xb, stridew * ow_bulk * sizeof(float));
        add(out_row_addr_xb, ow_bulk * BLOCK * sizeof(float));
        inc(ow_bulk_i_xb);
        cmp(ow_bulk_i_xb, new_ow / ow_bulk);
        jle(ow_loop, T_NEAR);
      }
    }

    // ow_remain = rightest index - ??
    int ow_remain = (ow - 1) - (new_ow + 2);
    if (ow_remain > 0 && new_ow >= 0) {  // right exists
      ow_rpad = (ow - 1) * stridew + ww - pw - iw;
      int r_pad = right ? ow_rpad : 0;
      cal_bulk(oc_group, ic_group, 0, r_pad, ow_remain);
    }
  };

  //
  auto cal_with_ic_fixed = [=](int ic_group) {
    // ic_group is fixed
    // according to oc !
    Xbyak::Label label_oc_remain;
    Xbyak::Label oc_done;
    if (oc_loop_n >= 1 && oc_remain == 0) {
      cal_out_whole_line(oc_block, ic_group);
    } else if (oc_loop_n >= 1 && oc_remain != 0) {
      cmp(oc_xb, oc_block);
      jne(label_oc_remain, T_NEAR);

      cal_out_whole_line(oc_block, ic_group);
      jmp(oc_done, T_NEAR);

      L(label_oc_remain);
      cal_out_whole_line(oc_remain, ic_group);

      L(oc_done);
    } else {
      cal_out_whole_line(oc_remain, ic_group);
    }
  };

  // here is real entry !
  Xbyak::Label label_ic_remain;
  Xbyak::Label done;
  if (ic_loop_n >= 1 && ic_remain == 0) {
    cal_with_ic_fixed(ic_block);
  } else if (ic_loop_n >= 1 && ic_remain != 0) {
    cmp(ic_xb, ic_block);
    jne(label_ic_remain, T_NEAR);

    cal_with_ic_fixed(ic_block);
    jmp(done, T_NEAR);

    L(label_ic_remain);
    cal_with_ic_fixed(ic_remain);
  } else {
    cal_with_ic_fixed(ic_remain);
  }

  L(done);
  // restore the values of some registers and ret
  postCode();
}

void conv_direct::run(const float* i_data,
                      const float* trans_weight,
                      float* trans_out,
                      int bs,
                      int ic,
                      int ih,
                      int iw,
                      int oc,
                      int oc_expand,
                      int oh,
                      int ow,
                      int ph,
                      int pw,
                      int wh,
                      int ww,
                      int strideh) {
#ifdef __AVX__
  constexpr int BLOCK = 8;
#else
  constexpr int BLOCK = 4;
#endif

  constexpr int oc_block = 4 * BLOCK;
  constexpr int ic_block = 8;
  int wc = ic;

  int ichw = ic * ih * iw;
  int ihw = ih * iw;
  int whwB = wh * ww * BLOCK;
  int ochw = oc * oh * ow;

  // strideh *
  if (iw >= 224) {
    // fetch bs_i th input feature map
    for (int bs_i = 0; bs_i < bs; bs_i++) {
      // calculate the result of each line of output
      auto cal_out_line = [=](const float* in_row_addr,
                              const float* trans_weight,
                              float* out_row_addr,
                              int wh) {
        for (int ic_i = 0; ic_i < ic; ic_i += ic_block) {
          for (int oc_gi = 0; oc_gi < oc; oc_gi += oc_block) {
            jit_param param;
            param.in_row_addr = in_row_addr + ic_i * ihw;
            param.kernel_addr =
                trans_weight + oc_gi / BLOCK * whwB * wc + ic_i * whwB;
            param.out_row_addr = out_row_addr + oc_gi * oh * ow;
            param.oc = oc_gi + oc_block - 1 < oc ? oc_block : oc - oc_gi;
            param.ic = ic_i + ic_block - 1 < ic ? ic_block : ic - ic_i;
            param.wh = wh;

            void (*f)(jit_param*) =
                CodeGenerator::getCode<void (*)(jit_param*)>();
            f(&param);
          }
        }
      };

      const float* in_row_addr = i_data + bs_i * ichw;
      float* out_row_addr = trans_out + bs_i * ochw;

      int oh_i = 0;
      if (ph > 0) {  // upper boundry
        int temp_wh =
            wh - ph;  // we olny need deal with temp_wh rows not wh rows!

        // check if the kernel will occupy lower boundry
        // if so, we need decrease temp_wh again
        if (ih + ph < wh) temp_wh -= (wh - ih - ph);
        cal_out_line(
            in_row_addr, trans_weight + ph * ww * BLOCK, out_row_addr, temp_wh);
        oh_i++;
        in_row_addr += (strideh - ph) * iw;
        out_row_addr += ow * BLOCK;
      }
      // middle
      for (; oh_i < oh - 1; oh_i++) {
        cal_out_line(in_row_addr, trans_weight, out_row_addr, wh);
        out_row_addr += ow * BLOCK;
        in_row_addr += strideh * iw;
      }

      if (oh_i >= oh) continue;

      // lower boundary,
      // compute how many boundry rows is used to the lowerest output row
      int lower = strideh * (oh - 1) + wh - ph - ih;

      if (lower > 0) {
        cal_out_line(in_row_addr, trans_weight, out_row_addr, wh - lower);
      } else {
        cal_out_line(in_row_addr, trans_weight, out_row_addr, wh);
      }
    }
  } else {
    // fetch bs_i th input feature map
    for (int bs_i = 0; bs_i < bs; bs_i++) {
      for (int ic_i = 0; ic_i < ic; ic_i += ic_block) {
        for (int oc_gi = 0; oc_gi < oc; oc_gi += oc_block) {
          const float* in_row_addr = i_data + bs_i * ichw + ic_i * ihw;
          float* out_row_addr = trans_out + bs_i * ochw + oc_gi * oh * ow;
          const float* weight =
              trans_weight + oc_gi * wc * whwB / BLOCK + ic_i * whwB;
          jit_param param;
          param.oc = oc_gi + oc_block - 1 < oc ? oc_block : oc - oc_gi;
          param.ic = ic_i + ic_block - 1 < ic ? ic_block : ic - ic_i;
          void (*f)(jit_param*) =
              CodeGenerator::getCode<void (*)(jit_param*)>();

          int oh_i = 0;
          if (ph > 0) {  // upper boundry
            int temp_wh =
                wh - ph;  // we olny need deal with temp_wh rows not wh rows!

            // check if the kernel will occupy lower boundry
            // if so, we need decrease temp_wh again
            if (ih + ph < wh) temp_wh -= (wh - ih - ph);

            param.wh = temp_wh;
            param.in_row_addr = in_row_addr;
            param.kernel_addr = weight + ph * ww * BLOCK;
            param.out_row_addr = out_row_addr;
            f(&param);
            oh_i++;
            in_row_addr += (strideh - ph) * iw;
            out_row_addr += ow * BLOCK;
          }

          // middle
          for (; oh_i < oh - 1; oh_i++) {
            param.wh = wh;
            param.in_row_addr = in_row_addr;
            param.kernel_addr = weight;
            param.out_row_addr = out_row_addr;
            f(&param);
            out_row_addr += ow * BLOCK;
            in_row_addr += strideh * iw;
          }

          if (oh_i >= oh) continue;

          // lower boundary,
          // compute how many boundry rows is used to the lowerest output row
          int lower = strideh * (oh - 1) + wh - ph - ih;

          if (lower > 0) {
            param.wh = wh - lower;
            param.in_row_addr = in_row_addr;
            param.kernel_addr = weight;
            param.out_row_addr = out_row_addr;
            f(&param);
          } else {
            param.wh = wh;
            param.in_row_addr = in_row_addr;
            param.kernel_addr = weight;
            param.out_row_addr = out_row_addr;
            f(&param);
          }
        }
      }
    }
  }
}

// we always assume oc % BLOCK == 0!
// convert [N C/8 H W 8] to [N C H W]!
void conv_direct_transpose_out(int bs,
                               int oc,
                               int oh,
                               int ow,
                               float* o_data,
                               float* trans_out,
                               const float* bias,
                               lite_api::ActivationType active_type,
                               operators::ActivationParam act_param) {
#ifdef __AVX__
  constexpr int BLOCK = 8;
#else
  constexpr int BLOCK = 4;
#endif

  int ohw = oh * ow;
  int ochw = oc * oh * ow;

  // fetch bs_i th input feature map
  for (int bs_i = 0; bs_i < bs; bs_i++) {
    for (int oc_gi = 0; oc_gi < oc; oc_gi += BLOCK) {
      // trans_out's start_index, we need fetch 8x8 element;
      float* from_address = trans_out + bs_i * oc * ohw + oc_gi * ohw;
      float* dst_address = o_data + bs_i * ochw + oc_gi * ohw;

      for (int oh_i = 0; oh_i < oh; oh_i++) {
        int ow_i = 0;

        for (; ow_i + BLOCK - 1 < ow; ow_i += BLOCK,
                                      from_address += BLOCK * BLOCK,
                                      dst_address += BLOCK) {
#ifdef __AVX__
          __m256 row0 = _mm256_loadu_ps(from_address + 0 * BLOCK);
          __m256 row1 = _mm256_loadu_ps(from_address + 1 * BLOCK);
          __m256 row2 = _mm256_loadu_ps(from_address + 2 * BLOCK);
          __m256 row3 = _mm256_loadu_ps(from_address + 3 * BLOCK);
          __m256 row4 = _mm256_loadu_ps(from_address + 4 * BLOCK);
          __m256 row5 = _mm256_loadu_ps(from_address + 5 * BLOCK);
          __m256 row6 = _mm256_loadu_ps(from_address + 6 * BLOCK);
          __m256 row7 = _mm256_loadu_ps(from_address + 7 * BLOCK);
          transpose8_ps(row0, row1, row2, row3, row4, row5, row6, row7);
#else

          __m128 row0 = _mm_loadu_ps(from_address + 0 * BLOCK);
          __m128 row1 = _mm_loadu_ps(from_address + 1 * BLOCK);
          __m128 row2 = _mm_loadu_ps(from_address + 2 * BLOCK);
          __m128 row3 = _mm_loadu_ps(from_address + 3 * BLOCK);
          _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

#endif

          if (bias != nullptr) {
#ifdef __AVX__
            row0 = _mm256_add_ps(row0, _mm256_set1_ps(bias[oc_gi + 0]));
            row1 = _mm256_add_ps(row1, _mm256_set1_ps(bias[oc_gi + 1]));
            row2 = _mm256_add_ps(row2, _mm256_set1_ps(bias[oc_gi + 2]));
            row3 = _mm256_add_ps(row3, _mm256_set1_ps(bias[oc_gi + 3]));
            row4 = _mm256_add_ps(row4, _mm256_set1_ps(bias[oc_gi + 4]));
            row5 = _mm256_add_ps(row5, _mm256_set1_ps(bias[oc_gi + 5]));
            row6 = _mm256_add_ps(row6, _mm256_set1_ps(bias[oc_gi + 6]));
            row7 = _mm256_add_ps(row7, _mm256_set1_ps(bias[oc_gi + 7]));
#else
            row0 = _mm_add_ps(row0, _mm_set1_ps(bias[oc_gi + 0]));
            row1 = _mm_add_ps(row1, _mm_set1_ps(bias[oc_gi + 1]));
            row2 = _mm_add_ps(row2, _mm_set1_ps(bias[oc_gi + 2]));
            row3 = _mm_add_ps(row3, _mm_set1_ps(bias[oc_gi + 3]));
#endif
          }

          if (active_type == lite_api::ActivationType::kRelu) {
#ifdef __AVX__
            __m256 vzero = _mm256_set1_ps(0.f);
            row0 = _mm256_max_ps(row0, vzero);
            row1 = _mm256_max_ps(row1, vzero);
            row2 = _mm256_max_ps(row2, vzero);
            row3 = _mm256_max_ps(row3, vzero);
            row4 = _mm256_max_ps(row4, vzero);
            row5 = _mm256_max_ps(row5, vzero);
            row6 = _mm256_max_ps(row6, vzero);
            row7 = _mm256_max_ps(row7, vzero);
#else
            row0 = _mm_max_ps(row0, _mm_set1_ps(0.f));
            row1 = _mm_max_ps(row1, _mm_set1_ps(0.f));
            row2 = _mm_max_ps(row2, _mm_set1_ps(0.f));
            row3 = _mm_max_ps(row3, _mm_set1_ps(0.f));
#endif
          } else if (active_type == lite_api::ActivationType::kRelu6) {
#ifdef __AVX__
            __m256 vzero = _mm256_set1_ps(0.f);
            __m256 vsix = _mm256_set1_ps(act_param.Relu_clipped_coef);
            row0 = _mm256_max_ps(row0, vzero);
            row1 = _mm256_max_ps(row1, vzero);
            row2 = _mm256_max_ps(row2, vzero);
            row3 = _mm256_max_ps(row3, vzero);
            row4 = _mm256_max_ps(row4, vzero);
            row5 = _mm256_max_ps(row5, vzero);
            row6 = _mm256_max_ps(row6, vzero);
            row7 = _mm256_max_ps(row7, vzero);
            row0 = _mm256_min_ps(row0, vsix);
            row1 = _mm256_min_ps(row1, vsix);
            row2 = _mm256_min_ps(row2, vsix);
            row3 = _mm256_min_ps(row3, vsix);
            row4 = _mm256_min_ps(row4, vsix);
            row5 = _mm256_min_ps(row5, vsix);
            row6 = _mm256_min_ps(row6, vsix);
            row7 = _mm256_min_ps(row7, vsix);

#else
            __m128 vzero = _mm_set1_ps(0.f);
            __m128 vsix = _mm_set1_ps(act_param.Relu_clipped_coef);
            row0 = _mm_max_ps(row0, vzero);
            row1 = _mm_max_ps(row1, vzero);
            row2 = _mm_max_ps(row2, vzero);
            row3 = _mm_max_ps(row3, vzero);
            row0 = _mm_min_ps(row0, vsix);
            row1 = _mm_min_ps(row1, vsix);
            row2 = _mm_min_ps(row2, vsix);
            row3 = _mm_min_ps(row3, vsix);
#endif

          } else if (active_type == lite_api::ActivationType::kLeakyRelu) {
#ifdef __AVX__
            __m256 vzero = _mm256_set1_ps(0.f);
            __m256 vscale = _mm256_set1_ps(act_param.Leaky_relu_alpha);
            row0 = _mm256_blendv_ps(_mm256_mul_ps(row0, vscale),
                                    row0,
                                    _mm256_cmp_ps(row0, vzero, _CMP_GT_OS));
            row1 = _mm256_blendv_ps(_mm256_mul_ps(row1, vscale),
                                    row1,
                                    _mm256_cmp_ps(row1, vzero, _CMP_GT_OS));
            row2 = _mm256_blendv_ps(_mm256_mul_ps(row2, vscale),
                                    row2,
                                    _mm256_cmp_ps(row2, vzero, _CMP_GT_OS));
            row3 = _mm256_blendv_ps(_mm256_mul_ps(row3, vscale),
                                    row3,
                                    _mm256_cmp_ps(row3, vzero, _CMP_GT_OS));
            row4 = _mm256_blendv_ps(_mm256_mul_ps(row4, vscale),
                                    row4,
                                    _mm256_cmp_ps(row4, vzero, _CMP_GT_OS));
            row5 = _mm256_blendv_ps(_mm256_mul_ps(row5, vscale),
                                    row5,
                                    _mm256_cmp_ps(row5, vzero, _CMP_GT_OS));
            row6 = _mm256_blendv_ps(_mm256_mul_ps(row6, vscale),
                                    row6,
                                    _mm256_cmp_ps(row6, vzero, _CMP_GT_OS));
            row7 = _mm256_blendv_ps(_mm256_mul_ps(row7, vscale),
                                    row7,
                                    _mm256_cmp_ps(row7, vzero, _CMP_GT_OS));
#else
            __m128 vzero = _mm_set1_ps(0.f);
            __m128 vscale = _mm_set1_ps(act_param.Leaky_relu_alpha);
            row0 = _mm_blendv_ps(_mm_mul_ps(row0, vscale),
                                 row0,
                                 _mm_cmp_ps(row0, vzero, _CMP_GT_OS));
            row1 = _mm_blendv_ps(_mm_mul_ps(row1, vscale),
                                 row1,
                                 _mm_cmp_ps(row1, vzero, _CMP_GT_OS));
            row2 = _mm_blendv_ps(_mm_mul_ps(row2, vscale),
                                 row2,
                                 _mm_cmp_ps(row2, vzero, _CMP_GT_OS));
            row3 = _mm_blendv_ps(_mm_mul_ps(row3, vscale),
                                 row3,
                                 _mm_cmp_ps(row3, vzero, _CMP_GT_OS));
#endif
          } else if (active_type == lite_api::ActivationType::kHardSwish) {
#ifdef __AVX__
            __m256 vzero = _mm256_set1_ps(0.f);
            __m256 voffset = _mm256_set1_ps(act_param.hard_swish_offset);
            __m256 vscale = _mm256_set1_ps(1.0 / act_param.hard_swish_scale);
            __m256 vthreshold = _mm256_set1_ps(act_param.hard_swish_threshold);
            row0 = _mm256_mul_ps(
                _mm256_min_ps(
                    vthreshold,
                    _mm256_max_ps(_mm256_add_ps(row0, voffset), vzero)),
                _mm256_mul_ps(row0, vscale));
            row1 = _mm256_mul_ps(
                _mm256_min_ps(
                    vthreshold,
                    _mm256_max_ps(_mm256_add_ps(row1, voffset), vzero)),
                _mm256_mul_ps(row1, vscale));
            row2 = _mm256_mul_ps(
                _mm256_min_ps(
                    vthreshold,
                    _mm256_max_ps(_mm256_add_ps(row2, voffset), vzero)),
                _mm256_mul_ps(row2, vscale));
            row3 = _mm256_mul_ps(
                _mm256_min_ps(
                    vthreshold,
                    _mm256_max_ps(_mm256_add_ps(row3, voffset), vzero)),
                _mm256_mul_ps(row3, vscale));
            row4 = _mm256_mul_ps(
                _mm256_min_ps(
                    vthreshold,
                    _mm256_max_ps(_mm256_add_ps(row4, voffset), vzero)),
                _mm256_mul_ps(row4, vscale));
            row5 = _mm256_mul_ps(
                _mm256_min_ps(
                    vthreshold,
                    _mm256_max_ps(_mm256_add_ps(row5, voffset), vzero)),
                _mm256_mul_ps(row5, vscale));
            row6 = _mm256_mul_ps(
                _mm256_min_ps(
                    vthreshold,
                    _mm256_max_ps(_mm256_add_ps(row6, voffset), vzero)),
                _mm256_mul_ps(row6, vscale));
            row7 = _mm256_mul_ps(
                _mm256_min_ps(
                    vthreshold,
                    _mm256_max_ps(_mm256_add_ps(row7, voffset), vzero)),
                _mm256_mul_ps(row7, vscale));
#else
            __m128 vzero = _mm_set1_ps(0.f);
            __m128 voffset = _mm_set1_ps(act_param.hard_swish_offset);
            __m128 vscale = _mm_set1_ps(1.0 / act_param.hard_swish_scale);
            __m128 vthreshold = _mm_set1_ps(act_param.hard_swish_threshold);
            row0 = _mm_mul_ps(
                _mm_min_ps(vthreshold,
                           _mm_max_ps(_mm_add_ps(row0, voffset), vzero)),
                _mm_mul_ps(row0, vscale));
            row1 = _mm_mul_ps(
                _mm_min_ps(vthreshold,
                           _mm_max_ps(_mm_add_ps(row1, voffset), vzero)),
                _mm_mul_ps(row1, vscale));
            row2 = _mm_mul_ps(
                _mm_min_ps(vthreshold,
                           _mm_max_ps(_mm_add_ps(row2, voffset), vzero)),
                _mm_mul_ps(row2, vscale));
            row3 = _mm_mul_ps(
                _mm_min_ps(vthreshold,
                           _mm_max_ps(_mm_add_ps(row3, voffset), vzero)),
                _mm_mul_ps(row3, vscale));
#endif
          } else if (active_type == lite_api::ActivationType::kIndentity) {
          } else {
            LOG(FATAL) << "[X86] unsupported Activation type";
          }

#ifdef __AVX__
          _mm256_storeu_ps(dst_address + 0 * ohw, row0);
          _mm256_storeu_ps(dst_address + 1 * ohw, row1);
          _mm256_storeu_ps(dst_address + 2 * ohw, row2);
          _mm256_storeu_ps(dst_address + 3 * ohw, row3);
          _mm256_storeu_ps(dst_address + 4 * ohw, row4);
          _mm256_storeu_ps(dst_address + 5 * ohw, row5);
          _mm256_storeu_ps(dst_address + 6 * ohw, row6);
          _mm256_storeu_ps(dst_address + 7 * ohw, row7);
#else
          _mm_storeu_ps(dst_address + 0 * ohw, row0);
          _mm_storeu_ps(dst_address + 1 * ohw, row1);
          _mm_storeu_ps(dst_address + 2 * ohw, row2);
          _mm_storeu_ps(dst_address + 3 * ohw, row3);
#endif
        }

        for (; ow_i < ow; ow_i++, from_address += BLOCK, dst_address += 1) {
#ifdef __AVX__
          __m256 row = _mm256_loadu_ps(from_address);
#else
          __m128 row = _mm_loadu_ps(from_address);
#endif
          if (bias != nullptr) {
#ifdef __AVX__
            row = _mm256_add_ps(row, _mm256_loadu_ps(&bias[oc_gi]));
#else
            row = _mm_add_ps(row, _mm_loadu_ps(&bias[oc_gi]));
#endif
          }
          if (active_type == lite_api::ActivationType::kRelu) {
#ifdef __AVX__
            row = _mm256_max_ps(row, _mm256_set1_ps(0.f));
#else
            row = _mm_max_ps(row, _mm_set1_ps(0.f));
#endif
          } else if (active_type == lite_api::ActivationType::kRelu6) {
#ifdef __AVX__
            row = _mm256_max_ps(row, _mm256_set1_ps(0.f));
            row =
                _mm256_min_ps(row, _mm256_set1_ps(act_param.Relu_clipped_coef));
#else
            row = _mm_max_ps(row, _mm_set1_ps(0.f));
            row = _mm_min_ps(row, _mm_set1_ps(act_param.Relu_clipped_coef));

#endif
          } else if (active_type == lite_api::ActivationType::kLeakyRelu) {
#ifdef __AVX__
            __m256 val_scale =
                _mm256_mul_ps(row, _mm256_set1_ps(act_param.Leaky_relu_alpha));
            row = _mm256_blendv_ps(
                val_scale,
                row,
                _mm256_cmp_ps(row, _mm256_setzero_ps(), _CMP_GT_OS));
#else
            __m128 val_scale =
                _mm_mul_ps(row, _mm_set1_ps(act_param.Leaky_relu_alpha));
            row = _mm_blendv_ps(
                val_scale, row, _mm_cmp_ps(row, _mm_setzero_ps(), _CMP_GT_OS));
#endif
          } else if (active_type == lite_api::ActivationType::kHardSwish) {
#ifdef __AVX__
            __m256 val_offset =
                _mm256_add_ps(row, _mm256_set1_ps(act_param.hard_swish_offset));
            __m256 val_scale = _mm256_mul_ps(
                row, _mm256_set1_ps(1.0 / act_param.hard_swish_scale));
            __m256 val =
                _mm256_min_ps(_mm256_set1_ps(act_param.hard_swish_threshold),
                              _mm256_max_ps(val_offset, _mm256_setzero_ps()));
            row = _mm256_mul_ps(val, val_scale);
#else
            __m128 val_offset =
                _mm_add_ps(row, _mm_set1_ps(act_param.hard_swish_offset));
            __m128 val_scale =
                _mm_mul_ps(row, _mm_set1_ps(1.0 / act_param.hard_swish_scale));
            __m128 val = _mm_min_ps(_mm_set1_ps(act_param.hard_swish_threshold),
                                    _mm_max_ps(val_offset, _mm_setzero_ps()));
            row = _mm_mul_ps(val, val_scale);

#endif
          } else if (active_type == lite_api::ActivationType::kIndentity) {
          } else {
            LOG(FATAL) << "[X86] unsupported Activation type";
          }
#ifdef __AVX__
          *(dst_address + 0 * oh * ow) = (reinterpret_cast<float*>(&row))[0];
          *(dst_address + 1 * oh * ow) = (reinterpret_cast<float*>(&row))[1];
          *(dst_address + 2 * oh * ow) = (reinterpret_cast<float*>(&row))[2];
          *(dst_address + 3 * oh * ow) = (reinterpret_cast<float*>(&row))[3];
          *(dst_address + 4 * oh * ow) = (reinterpret_cast<float*>(&row))[4];
          *(dst_address + 5 * oh * ow) = (reinterpret_cast<float*>(&row))[5];
          *(dst_address + 6 * oh * ow) = (reinterpret_cast<float*>(&row))[6];
          *(dst_address + 7 * oh * ow) = (reinterpret_cast<float*>(&row))[7];
#else
          *(dst_address + 0 * oh * ow) = (reinterpret_cast<float*>(&row))[0];
          *(dst_address + 1 * oh * ow) = (reinterpret_cast<float*>(&row))[1];
          *(dst_address + 2 * oh * ow) = (reinterpret_cast<float*>(&row))[2];
          *(dst_address + 3 * oh * ow) = (reinterpret_cast<float*>(&row))[3];
#endif
        }
      }
    }
  }
}
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
