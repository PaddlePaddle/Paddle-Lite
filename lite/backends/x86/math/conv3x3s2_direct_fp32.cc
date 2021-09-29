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
#include "lite/backends/x86/math/conv3x3s2_direct_fp32.h"
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

conv_direct_3x3s2::conv_direct_3x3s2() : JitCode(8192, Xbyak::AutoGrow) {}

void conv_direct_3x3s2::generate_code(int ic,
                                      int ih,
                                      int iw,
                                      int oc,
                                      int oc_expand,
                                      int oh,
                                      int ow,
                                      int ph,
                                      int pw) {
#ifdef __AVX__
  constexpr int BLOCK = 8;
#else
  constexpr int BLOCK = 4;
#endif

  constexpr int ww = 3;
  constexpr int wh = 3;
  constexpr int stridew = 2;
  constexpr int oc_block = 4 * BLOCK;
  const int oc_loop_n =
      oc_expand / oc_block;  // we deal with every 32 output channels
  const int oc_remain = oc_expand % oc_block;
  int temp;
  constexpr int ow_bulk = 3;

  constexpr int ic_block = 8;
  const int ic_loop_n = ic / ic_block;  // we deal with every 8 input channels
  const int ic_remain = ic % ic_block;

  int new_ow;
  bool right = false;
  if (ph == 0 && pw == 0) {
    new_ow = ow / ow_bulk * ow_bulk;
    new_ow -= ow_bulk;
  } else if (ph == 1 && pw == 1) {
    // whether padding is required for the rightet output
    right = (-pw + stridew * (ow - 1) + ww - 1) == ((iw - 1) + pw);
    if (right)
      new_ow = ow % ow_bulk ? ow / ow_bulk * ow_bulk : ow - ow_bulk;
    else
      new_ow = ow / ow_bulk * ow_bulk;
    new_ow -= ow_bulk;
  } else {
    LOG(FATAL) << "[X86] conv_direct only support 3x3s2 with padding = 0 or 1";
  }

  // The number of channels of convolution kernel
  // and ic are always the same !
  int wc = ic;
  int ihw = ih * iw;
  int wchw = wc * wh * ww;
  int whwB = wh * ww * BLOCK;
  int ohw = oh * ow;

  preCode();
  using reg64_t = const Xbyak::Reg64;
  reg64_t ow_i = rax;
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

  // Take out oc_group/BLOCK * bulk results at a time
  // bulk are usually ow_bulk
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
  auto fmadd_one_line = [=, &temp](
      int oc_group, int ic_group, int l_pad, int r_pad, int bulk) {
    // kernel has ww columns, but I only take one number at a time
    for (int ww_i = 0; ww_i < ww; ww_i++) {
      for (int ic_i = 0; ic_i < ic_group; ic_i++) {  // in channel
        // get three input data for most cases because bulk is usually 3
        for (int j = 0; j < bulk; j++) {
          // no need to fetch this input
          if (ww_i + stridew * j < l_pad) continue;
          if (ww_i + stridew * j >= stridew * bulk + 1 - r_pad) continue;

          Vmm input = Vmm(oc_group / BLOCK * bulk + j);
          temp = (ww_i - l_pad + stridew * j + ic_i * ihw) * sizeof(float);
          vbroadcastss(input, ptr[in_row_addr_xb + temp]);
        }

        for (int oc_gi = 0; oc_gi < oc_group; oc_gi += BLOCK) {
          //  fetch one float number in kernel
          Vmm kernel = Vmm(15);
          temp = (oc_gi * wchw + ic_i * whwB + ww_i * BLOCK) * sizeof(float);
          vmovups(kernel, ptr[kernel_addr_xb + temp]);

          for (int j = 0; j < bulk; j++) {
            // no need to fetch this input
            if (ww_i + stridew * j < l_pad) continue;
            if (ww_i + stridew * j >= stridew * bulk + 1 - r_pad) continue;

            Vmm input = Vmm(oc_group / BLOCK * bulk + j);
            Vmm res(oc_gi / BLOCK * bulk + j);
            vfmadd231ps(res, kernel, input);
          }
        }
      }
    }
  };

  auto cal_out_whole_line = [=, &temp](int oc_group, int ic_group) {
    auto cal_bulk = [=, &temp](
        int oc_group, int ic_group, int l_pad, int r_pad, int bulk) {
      load(oc_group, bulk);
      push(in_row_addr_xb);
      push(kernel_addr_xb);

      Xbyak::Label wh_loop;
      mov(wh_i_xb, wh_xb);

      L(wh_loop);
      {
        fmadd_one_line(oc_group, ic_group, l_pad, r_pad, bulk);
        add(in_row_addr_xb, iw * sizeof(float));
        add(kernel_addr_xb, ww * BLOCK * sizeof(float));
        dec(wh_i_xb);
        cmp(wh_i_xb, 0);
        jg(wh_loop, T_NEAR);
      }

      pop(kernel_addr_xb);
      pop(in_row_addr_xb);

      store(oc_group, bulk);
    };

    // entry !
    // left
    mov(ow_i, 0);
    if (pw == 1) {
      cal_bulk(oc_group, ic_group, pw, 0, ow_bulk);
      add(ow_i, ow_bulk);
      add(in_row_addr_xb, 5 * sizeof(float));
      add(out_row_addr_xb, ow_bulk * BLOCK * sizeof(float));
    }
    // middle !
    Xbyak::Label ow_loop;
    L(ow_loop);
    {
      cal_bulk(oc_group, ic_group, 0, 0, ow_bulk);

      add(in_row_addr_xb, stridew * ow_bulk * sizeof(float));
      add(out_row_addr_xb, ow_bulk * BLOCK * sizeof(float));
      add(ow_i, ow_bulk);
      cmp(ow_i, new_ow);
      jle(ow_loop, T_NEAR);
    }
    // right
    int ow_remain = (ow - 1) - (new_ow + 2);
    if (ow_remain) {
      int r_pad = right ? pw : 0;
      cal_bulk(oc_group, ic_group, 0, r_pad, ow_remain);
    }
  };

  //
  auto cal_with_ic_fixed = [=, &temp](int ic_group) {
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
  postCode();
}

void conv_direct_3x3s2::run(const float* i_data,
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
                            int pw) {
#ifdef __AVX__
  constexpr int BLOCK = 8;
#else
  constexpr int BLOCK = 4;
#endif

  constexpr int ww = 3;
  constexpr int wh = 3;
  constexpr int strideh = 2;
  constexpr int oc_block = 4 * BLOCK;
  constexpr int ic_block = 8;
  int wc = ic;

  int ichw = ic * ih * iw;
  int ihw = ih * iw;
  int whwB = wh * ww * BLOCK;
  int ochw = oc * oh * ow;

  // fetch bs_i th input feature map
  for (int bs_i = 0; bs_i < bs; bs_i++) {
    // calculate the result of each line of output
    auto cal_out_line = [=](const float* in_row_addr,
                            const float* trans_weight,
                            float* out_row_addr,
                            int wh) {
      for (int ic_i = 0; ic_i < ic; ic_i += 8) {
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
              reinterpret_cast<void (*)(jit_param*)>(getCodeInternal());
          f(&param);
        }
      }
    };

    const float* in_row_addr = i_data + bs_i * ichw;
    float* out_row_addr = trans_out + bs_i * ochw;

    int oh_i = 0;
    if (ph == 1) {  // upper boundry
      cal_out_line(in_row_addr, trans_weight + ww * BLOCK, out_row_addr, 2);
      oh_i++;
      in_row_addr += iw;
      out_row_addr += ow * BLOCK;
    }
    // middle
    for (; oh_i < oh - 1; oh_i++) {
      cal_out_line(in_row_addr, trans_weight, out_row_addr, wh);
      out_row_addr += ow * BLOCK;
      in_row_addr += strideh * iw;
    }

    // lower boundary
    if (oh_i >= oh) continue;

    bool lower = (-ph + strideh * (oh - 1) + wh - 1) == ((ih - 1) + ph);

    if (ph == 1 && lower) {
      cal_out_line(in_row_addr, trans_weight, out_row_addr, 2);
    } else {
      cal_out_line(in_row_addr, trans_weight, out_row_addr, wh);
    }
  }
}

// we always assume oc % BLOCK == 0!
// convert [N C/8 H W 8] to [N C H W]!
void conv_direct_3x3s2_tranpose_out(int bs,
                                    int oc,
                                    int oh,
                                    int ow,
                                    float* o_data,
                                    float* trans_out,
                                    const float* bias,
                                    lite_api::ActivationType active_type) {
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
            __m256 vsix = _mm256_set1_ps(6.f);
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
            __m128 vsix = _mm_set1_ps(6.f);
            row0 = _mm_max_ps(row0, vzero);
            row1 = _mm_max_ps(row1, vzero);
            row2 = _mm_max_ps(row2, vzero);
            row3 = _mm_max_ps(row3, vzero);
            row0 = _mm_min_ps(row0, vsix);
            row1 = _mm_min_ps(row1, vsix);
            row2 = _mm_min_ps(row2, vsix);
            row3 = _mm_min_ps(row3, vsix);
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
            row = _mm256_min_ps(row, _mm256_set1_ps(6.f));
#else
            row = _mm_max_ps(row, _mm_set1_ps(0.f));
            row = _mm_min_ps(row, _mm_set1_ps(6.f));
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
