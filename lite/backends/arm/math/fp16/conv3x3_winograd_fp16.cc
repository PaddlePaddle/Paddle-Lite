// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/backends/arm/math/fp16/conv_block_utils_fp16.h"
#include "lite/backends/arm/math/fp16/conv_impl_fp16.h"
#include "lite/backends/arm/math/fp16/gemm_c8_fp16.h"
#include "lite/core/parallel_defines.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif
#include <arm_neon.h>
namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
void input_trans_c8_4x4_fp16(const float16_t* src,
                             int src_stride,
                             int src_h_stride,
                             float16_t* dest,
                             int dest_stride,
                             int dest_h_stride);

void input_trans_c8_6x6_fp16(const float16_t* src,
                             int src_stride,
                             float16_t* dest,
                             int dest_stride);

void output_trans_c8_post_2x4_fp16(const float16_t* src,
                                   int src_stride,
                                   int src_h_stride,
                                   float16_t* dest,
                                   int dest_stride,
                                   int dest_h_stride);

void output_trans_c8_post_4x6_fp16(const float16_t* src,
                                   int src_stride,
                                   float16_t* dest,
                                   int dest_stride);
void weight_trans_c8_4x4_fp16(
    float16_t* dest, const float16_t* src, int ic, int oc, void* workspace);
void weight_trans_c8_6x6_fp16(
    float16_t* dest, const float16_t* src, int ic, int oc, void* workspace);
// F(2,3)
void conv_compute_2x2_3x3_fp16(const float16_t* input,
                               float16_t* output,
                               int num,
                               int chout,
                               int hout,
                               int wout,
                               int chin,
                               int hin,
                               int win,
                               const float16_t* weight,
                               const float16_t* bias,
                               const operators::ConvParam& param,
                               ARMContext* ctx) {
  auto act_param = param.activation_param;
  const int pad_h0 = (*param.paddings)[0];
  const int pad_h1 = (*param.paddings)[1];
  const int pad_w0 = (*param.paddings)[2];
  const int pad_w1 = (*param.paddings)[3];
  float16_t* tmp_work_space =
      ctx->workspace_data<float16_t>() + ctx->llc_size() / sizeof(float16_t);

  int in_n_stride = chin * hin * win;
  int out_n_stride = chout * hout * wout;
  int ic_stride = win * hin;
  int oc_stride = wout * hout;
  int ic_8 = (chin + 7) / 8;
  int oc_8 = (chout + 7) / 8;

  int tile_w = (wout + 1) / 2;
  int tile_h = (hout + 1) / 2;
  int size_tile = tile_h * tile_w;

  int w_pad = win + pad_w0 + pad_w1;
  int h_pad = hin + pad_h0 + pad_h1;

  const int zero_len = (w_pad + 3) / 4 * 4;
  float16_t zero_ptr[zero_len];  // NOLINT
  memset(zero_ptr, 0, zero_len * sizeof(float16_t));

  float16_t* input_c8 = tmp_work_space;
  int new_h_stride = w_pad * 8;
  int new_c_stride = new_h_stride * h_pad;

  int ic_8_stride = w_pad * h_pad * 8;
  int oc_8_stride = wout * hout * 8;

  int tile_block = 8;
  int block_count = (size_tile + tile_block - 1) / tile_block;

  int threads = ctx->threads();
  float16_t* g_tmp_data = tmp_work_space + ic_8 * ic_8_stride;
  int tmp_input_thread_stride = tile_block * ic_8 * 128;
  int tmp_output_thread_stride = tile_block * oc_8 * 128;
  int tmp_data_thread_stride =
      tmp_input_thread_stride + tmp_output_thread_stride;
  memset(g_tmp_data, 0, threads * tmp_data_thread_stride * sizeof(float16_t));
  float16_t* g_trans_tmp_data = g_tmp_data + threads * tmp_data_thread_stride;
  float16_t* g_trans_remain_tmp_data = g_trans_tmp_data + threads * 128;
  bool flag_bias = (bias != nullptr);
  auto act_type = act_param.active_type;
  float16_t local_alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  float16_t offset = 0.f;
  float16_t threshold = 6.f;

  if (act_param.has_active) {
    act_acquire(act_type, flag_act, local_alpha, offset, threshold, act_param);
  }

  // begin compute
  for (int ni = 0; ni < num; ++ni) {
    // trans input to c4
    for (int i = 0; i < ic_8; ++i) {
      prepack_input_nxwc8_fp16_dw(input + ni * in_n_stride,
                                  input_c8 + i * new_c_stride,
                                  i * 8,
                                  -pad_h0,
                                  hin + pad_h1,
                                  -pad_w0,
                                  win + pad_w1,
                                  chin,
                                  win,
                                  hin,
                                  zero_ptr);
    }
    float16_t* output_ptr = output + ni * out_n_stride;

    const float16_t* weight_ptr = weight;
    const float16_t* bias_ptr = bias;
    LITE_PARALLEL_BEGIN(tbi, tid, block_count) {
#ifdef LITE_USE_THREAD_POOL
      float16_t* tmp_data = g_tmp_data + tid * tmp_data_thread_stride;
      float16_t* trans_tmp_data = g_trans_tmp_data + tid * 128;
      float16_t* trans_remain_tmp_data = g_trans_remain_tmp_data + tid * 128;
#elif defined(ARM_WITH_OMP)
      float16_t* tmp_data =
          g_tmp_data + omp_get_thread_num() * tmp_data_thread_stride;
      float16_t* trans_tmp_data = g_trans_tmp_data + omp_get_thread_num() * 128;
      float16_t* trans_remain_tmp_data =
          g_trans_remain_tmp_data + omp_get_thread_num() * 128;
#else
      float16_t* tmp_data = g_tmp_data;
      float16_t* trans_tmp_data = g_trans_tmp_data;
      float16_t* trans_remain_tmp_data = g_trans_remain_tmp_data;
#endif
      int tile_index = tbi * tile_block;
      int tile_remain = size_tile - tile_index;
      int tile_count = tile_remain > tile_block ? tile_block : tile_remain;

      // input trans
      int c_gi_stride = tile_count * oc_8 * 8;
      int b_gi_stride = tile_count * ic_8 * 8;
      //*
      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int src_x = tw_index + tw_index;
        int src_y = th_index + th_index;
        int ex = src_x + 4 > w_pad ? w_pad - src_x : 4;
        int ey = src_y + 4 > h_pad ? h_pad - src_y : 4;

        float16_t* dst_ptr = tmp_data + ti * 8;
        const float16_t* src_ptr = input_c8 + (src_y * w_pad + src_x) * 8;

        if (ex == 4 && ey == 4) {
          // trans input
          for (int ci = 0; ci < ic_8; ++ci) {
            const float16_t* src_ci = src_ptr + ci * ic_8_stride;
            float16_t* dst_ci = dst_ptr + ci * tile_count * 8;
            input_trans_c8_4x4_fp16(
                src_ci, 8, w_pad * 8, dst_ci, b_gi_stride, b_gi_stride * 4);
          }
        } else {
          // trans remain input
          int x_size = ex;
          for (int ci = 0; ci < ic_8; ++ci) {
            const float16_t* src_ci = src_ptr + ci * ic_8_stride;
            // pad
            memset(trans_remain_tmp_data, 0, 128 * sizeof(float16_t));
            if (x_size > 0) {
              for (int yi = 0; yi < ey; ++yi) {
                float16_t* dst_yi = trans_remain_tmp_data + yi * 32;
                const float16_t* src_yi = src_ci + w_pad * yi * 8;
                memcpy(dst_yi, src_yi, x_size * sizeof(float16_t) * 8);
              }
            }

            // trans
            float16_t* dst_ci = dst_ptr + ci * tile_count * 8;
            input_trans_c8_4x4_fp16(trans_remain_tmp_data,
                                    8,
                                    32,
                                    dst_ci,
                                    b_gi_stride,
                                    b_gi_stride * 4);
          }  // for ci_8
        }
      }
      //*/
      // input trans end
      // begin compute
      float16_t* dst_temp_data = tmp_data + tmp_input_thread_stride;
      float16_t* b_ptr = tmp_data;
      int w_gi_stride = ic_8 * oc_8 * 64;
      for (int gi = 0; gi < 16; ++gi) {
        float16_t* origin_C = dst_temp_data + gi * c_gi_stride;
        float16_t* origin_B = b_ptr + gi * b_gi_stride;
        const float16_t* origin_A = weight + gi * w_gi_stride;
        gemm_prepack_c8_fp16_small(
            oc_8 * 8, tile_count, ic_8 * 8, origin_A, origin_B, origin_C, ctx);
      }
      // output trans
      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int dst_x = tw_index * 2;
        int dst_y = th_index * 2;

        int ex = dst_x + 2 > wout ? wout - dst_x : 2;
        int ey = dst_y + 2 > hout ? hout - dst_y : 2;

        float16_t* dst_ptr = output + (dst_y * wout + dst_x) * 8;
        float16_t* src_ptr = dst_temp_data + ti * 8;
        if (ex == 2) {
          // trans output
          for (int ci = 0; ci < oc_8; ++ci) {
            float16_t* dst_ci = dst_ptr + ci * oc_8_stride;
            float16_t* src_ci = src_ptr + ci * tile_count * 8;
            output_trans_c8_post_2x4_fp16(src_ci,
                                          c_gi_stride,
                                          c_gi_stride * 4,
                                          trans_remain_tmp_data,
                                          8,
                                          16);
            write_to_oc8_fp16(trans_remain_tmp_data,
                              output_ptr,
                              ci * 8,
                              ci * 8 + 8,
                              dst_y,
                              dst_y + ey,
                              dst_x,
                              dst_x + ex,
                              chout,
                              hout,
                              wout,
                              flag_act,
                              local_alpha,
                              bias + ci * 8,
                              flag_bias,
                              offset,
                              threshold);
          }
        } else {
          for (int ci = 0; ci < oc_8; ++ci) {
            // trans output
            float16_t* dst_ci = dst_ptr + ci * oc_8_stride;
            float16_t* src_ci = src_ptr + ci * tile_count * 8;
            output_trans_c8_post_2x4_fp16(src_ci,
                                          c_gi_stride,
                                          c_gi_stride * 4,
                                          trans_remain_tmp_data,
                                          8,
                                          16);
            // copy to dest
            memset(trans_tmp_data, 0, 32 * sizeof(float16_t));
            for (int i = 0; i < ey; ++i) {
              memcpy(trans_tmp_data + i * ex * 8,
                     trans_remain_tmp_data + i * 16,
                     ex * sizeof(float16_t) * 8);
            }
            write_to_oc8_fp16(trans_tmp_data,
                              output_ptr,
                              ci * 8,
                              ci * 8 + 8,
                              dst_y,
                              dst_y + ey,
                              dst_x,
                              dst_x + ex,
                              chout,
                              hout,
                              wout,
                              flag_act,
                              local_alpha,
                              bias + ci * 8,
                              flag_bias,
                              offset,
                              threshold);
          }
        }
      }
    }  // for block_count
    LITE_PARALLEL_END();
  }  // for num
}  // conv_compute

void conv_compute_4x4_3x3_fp16(const float16_t* input,
                               float16_t* output,
                               int num,
                               int chout,
                               int hout,
                               int wout,
                               int chin,
                               int hin,
                               int win,
                               const float16_t* weight,
                               const float16_t* bias,
                               const operators::ConvParam& param,
                               ARMContext* ctx) {
  auto act_param = param.activation_param;
  const int pad_h0 = (*param.paddings)[0];
  const int pad_h1 = (*param.paddings)[1];
  const int pad_w0 = (*param.paddings)[2];
  const int pad_w1 = (*param.paddings)[3];
  float16_t* tmp_work_space =
      ctx->workspace_data<float16_t>() + ctx->llc_size() / sizeof(float16_t);

  int in_n_stride = chin * hin * win;
  int out_n_stride = chout * hout * wout;
  int ic_stride = win * hin;
  int oc_stride = wout * hout;
  int ic_8 = (chin + 7) / 8;   // up_div c8
  int oc_8 = (chout + 7) / 8;  // up_div c8

  int tile_w = (wout + 3) / 4;  // up_div out_w 4x4
  int tile_h = (hout + 3) / 4;  // up_div out_h 4x4
  int size_tile = tile_h * tile_w;

  int w_pad = win + pad_w0 + pad_w1;
  int h_pad = hin + pad_h0 + pad_h1;

  const int zero_len = (w_pad + 5) / 6 * 6;  // up_div in_w_pad 6x6
  float16_t zero_ptr[zero_len];              // NOLINT
  memset(zero_ptr, 0, zero_len * sizeof(float16_t));

  float16_t* input_c8 = tmp_work_space;  // input_c8 for input layout transform
  int new_h_stride = w_pad * 8;          // 8 is c8
  int new_c_stride = new_h_stride * h_pad;  // in stride w_pad*h_pad*8

  int ic_8_stride = w_pad * h_pad * 8;
  int oc_8_stride = wout * hout * 8;

  int tile_block = 8;
  int block_count = (size_tile + tile_block - 1) / tile_block;

  int threads = ctx->threads();

  float16_t* g_tmp_data = tmp_work_space + ic_8 * ic_8_stride;
  int tmp_input_thread_stride =
      tile_block * ic_8 * 288;  // 128 = 8*4*4, 8*6*6=288

  int tmp_output_thread_stride =
      tile_block * oc_8 * 288;  // 128 = 8*4*4, 8*6*6=288
  int tmp_data_thread_stride =
      tmp_input_thread_stride + tmp_output_thread_stride;
  memset(g_tmp_data, 0, threads * tmp_data_thread_stride * sizeof(float16_t));
  float16_t* g_trans_tmp_data = g_tmp_data + threads * tmp_data_thread_stride;
  float16_t* g_trans_remain_tmp_data = g_trans_tmp_data + threads * 288;
  auto act_type = act_param.active_type;
  float16_t local_alpha = 0.f;
  bool flag_bias = (bias != nullptr);
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  float16_t offset = 0.f;
  float16_t threshold = 6.f;

  if (act_param.has_active) {
    act_acquire(act_type, flag_act, local_alpha, offset, threshold, act_param);
  }

  // begin compute
  for (int ni = 0; ni < num; ++ni) {
    // trans input to c8
    for (int i = 0; i < ic_8; ++i) {
      prepack_input_nxwc8_fp16_dw(input + ni * in_n_stride,
                                  input_c8 + i * new_c_stride,
                                  i * 8,
                                  -pad_h0,
                                  hin + pad_h1,
                                  -pad_w0,
                                  win + pad_w1,
                                  chin,
                                  win,
                                  hin,
                                  zero_ptr);
    }
    float16_t* output_ptr = output + ni * out_n_stride;

    const float16_t* weight_ptr = weight;
    const float16_t* bias_ptr = bias;
    //
    LITE_PARALLEL_BEGIN(tbi, tid, block_count) {
#ifdef LITE_USE_THREAD_POOL
      float16_t* tmp_data = g_tmp_data + tid * tmp_data_thread_stride;
      float16_t* trans_tmp_data = g_trans_tmp_data + tid * 288;
      float16_t* trans_remain_tmp_data = g_trans_remain_tmp_data + tid * 288;
#elif ARM_WITH_OMP
      float16_t* tmp_data =
          g_tmp_data + omp_get_thread_num() * tmp_data_thread_stride;
      float16_t* trans_tmp_data = g_trans_tmp_data + omp_get_thread_num() * 288;
      float16_t* trans_remain_tmp_data =
          g_trans_remain_tmp_data + omp_get_thread_num() * 288;
#else
      float16_t* tmp_data = g_tmp_data;
      float16_t* trans_tmp_data = g_trans_tmp_data;
      float16_t* trans_remain_tmp_data = g_trans_remain_tmp_data;
#endif
      int tile_index = tbi * tile_block;
      int tile_remain = size_tile - tile_index;
      int tile_count = tile_remain > tile_block ? tile_block : tile_remain;
      // input trans
      int c_gi_stride = tile_count * oc_8 * 8;
      int b_gi_stride = tile_count * ic_8 * 8;
      //*
      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int src_x = tw_index * 4;
        int src_y = th_index * 4;
        int ex = src_x + 6 > w_pad ? w_pad - src_x : 6;
        int ey = src_y + 6 > h_pad ? h_pad - src_y : 6;
        float16_t* dst_ptr = tmp_data + ti * 8;
        const float16_t* src_ptr = input_c8 + (src_y * w_pad + src_x) * 8;

        if (ex == 6 && ey == 6) {
          // trans input
          for (int ci = 0; ci < ic_8; ++ci) {
            const float16_t* src_ci = src_ptr + ci * ic_8_stride;
            float16_t* dst_ci = dst_ptr + ci * tile_count * 8;
            for (int i = 0; i < 6; ++i) {
              const float16_t* ci_ptr = src_ci + i * w_pad * 8;
              input_trans_c8_6x6_fp16(ci_ptr, 8, trans_tmp_data + i * 8, 48);
            }
            for (int i = 0; i < 6; ++i) {
              input_trans_c8_6x6_fp16(trans_tmp_data + i * 48,
                                      8,
                                      dst_ci + i * b_gi_stride * 6,
                                      b_gi_stride);
            }
          }
        } else {
          // trans remain input
          int x_size = ex;
          for (int ci = 0; ci < ic_8; ++ci) {
            const float16_t* src_ci = src_ptr + ci * ic_8_stride;
            // pad
            memset(trans_remain_tmp_data, 0, 288 * sizeof(float16_t));
            if (x_size > 0) {
              for (int yi = 0; yi < ey; ++yi) {
                float16_t* dst_yi = trans_remain_tmp_data +
                                    yi * 48;  // 32=4(4x4)*c8, 48=6(6x6)*c8
                const float16_t* src_yi = src_ci + w_pad * yi * 8;
                memcpy(dst_yi, src_yi, x_size * sizeof(float16_t) * 8);
              }
            }
            // trans
            for (int i = 0; i < 6; ++i) {
              float16_t* ci_ptr = trans_remain_tmp_data + i * 48;
              input_trans_c8_6x6_fp16(ci_ptr, 8, trans_tmp_data + i * 8, 48);
            }
            float16_t* dst_ci = dst_ptr + ci * tile_count * 8;
            for (int i = 0; i < 6; ++i) {
              input_trans_c8_6x6_fp16(trans_tmp_data + i * 48,
                                      8,
                                      dst_ci + i * b_gi_stride * 6,
                                      b_gi_stride);
            }
          }  // for ci_8
        }
      }
      //*/
      // input trans end
      // begin compute
      float16_t* dst_temp_data = tmp_data + tmp_input_thread_stride;
      float16_t* b_ptr = tmp_data;
      int w_gi_stride = ic_8 * oc_8 * 64;
      for (int gi = 0; gi < 36; ++gi) {
        float16_t* origin_C = dst_temp_data + gi * c_gi_stride;
        float16_t* origin_B = b_ptr + gi * b_gi_stride;
        const float16_t* origin_A = weight + gi * w_gi_stride;
        gemm_prepack_c8_fp16_small(
            oc_8 * 8, tile_count, ic_8 * 8, origin_A, origin_B, origin_C, ctx);
      }
      //*/
      //*
      // output trans
      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int dst_x = tw_index * 4;
        int dst_y = th_index * 4;

        int ex = dst_x + 4 > wout ? wout - dst_x : 4;
        int ey = dst_y + 4 > hout ? hout - dst_y : 4;

        float16_t* dst_ptr = output + (dst_y * wout + dst_x) * 8;
        float16_t* src_ptr = dst_temp_data + ti * 8;
        if (ex == 4) {
          // trans output
          for (int ci = 0; ci < oc_8; ++ci) {
            float16_t* dst_ci = dst_ptr + ci * oc_8_stride;
            float16_t* src_ci = src_ptr + ci * tile_count * 8;
            for (int i = 0; i < 6; ++i) {
              output_trans_c8_post_4x6_fp16(src_ci + i * c_gi_stride * 6,
                                            c_gi_stride,
                                            trans_tmp_data + i * 8,
                                            48);  // 6*c8=48
            }
            for (int i = 0; i < ey; ++i) {
              output_trans_c8_post_4x6_fp16(trans_tmp_data + i * 48,
                                            8,
                                            trans_remain_tmp_data + i * 32,
                                            8);
            }
            write_to_oc8_fp16(trans_remain_tmp_data,
                              output_ptr,
                              ci * 8,
                              ci * 8 + 8,
                              dst_y,
                              dst_y + ey,
                              dst_x,
                              dst_x + ex,
                              chout,
                              hout,
                              wout,
                              flag_act,
                              local_alpha,
                              bias + ci * 8,
                              flag_bias,
                              offset,
                              threshold);
          }
        } else {
          for (int ci = 0; ci < oc_8; ++ci) {
            // trans output
            float16_t* dst_ci = dst_ptr + ci * oc_8_stride;
            float16_t* src_ci = src_ptr + ci * tile_count * 8;
            for (int i = 0; i < 6; ++i) {
              output_trans_c8_post_4x6_fp16(src_ci + i * c_gi_stride * 6,
                                            c_gi_stride,
                                            trans_tmp_data + i * 8,
                                            48);
            }
            for (int i = 0; i < ey; ++i) {
              output_trans_c8_post_4x6_fp16(trans_tmp_data + i * 48,
                                            8,  // 4(4x4)*c8=32
                                            trans_remain_tmp_data + i * 32,
                                            8);
            }
            // copy to dest
            memset(trans_tmp_data, 0, 128 * sizeof(float16_t));
            for (int i = 0; i < ey; ++i) {
              memcpy(trans_tmp_data + i * ex * 8,
                     trans_remain_tmp_data + i * 32,
                     ex * sizeof(float16_t) * 8);
            }
            write_to_oc8_fp16(trans_tmp_data,
                              output_ptr,
                              ci * 8,
                              ci * 8 + 8,
                              dst_y,
                              dst_y + ey,
                              dst_x,
                              dst_x + ex,
                              chout,
                              hout,
                              wout,
                              flag_act,
                              local_alpha,
                              bias + ci * 8,
                              flag_bias,
                              offset,
                              threshold);
          }
        }
      }
      //*/
    }  // for block_count
    LITE_PARALLEL_END();
  }  // for num
}  // conv_compute

// BT=[1, 0, -1, 0,
//    0, 1,  1, 0,
//    0, -1, 1, 0,
//    0, 1,  0, -1]
void input_trans_c8_4x4_fp16(const float16_t* src,
                             int src_stride,
                             int src_h_stride,
                             float16_t* dest,
                             int dest_stride,
                             int dest_h_stride) {
  float16x8_t src00 = vld1q_f16(src);
  float16x8_t src01 = vld1q_f16(src + src_stride);
  float16x8_t src02 = vld1q_f16(src + src_stride + src_stride);
  float16x8_t src03 = vld1q_f16(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  float16x8_t src10 = vld1q_f16(src);
  float16x8_t src11 = vld1q_f16(src + src_stride);
  float16x8_t src12 = vld1q_f16(src + src_stride + src_stride);
  float16x8_t src13 = vld1q_f16(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  float16x8_t src20 = vld1q_f16(src);
  float16x8_t src21 = vld1q_f16(src + src_stride);
  float16x8_t src22 = vld1q_f16(src + src_stride + src_stride);
  float16x8_t src23 = vld1q_f16(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  float16x8_t src30 = vld1q_f16(src);
  float16x8_t src31 = vld1q_f16(src + src_stride);
  float16x8_t src32 = vld1q_f16(src + src_stride + src_stride);
  float16x8_t src33 = vld1q_f16(src + src_stride + src_stride + src_stride);

  float16x8_t dst00 = vsubq_f16(src00, src02);
  float16x8_t dst10 = vaddq_f16(src01, src02);
  float16x8_t dst20 = vsubq_f16(src02, src01);
  float16x8_t dst30 = vsubq_f16(src01, src03);

  float16x8_t dst01 = vsubq_f16(src10, src12);
  float16x8_t dst11 = vaddq_f16(src11, src12);
  float16x8_t dst21 = vsubq_f16(src12, src11);
  float16x8_t dst31 = vsubq_f16(src11, src13);

  float16x8_t dst02 = vsubq_f16(src20, src22);
  float16x8_t dst12 = vaddq_f16(src21, src22);
  float16x8_t dst22 = vsubq_f16(src22, src21);
  float16x8_t dst32 = vsubq_f16(src21, src23);

  float16x8_t dst03 = vsubq_f16(src30, src32);
  float16x8_t dst13 = vaddq_f16(src31, src32);
  float16x8_t dst23 = vsubq_f16(src32, src31);
  float16x8_t dst33 = vsubq_f16(src31, src33);

  float16x8_t dest00 = vsubq_f16(dst00, dst02);
  float16x8_t dest10 = vaddq_f16(dst01, dst02);
  float16x8_t dest20 = vsubq_f16(dst02, dst01);
  float16x8_t dest30 = vsubq_f16(dst01, dst03);

  float16x8_t dest01 = vsubq_f16(dst10, dst12);
  float16x8_t dest11 = vaddq_f16(dst11, dst12);
  float16x8_t dest21 = vsubq_f16(dst12, dst11);
  float16x8_t dest31 = vsubq_f16(dst11, dst13);

  float16x8_t dest02 = vsubq_f16(dst20, dst22);
  float16x8_t dest12 = vaddq_f16(dst21, dst22);
  float16x8_t dest22 = vsubq_f16(dst22, dst21);
  float16x8_t dest32 = vsubq_f16(dst21, dst23);

  float16x8_t dest03 = vsubq_f16(dst30, dst32);
  float16x8_t dest13 = vaddq_f16(dst31, dst32);
  float16x8_t dest23 = vsubq_f16(dst32, dst31);
  float16x8_t dest33 = vsubq_f16(dst31, dst33);

  vst1q_f16(dest, dest00);
  vst1q_f16(dest + dest_stride, dest10);
  vst1q_f16(dest + dest_stride + dest_stride, dest20);
  vst1q_f16(dest + dest_stride + dest_stride + dest_stride, dest30);
  dest += dest_h_stride;
  vst1q_f16(dest, dest01);
  vst1q_f16(dest + dest_stride, dest11);
  vst1q_f16(dest + dest_stride + dest_stride, dest21);
  vst1q_f16(dest + dest_stride + dest_stride + dest_stride, dest31);
  dest += dest_h_stride;
  vst1q_f16(dest, dest02);
  vst1q_f16(dest + dest_stride, dest12);
  vst1q_f16(dest + dest_stride + dest_stride, dest22);
  vst1q_f16(dest + dest_stride + dest_stride + dest_stride, dest32);
  dest += dest_h_stride;
  vst1q_f16(dest, dest03);
  vst1q_f16(dest + dest_stride, dest13);
  vst1q_f16(dest + dest_stride + dest_stride, dest23);
  vst1q_f16(dest + dest_stride + dest_stride + dest_stride, dest33);
}

// BT[6][6] = {
//     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
//     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
//     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
//     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
//     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
//     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
// };
void input_trans_c8_6x6_fp16(const float16_t* src,
                             int src_stride,
                             float16_t* dest,
                             int dest_stride) {
  float16x8_t src0 = vld1q_f16(src);
  float16x8_t src1 = vld1q_f16(src + src_stride);
  float16x8_t src2 = vld1q_f16(src + src_stride * 2);
  float16x8_t src3 = vld1q_f16(src + src_stride * 3);
  float16x8_t src4 = vld1q_f16(src + src_stride * 4);
  float16x8_t src5 = vld1q_f16(src + src_stride * 5);

  float16x8_t dst0 =
      vaddq_f16(vsubq_f16(vmulq_n_f16(src0, 4), vmulq_n_f16(src2, 5)), src4);
  float16x8_t tmp1 = vsubq_f16(src4, vmulq_n_f16(src2, 4));
  float16x8_t tmp2 = vsubq_f16(vmulq_n_f16(src1, 4), src3);
  float16x8_t dst1 = vsubq_f16(tmp1, tmp2);
  float16x8_t dst2 = vaddq_f16(tmp1, tmp2);
  float16x8_t tmp3 = vsubq_f16(src4, src2);
  float16x8_t tmp4 = vmulq_n_f16(vsubq_f16(src1, src3), 2);
  float16x8_t dst3 = vsubq_f16(tmp3, tmp4);
  float16x8_t dst4 = vaddq_f16(tmp3, tmp4);
  float16x8_t dst5 =
      vaddq_f16(vsubq_f16(vmulq_n_f16(src1, 4), vmulq_n_f16(src3, 5)), src5);

  vst1q_f16(dest, dst0);
  vst1q_f16(dest + dest_stride, dst1);
  vst1q_f16(dest + dest_stride * 2, dst2);
  vst1q_f16(dest + dest_stride * 3, dst3);
  vst1q_f16(dest + dest_stride * 4, dst4);
  vst1q_f16(dest + dest_stride * 5, dst5);
}

// AT=[1, 1,  1,  0,
//    0, 1, -1, -1]
void output_trans_c8_post_2x4_fp16(const float16_t* src,
                                   int src_stride,
                                   int src_h_stride,
                                   float16_t* dest,
                                   int dest_stride,
                                   int dest_h_stride) {
  float16x8_t src00 = vld1q_f16(src);
  float16x8_t src01 = vld1q_f16(src + src_stride);
  float16x8_t src02 = vld1q_f16(src + src_stride + src_stride);
  float16x8_t src03 = vld1q_f16(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  float16x8_t src10 = vld1q_f16(src);
  float16x8_t src11 = vld1q_f16(src + src_stride);
  float16x8_t src12 = vld1q_f16(src + src_stride + src_stride);
  float16x8_t src13 = vld1q_f16(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  float16x8_t src20 = vld1q_f16(src);
  float16x8_t src21 = vld1q_f16(src + src_stride);
  float16x8_t src22 = vld1q_f16(src + src_stride + src_stride);
  float16x8_t src23 = vld1q_f16(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  float16x8_t src30 = vld1q_f16(src);
  float16x8_t src31 = vld1q_f16(src + src_stride);
  float16x8_t src32 = vld1q_f16(src + src_stride + src_stride);
  float16x8_t src33 = vld1q_f16(src + src_stride + src_stride + src_stride);

  float16x8_t dst00 = vaddq_f16(vaddq_f16(src00, src01), src02);
  float16x8_t dst10 = vsubq_f16(vsubq_f16(src01, src02), src03);
  float16x8_t dst01 = vaddq_f16(vaddq_f16(src10, src11), src12);
  float16x8_t dst11 = vsubq_f16(vsubq_f16(src11, src12), src13);
  float16x8_t dst02 = vaddq_f16(vaddq_f16(src20, src21), src22);
  float16x8_t dst12 = vsubq_f16(vsubq_f16(src21, src22), src23);
  float16x8_t dst03 = vaddq_f16(vaddq_f16(src30, src31), src32);
  float16x8_t dst13 = vsubq_f16(vsubq_f16(src31, src32), src33);

  float16x8_t dest00 = vaddq_f16(vaddq_f16(dst00, dst01), dst02);
  float16x8_t dest10 = vsubq_f16(vsubq_f16(dst01, dst02), dst03);
  float16x8_t dest01 = vaddq_f16(vaddq_f16(dst10, dst11), dst12);
  float16x8_t dest11 = vsubq_f16(vsubq_f16(dst11, dst12), dst13);

  vst1q_f16(dest, dest00);
  vst1q_f16(dest + dest_stride, dest10);
  dest += dest_h_stride;
  vst1q_f16(dest, dest01);
  vst1q_f16(dest + dest_stride, dest11);
}

/*
AT = [
    1   1   1   1   1   0
    0   1   -1  2   -2  0
    0   1   1   4   4   0
    0   1   -1  8   -8  1
]
*/
void output_trans_c8_post_4x6_fp16(const float16_t* src,
                                   int src_stride,
                                   float16_t* dest,
                                   int dest_stride) {
  const float16x8_t src0 = vld1q_f16(src);
  const float16x8_t src1 = vld1q_f16(src + src_stride);
  const float16x8_t src2 = vld1q_f16(src + src_stride * 2);
  const float16x8_t src3 = vld1q_f16(src + src_stride * 3);
  const float16x8_t src4 = vld1q_f16(src + src_stride * 4);
  const float16x8_t src5 = vld1q_f16(src + src_stride * 5);

  float16x8_t tmp12a = vaddq_f16(src1, src2);
  float16x8_t tmp12b = vsubq_f16(src1, src2);
  float16x8_t tmp34a = vaddq_f16(src3, src4);
  float16x8_t tmp34b = vsubq_f16(src3, src4);

  float16x8_t dest0 = vaddq_f16(vaddq_f16(src0, tmp12a), tmp34a);
  float16x8_t dest2 = vaddq_f16(tmp12a, vmulq_n_f16(tmp34a, 4));
  float16x8_t dest1 = vaddq_f16(tmp12b, vmulq_n_f16(tmp34b, 2));
  float16x8_t dest3 =
      vaddq_f16(vaddq_f16(tmp12b, vmulq_n_f16(tmp34b, 8)), src5);

  vst1q_f16(dest, dest0);
  vst1q_f16(dest + dest_stride, dest1);
  vst1q_f16(dest + dest_stride * 2, dest2);
  vst1q_f16(dest + dest_stride * 3, dest3);
}

void weights_trans_c8_fp16(float16_t* dest,
                           const float16_t* din,
                           const float16_t (*coeff)[3],
                           int num,
                           int ch_in,
                           int ch_out,
                           void* workspace) {
  int num_square = num * num;
  float16_t* ptr_out = static_cast<float16_t*>(workspace);
  for (int i = 0; i < ch_out; i++) {
    for (int j = 0; j < ch_in; j++) {
      const float16_t* kernel0 =
          static_cast<const float16_t*>(din) + (i * ch_in + j) * 9;
      float16_t* ptr_channel = ptr_out + (i * ch_in + j) * num_square;

      //! transform kernel, transposed
      const float16_t* k0 = kernel0;
      const float16_t* k1 = kernel0 + 3;
      const float16_t* k2 = kernel0 + 6;

      //! h
      float16_t tmp[num][3];
      for (int i = 0; i < num; i++) {
        tmp[i][0] =
            k0[0] * coeff[i][0] + k0[1] * coeff[i][1] + k0[2] * coeff[i][2];
        tmp[i][1] =
            k1[0] * coeff[i][0] + k1[1] * coeff[i][1] + k1[2] * coeff[i][2];
        tmp[i][2] =
            k2[0] * coeff[i][0] + k2[1] * coeff[i][1] + k2[2] * coeff[i][2];
      }

      //! v
      for (int j = 0; j < num; j++) {
        float16_t* tmpp = &tmp[j][0];
        for (int i = 0; i < num; i++) {
          ptr_channel[j * num + i] = tmpp[0] * coeff[i][0] +
                                     tmpp[1] * coeff[i][1] +
                                     tmpp[2] * coeff[i][2];
        }
      }
    }
  }

  int oc_pad = (ch_out + 7) / 8 * 8;
  int ic_pad = (ch_in + 7) / 8 * 8;
  int sum = ch_out * ch_in * num_square;
  int ch_in_stride = ch_in * num_square;
  int ch_in_s = ch_in_stride * 8;
  int c_stride = ic_pad * oc_pad;
  for (int i = 0; i < sum; ++i) {
    int new_c = i % num_square;
    int new_oc = i / ch_in_s;
    int new_ic = i / num_square % ch_in;
    int new_inner = i / ch_in_stride % 8;
    int dest_ind =
        new_c * c_stride + new_oc * ic_pad * 8 + new_ic * 8 + new_inner;
    dest[dest_ind] = ptr_out[i];
  }
}
// Input weight Layout: K*C*R*R (RR=3x3)
// Output weight Layout: G*G*[K/8]*[C]*8 (GG=6x6, [x] means round up to integer)
// Temp data Layout: K*C*G*G
void weight_trans_c8_6x6_fp16(float16_t* dest,
                              const float16_t* din,
                              int ch_in,
                              int ch_out,
                              void* workspace) {
  const float16_t coeff[6][3] = {{0.25f, 0.0f, 0.0f},
                                 {-1.0f / 6, -1.0f / 6, -1.0f / 6},
                                 {-1.0f / 6, 1.0f / 6, -1.0f / 6},
                                 {1.0f / 24, 1.0f / 12, 1.0f / 6},
                                 {1.0f / 24, -1.0f / 12, 1.0f / 6},
                                 {0.0f, 0.0f, 1.0f}};
  weights_trans_c8_fp16(dest, din, coeff, 6, ch_in, ch_out, workspace);
}

void weight_trans_c8_4x4_fp16(float16_t* dest,
                              const float16_t* din,
                              int ch_in,
                              int ch_out,
                              void* workspace) {
  const float16_t coeff[4][3] = {{1.0f, 0.0f, 0.0f},
                                 {0.5f, 0.5f, 0.5f},
                                 {0.5f, -0.5f, 0.5f},
                                 {0.0f, 0.0f, 1.0f}};
  weights_trans_c8_fp16(dest, din, coeff, 4, ch_in, ch_out, workspace);
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
