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

#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/backends/arm/math/packed_sgemm_c4.h"
#include "lite/core/parallel_defines.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif
#include <arm_neon.h>

namespace paddle {
namespace lite {
namespace arm {
namespace math {
void input_trans_c4_8x8(const float* src,
                        int src_stride,
                        float* dest,
                        int dest_stride);
void output_trans_c4_6x8(const float* src,
                         int src_stride,
                         float* dest,
                         int dest_stride);
void output_trans_c4_post_6x8(const float* src,
                              int src_stride,
                              float* dest,
                              int dest_stride,
                              float* bias_value,
                              bool has_relu);
void input_trans_c4_6x6(const float* src,
                        int src_stride,
                        float* dest,
                        int dest_stride);
void output_trans_c4_4x6(const float* src,
                         int src_stride,
                         float* dest,
                         int dest_stride);
void output_trans_c4_post_4x6(const float* src,
                              int src_stride,
                              float* dest,
                              int dest_stride,
                              float* bias_value,
                              bool has_relu);
void input_trans_c4_4x4(const float* src,
                        int src_stride,
                        int src_h_stride,
                        float* dest,
                        int dest_stride,
                        int dest_h_stride);
void output_trans_c4_post_2x4(const float* src,
                              int src_stride,
                              int src_h_stride,
                              float* dest,
                              int dest_stride,
                              int dest_h_stride,
                              float* bias_value,
                              bool has_relu);
void weight_trans_c4_8x8(
    float* dest, const float* src, int ic, int oc, void* workspace);
void weight_trans_c4_6x6(
    float* dest, const float* src, int ic, int oc, void* workspace);
void weight_trans_c4_4x4(
    float* dest, const float* src, int ic, int oc, void* workspace);

/*
*The following function conv_compute_6x6_3x3 and conv_compute_2x2_3x3[_small] is
*base on
*MNN[https://github.com/alibaba/MNN]
*
*Copyright © 2018, Alibaba Group Holding Limited
*/

// F(6,3)
void conv_compute_6x6_3x3(const float* input,
                          float* output,
                          int num,
                          int chout,
                          int hout,
                          int wout,
                          int chin,
                          int hin,
                          int win,
                          const float* weight,
                          const float* bias,
                          const operators::ConvParam& param,
                          ARMContext* ctx) {
  auto act_param = param.activation_param;
  const int pad_h0 = (*param.paddings)[0];
  const int pad_h1 = (*param.paddings)[1];
  const int pad_w0 = (*param.paddings)[2];
  const int pad_w1 = (*param.paddings)[3];
  float* tmp_work_space =
      ctx->workspace_data<float>() + ctx->llc_size() / sizeof(float);

  int in_n_stride = chin * hin * win;
  int out_n_stride = chout * hout * wout;
  int ic_stride = win * hin;
  int oc_stride = wout * hout;
  int ic_4 = (chin + 3) / 4;
  int oc_4 = (chout + 3) / 4;

  int tile_w = (wout + 5) / 6;
  int tile_h = (hout + 5) / 6;
  int size_tile = tile_h * tile_w;

  int w_pad = win + pad_w0 + pad_w1;
  int h_pad = hin + pad_h0 + pad_h1;

  const int zero_len = w_pad;
  float zero_ptr[zero_len];  // NOLINT
  memset(zero_ptr, 0, zero_len * sizeof(float));

  float* input_c4 = tmp_work_space;
  int new_h_stride = w_pad * 4;
  int new_c_stride = new_h_stride * h_pad;

  int ic_4_stride = w_pad * h_pad * 4;
  int oc_4_stride = wout * hout * 4;

  int tile_block = 8;
  int block_count = (size_tile + tile_block - 1) / tile_block;

  int threads = ctx->threads();
  float* g_tmp_data = tmp_work_space + ic_4 * new_c_stride;
  int tmp_data_thread_stride = tile_block * (oc_4 + ic_4) * 256;
  memset(g_tmp_data, 0, threads * tmp_data_thread_stride * sizeof(float));
  float* g_trans_tmp_data = g_tmp_data + threads * tmp_data_thread_stride;
  float* g_trans_remain_tmp_data = g_trans_tmp_data + threads * 256;

  // begin compute
  for (int ni = 0; ni < num; ++ni) {
    // trans input to c4
    for (int i = 0; i < ic_4; ++i) {
      prepack_input_nxwc4_dw(input + ni * in_n_stride,
                             input_c4 + i * new_c_stride,
                             i * 4,
                             -pad_h0,
                             hin + pad_h1,
                             -pad_w0,
                             win + pad_w1,
                             chin,
                             win,
                             hin,
                             zero_ptr);
    }
    float* output_ptr = output + ni * out_n_stride;

    const float* weight_ptr = weight;
    const float* bias_ptr = bias;
    LITE_PARALLEL_BEGIN(tbi, tid, block_count) {
#ifdef LITE_USE_THREAD_POOL
      float* tmp_data = g_tmp_data + tid * tmp_data_thread_stride;
      float* trans_tmp_data = g_trans_tmp_data + tid * 256;
      float* trans_remain_tmp_data = g_trans_remain_tmp_data + tid * 256;
#elif defined(ARM_WITH_OMP)
      float* tmp_data =
          g_tmp_data + omp_get_thread_num() * tmp_data_thread_stride;
      float* trans_tmp_data = g_trans_tmp_data + omp_get_thread_num() * 256;
      float* trans_remain_tmp_data =
          g_trans_remain_tmp_data + omp_get_thread_num() * 256;
#else
      float* tmp_data = g_tmp_data;
      float* trans_tmp_data = g_trans_tmp_data;
      float* trans_remain_tmp_data = g_trans_remain_tmp_data;
#endif
      int tile_index = tbi * tile_block;
      int tile_remain = size_tile - tile_index;
      int tile_count = tile_remain > tile_block ? tile_block : tile_remain;

      // input trans
      int c_gi_stride = tile_count * oc_4 * 4;
      int b_gi_stride = tile_count * ic_4 * 4;
      //*
      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int src_x = tw_index * 6;
        int src_y = th_index * 6;
        int ex = src_x + 8 > w_pad ? w_pad - src_x : 8;
        int ey = src_y + 8 > h_pad ? h_pad - src_y : 8;

        float* dst_ptr = tmp_data + ti * 4;
        const float* src_ptr = input_c4 + (src_y * w_pad + src_x) * 4;

        if (ex == 8 && ey == 8) {
          // trans input
          for (int ci = 0; ci < ic_4; ++ci) {
            const float* src_ci = src_ptr + ci * ic_4_stride;
            for (int i = 0; i < 8; ++i) {
              const float* ci_ptr = src_ci + i * w_pad * 4;
              input_trans_c4_8x8(ci_ptr, 4, trans_tmp_data + i * 4, 32);
            }
            float* dst_ci = dst_ptr + ci * tile_count * 4;
            for (int i = 0; i < 8; ++i) {
              input_trans_c4_8x8(trans_tmp_data + i * 32,
                                 4,
                                 dst_ci + i * b_gi_stride * 8,
                                 b_gi_stride);
            }
          }
        } else {
          // trans remain input
          int x_size = ex;
          for (int ci = 0; ci < ic_4; ++ci) {
            const float* src_ci = src_ptr + ci * ic_4_stride;
            // pad
            memset(trans_remain_tmp_data, 0, 256 * sizeof(float));
            if (x_size > 0) {
              for (int yi = 0; yi < ey; ++yi) {
                float* dst_yi = trans_remain_tmp_data + yi * 32;
                const float* src_yi = src_ci + w_pad * yi * 4;
                memcpy(dst_yi, src_yi, x_size * sizeof(float) * 4);
              }
            }
            // trans
            for (int i = 0; i < 8; ++i) {
              float* ci_ptr = trans_remain_tmp_data + i * 32;
              input_trans_c4_8x8(ci_ptr, 4, trans_tmp_data + i * 4, 32);
            }
            float* dst_ci = dst_ptr + ci * tile_count * 4;
            for (int i = 0; i < 8; ++i) {
              input_trans_c4_8x8(trans_tmp_data + i * 32,
                                 4,
                                 dst_ci + i * b_gi_stride * 8,
                                 b_gi_stride);
            }
          }  // for ci_4
        }
      }
      //*/
      // input trans end
      // *begin compute dot
      // *
      //*
      float* dst_temp_data = tmp_data + tile_block * ic_4 * 256;
      float* b_ptr = tmp_data;
      int w_gi_stride = ic_4 * oc_4 * 16;
      for (int gi = 0; gi < 64; ++gi) {
        float* origin_C = dst_temp_data + gi * c_gi_stride;
        float* origin_B = b_ptr + gi * b_gi_stride;
        const float* origin_A = weight + gi * w_gi_stride;
        sgemm_prepack_c4_small(
            oc_4 * 4, tile_count, ic_4 * 4, origin_A, origin_B, origin_C, ctx);
      }
      //*/
      //*
      // output trans
      float bias_value[4];
      memset(bias_value, 0, 4 * sizeof(float));
      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int dst_x = tw_index * 6;
        int dst_y = th_index * 6;

        int ex = dst_x + 6 > wout ? wout - dst_x : 6;
        int ey = dst_y + 6 > hout ? hout - dst_y : 6;

        float* dst_ptr = output + (dst_y * wout + dst_x) * 4;
        float* src_ptr = dst_temp_data + ti * 4;
        if (ex == 6) {
          // trans output
          for (int ci = 0; ci < oc_4; ++ci) {
            if (param.bias) {
              if (ci * 4 + 4 < chout) {
                bias_value[0] = bias[ci * 4];
                bias_value[1] = bias[ci * 4 + 1];
                bias_value[2] = bias[ci * 4 + 2];
                bias_value[3] = bias[ci * 4 + 3];
              } else {
                for (int p = 0; p < 4 && ci * 4 + p < chout; p++) {
                  bias_value[p] = bias[ci * 4 + p];
                }
              }
            }

            float* dst_ci = dst_ptr + ci * oc_4_stride;
            float* src_ci = src_ptr + ci * tile_count * 4;
            for (int i = 0; i < 8; ++i) {
              output_trans_c4_6x8(src_ci + i * c_gi_stride * 8,
                                  c_gi_stride,
                                  trans_tmp_data + i * 4,
                                  32);
            }
            for (int i = 0; i < ey; ++i) {
              output_trans_c4_post_6x8(trans_tmp_data + i * 32,
                                       4,
                                       trans_remain_tmp_data + i * 24,
                                       4,
                                       bias_value,
                                       param.fuse_relu);
            }
            write_to_output_c4_fp32(trans_remain_tmp_data,
                                    output_ptr,
                                    ci * 4,
                                    ci * 4 + 4,
                                    dst_y,
                                    dst_y + ey,
                                    dst_x,
                                    dst_x + ex,
                                    chout,
                                    hout,
                                    wout,
                                    false,
                                    zero_ptr,
                                    &act_param);
          }
        } else {
          for (int ci = 0; ci < oc_4; ++ci) {
            if (param.bias) {
              if (ci * 4 + 4 < chout) {
                bias_value[0] = bias[ci * 4];
                bias_value[1] = bias[ci * 4 + 1];
                bias_value[2] = bias[ci * 4 + 2];
                bias_value[3] = bias[ci * 4 + 3];
              } else {
                for (int p = 0; p < 4 && ci * 4 + p < chout; p++) {
                  bias_value[p] = bias[ci * 4 + p];
                }
              }
            }
            // trans output
            float* dst_ci = dst_ptr + ci * oc_4_stride;
            float* src_ci = src_ptr + ci * tile_count * 4;
            for (int i = 0; i < 8; ++i) {
              output_trans_c4_6x8(src_ci + i * c_gi_stride * 8,
                                  c_gi_stride,
                                  trans_tmp_data + i * 4,
                                  32);
            }
            for (int i = 0; i < ey; ++i) {
              output_trans_c4_post_6x8(trans_tmp_data + i * 32,
                                       4,
                                       trans_remain_tmp_data + i * 24,
                                       4,
                                       bias_value,
                                       param.fuse_relu);
            }
            // copy to dest
            memset(trans_tmp_data, 0, 144 * sizeof(float));
            for (int i = 0; i < ey; ++i) {
              memcpy(trans_tmp_data + i * ex * 4,
                     trans_remain_tmp_data + i * 24,
                     ex * sizeof(float) * 4);
            }
            write_to_output_c4_fp32(trans_tmp_data,
                                    output_ptr,
                                    ci * 4,
                                    ci * 4 + 4,
                                    dst_y,
                                    dst_y + ey,
                                    dst_x,
                                    dst_x + ex,
                                    chout,
                                    hout,
                                    wout,
                                    false,
                                    zero_ptr,
                                    &act_param);
          }
        }
      }
      //*/
    }  // for block_count
    LITE_PARALLEL_END();
  }  // for num
}  // conv_compute

// F(4,3)
void conv_compute_4x4_3x3(const float* input,
                          float* output,
                          int num,
                          int chout,
                          int hout,
                          int wout,
                          int chin,
                          int hin,
                          int win,
                          const float* weight,
                          const float* bias,
                          const operators::ConvParam& param,
                          ARMContext* ctx) {
  auto act_param = param.activation_param;
  const int pad_h0 = (*param.paddings)[0];
  const int pad_h1 = (*param.paddings)[1];
  const int pad_w0 = (*param.paddings)[2];
  const int pad_w1 = (*param.paddings)[3];
  float* tmp_work_space =
      ctx->workspace_data<float>() + ctx->llc_size() / sizeof(float);

  int in_n_stride = chin * hin * win;
  int out_n_stride = chout * hout * wout;
  int ic_stride = win * hin;
  int oc_stride = wout * hout;
  int ic_4 = (chin + 3) / 4;
  int oc_4 = (chout + 3) / 4;

  int tile_w = (wout + 3) / 4;
  int tile_h = (hout + 3) / 4;
  int size_tile = tile_h * tile_w;

  int w_pad = win + pad_w0 + pad_w1;
  int h_pad = hin + pad_h0 + pad_h1;
  const int zero_len = w_pad;
  float zero_ptr[zero_len];  // NOLINT
  memset(zero_ptr, 0, zero_len * sizeof(float));

  float* input_c4 = tmp_work_space;
  int new_h_stride = w_pad * 4;
  int new_c_stride = new_h_stride * h_pad;

  int ic_4_stride = w_pad * h_pad * 4;
  int oc_4_stride = wout * hout * 4;

  int tile_block = 8;
  int block_count = (size_tile + tile_block - 1) / tile_block;

  int threads = ctx->threads();
  float* g_tmp_data = tmp_work_space + ic_4 * new_c_stride;
  int tmp_data_thread_stride =
      tile_block * (oc_4 + ic_4) * 144;  // 64=16*4,256=64*4,144=36*4
  memset(g_tmp_data, 0, threads * tmp_data_thread_stride * sizeof(float));
  float* g_trans_tmp_data = g_tmp_data + threads * tmp_data_thread_stride;
  float* g_trans_remain_tmp_data = g_trans_tmp_data + threads * 144;

  // begin compute
  for (int ni = 0; ni < num; ++ni) {
    // trans input to c4
    for (int i = 0; i < ic_4; ++i) {
      prepack_input_nxwc4_dw(input + ni * in_n_stride,
                             input_c4 + i * new_c_stride,
                             i * 4,
                             -pad_h0,
                             hin + pad_h1,
                             -pad_w0,
                             win + pad_w1,
                             chin,
                             win,
                             hin,
                             zero_ptr);
    }
    float* output_ptr = output + ni * out_n_stride;
    const float* weight_ptr = weight;
    const float* bias_ptr = bias;
    LITE_PARALLEL_BEGIN(tbi, tid, block_count) {
#ifdef LITE_USE_THREAD_POOL
      float* tmp_data = g_tmp_data + tid * tmp_data_thread_stride;
      float* trans_tmp_data = g_trans_tmp_data + tid * 144;
      float* trans_remain_tmp_data = g_trans_remain_tmp_data + tid * 144;
#elif defined(ARM_WITH_OMP)
      float* tmp_data =
          g_tmp_data + omp_get_thread_num() * tmp_data_thread_stride;
      float* trans_tmp_data = g_trans_tmp_data + omp_get_thread_num() * 144;
      float* trans_remain_tmp_data =
          g_trans_remain_tmp_data + omp_get_thread_num() * 144;
#else
      float* tmp_data = g_tmp_data;
      float* trans_tmp_data = g_trans_tmp_data;
      float* trans_remain_tmp_data = g_trans_remain_tmp_data;
#endif
      int tile_index = tbi * tile_block;
      int tile_remain = size_tile - tile_index;
      int tile_count = tile_remain > tile_block ? tile_block : tile_remain;
      // input trans
      int c_gi_stride = tile_count * oc_4 * 4;
      int b_gi_stride = tile_count * ic_4 * 4;
      //*
      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int src_x = tw_index * 4;
        int src_y = th_index * 4;
        int ex = src_x + 6 > w_pad ? w_pad - src_x : 6;
        int ey = src_y + 6 > h_pad ? h_pad - src_y : 6;
        float* dst_ptr = tmp_data + ti * 4;
        const float* src_ptr = input_c4 + (src_y * w_pad + src_x) * 4;

        if (ex == 6 && ey == 6) {
          // trans input
          for (int ci = 0; ci < ic_4; ++ci) {
            const float* src_ci = src_ptr + ci * ic_4_stride;
            for (int i = 0; i < 6; ++i) {
              const float* ci_ptr = src_ci + i * w_pad * 4;
              input_trans_c4_6x6(ci_ptr, 4, trans_tmp_data + i * 4, 24);
            }
            float* dst_ci = dst_ptr + ci * tile_count * 4;
            for (int i = 0; i < 6; ++i) {
              input_trans_c4_6x6(trans_tmp_data + i * 24,
                                 4,
                                 dst_ci + i * b_gi_stride * 6,
                                 b_gi_stride);
            }
          }
        } else {
          // trans remain input
          int x_size = ex;
          for (int ci = 0; ci < ic_4; ++ci) {
            const float* src_ci = src_ptr + ci * ic_4_stride;
            // pad
            memset(trans_remain_tmp_data, 0, 144 * sizeof(float));
            if (x_size > 0) {
              for (int yi = 0; yi < ey; ++yi) {
                float* dst_yi = trans_remain_tmp_data + yi * 24;
                const float* src_yi = src_ci + w_pad * yi * 4;
                memcpy(dst_yi, src_yi, x_size * sizeof(float) * 4);
              }
            }
            // trans
            for (int i = 0; i < 6; ++i) {
              float* ci_ptr = trans_remain_tmp_data + i * 24;
              input_trans_c4_6x6(ci_ptr, 4, trans_tmp_data + i * 4, 24);
            }
            float* dst_ci = dst_ptr + ci * tile_count * 4;
            for (int i = 0; i < 6; ++i) {
              input_trans_c4_6x6(trans_tmp_data + i * 24,
                                 4,
                                 dst_ci + i * b_gi_stride * 6,
                                 b_gi_stride);
            }
          }  // for ci_4
        }
      }
      //*/
      // input trans end
      // *begin compute dot
      // *
      //*
      float* dst_temp_data = tmp_data + tile_block * ic_4 * 144;
      float* b_ptr = tmp_data;
      int w_gi_stride = ic_4 * oc_4 * 16;
      for (int gi = 0; gi < 36; ++gi) {
        float* origin_C = dst_temp_data + gi * c_gi_stride;
        float* origin_B = b_ptr + gi * b_gi_stride;
        const float* origin_A = weight + gi * w_gi_stride;
        sgemm_prepack_c4_small(
            oc_4 * 4, tile_count, ic_4 * 4, origin_A, origin_B, origin_C, ctx);
      }
      //*/
      //*
      // output trans
      float bias_value[4];
      memset(bias_value, 0, 4 * sizeof(float));

      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int dst_x = tw_index * 4;
        int dst_y = th_index * 4;

        int ex = dst_x + 4 > wout ? wout - dst_x : 4;
        int ey = dst_y + 4 > hout ? hout - dst_y : 4;

        float* dst_ptr = output + (dst_y * wout + dst_x) * 4;
        float* src_ptr = dst_temp_data + ti * 4;
        if (ex == 4) {
          // trans output
          for (int ci = 0; ci < oc_4; ++ci) {
            if (param.bias) {
              if (ci * 4 + 4 < chout) {
                bias_value[0] = bias[ci * 4];
                bias_value[1] = bias[ci * 4 + 1];
                bias_value[2] = bias[ci * 4 + 2];
                bias_value[3] = bias[ci * 4 + 3];
              } else {
                for (int p = 0; p < 4 && ci * 4 + p < chout; p++) {
                  bias_value[p] = bias[ci * 4 + p];
                }
              }
            }

            float* dst_ci = dst_ptr + ci * oc_4_stride;
            float* src_ci = src_ptr + ci * tile_count * 4;
            for (int i = 0; i < 6; ++i) {
              output_trans_c4_4x6(src_ci + i * c_gi_stride * 6,
                                  c_gi_stride,
                                  trans_tmp_data + i * 4,
                                  24);
            }
            for (int i = 0; i < ey; ++i) {
              output_trans_c4_post_4x6(trans_tmp_data + i * 24,
                                       4,
                                       trans_remain_tmp_data + i * 16,
                                       4,
                                       bias_value,
                                       param.fuse_relu);
            }
            write_to_output_c4_fp32(trans_remain_tmp_data,
                                    output_ptr,
                                    ci * 4,
                                    ci * 4 + 4,
                                    dst_y,
                                    dst_y + ey,
                                    dst_x,
                                    dst_x + ex,
                                    chout,
                                    hout,
                                    wout,
                                    false,
                                    zero_ptr,
                                    &act_param);
          }
        } else {
          for (int ci = 0; ci < oc_4; ++ci) {
            if (param.bias) {
              if (ci * 4 + 4 < chout) {
                bias_value[0] = bias[ci * 4];
                bias_value[1] = bias[ci * 4 + 1];
                bias_value[2] = bias[ci * 4 + 2];
                bias_value[3] = bias[ci * 4 + 3];
              } else {
                for (int p = 0; p < 4 && ci * 4 + p < chout; p++) {
                  bias_value[p] = bias[ci * 4 + p];
                }
              }
            }
            // trans output
            float* dst_ci = dst_ptr + ci * oc_4_stride;
            float* src_ci = src_ptr + ci * tile_count * 4;
            for (int i = 0; i < 6; ++i) {
              output_trans_c4_4x6(src_ci + i * c_gi_stride * 6,
                                  c_gi_stride,
                                  trans_tmp_data + i * 4,
                                  24);
            }
            for (int i = 0; i < ey; ++i) {
              output_trans_c4_post_4x6(trans_tmp_data + i * 24,
                                       4,
                                       trans_remain_tmp_data + i * 16,
                                       4,
                                       bias_value,
                                       param.fuse_relu);
            }
            // copy to dest
            memset(trans_tmp_data, 0, 64 * sizeof(float));
            for (int i = 0; i < ey; ++i) {
              memcpy(trans_tmp_data + i * ex * 4,
                     trans_remain_tmp_data + i * 16,
                     ex * sizeof(float) * 4);
            }
            write_to_output_c4_fp32(trans_tmp_data,
                                    output_ptr,
                                    ci * 4,
                                    ci * 4 + 4,
                                    dst_y,
                                    dst_y + ey,
                                    dst_x,
                                    dst_x + ex,
                                    chout,
                                    hout,
                                    wout,
                                    false,
                                    zero_ptr,
                                    &act_param);
          }
        }
      }
      //*/
    }  // for block_count
    LITE_PARALLEL_END();
  }  // for num
}  // conv_compute

// F(2,3)
void conv_compute_2x2_3x3(const float* input,
                          float* output,
                          int num,
                          int chout,
                          int hout,
                          int wout,
                          int chin,
                          int hin,
                          int win,
                          const float* weight,
                          const float* bias,
                          const operators::ConvParam& param,
                          ARMContext* ctx) {
  auto act_param = param.activation_param;
  const int pad_h0 = (*param.paddings)[0];
  const int pad_h1 = (*param.paddings)[1];
  const int pad_w0 = (*param.paddings)[2];
  const int pad_w1 = (*param.paddings)[3];
  float* tmp_work_space =
      ctx->workspace_data<float>() + ctx->llc_size() / sizeof(float);

  int in_n_stride = chin * hin * win;
  int out_n_stride = chout * hout * wout;
  int ic_stride = win * hin;
  int oc_stride = wout * hout;
  int ic_4 = (chin + 3) / 4;
  int oc_4 = (chout + 3) / 4;

  int tile_w = (wout + 1) / 2;
  int tile_h = (hout + 1) / 2;
  int size_tile = tile_h * tile_w;

  int w_pad = win + pad_w0 + pad_w1;
  int h_pad = hin + pad_h0 + pad_h1;

  const int zero_len = w_pad;
  float zero_ptr[zero_len];  // NOLINT
  memset(zero_ptr, 0, zero_len * sizeof(float));

  float* input_c4 = tmp_work_space;
  int new_h_stride = w_pad * 4;
  int new_c_stride = new_h_stride * h_pad;

  int ic_4_stride = w_pad * h_pad * 4;
  int oc_4_stride = wout * hout * 4;

  int tile_block = 8;
  int block_count = (size_tile + tile_block - 1) / tile_block;

  int threads = ctx->threads();
  float* g_tmp_data = tmp_work_space + ic_4 * new_c_stride;
  int tmp_data_thread_stride = tile_block * (oc_4 + ic_4) * 64;
  memset(g_tmp_data, 0, threads * tmp_data_thread_stride * sizeof(float));
  float* g_trans_tmp_data = g_tmp_data + threads * tmp_data_thread_stride;
  float* g_trans_remain_tmp_data = g_trans_tmp_data + threads * 64;

  // begin compute
  for (int ni = 0; ni < num; ++ni) {
    // trans input to c4
    for (int i = 0; i < ic_4; ++i) {
      prepack_input_nxwc4_dw(input + ni * in_n_stride,
                             input_c4 + i * new_c_stride,
                             i * 4,
                             -pad_h0,
                             hin + pad_h1,
                             -pad_w0,
                             win + pad_w1,
                             chin,
                             win,
                             hin,
                             zero_ptr);
    }
    float* output_ptr = output + ni * out_n_stride;

    const float* weight_ptr = weight;
    const float* bias_ptr = bias;
    LITE_PARALLEL_BEGIN(tbi, tid, block_count) {
#ifdef LITE_USE_THREAD_POOL
      float* tmp_data = g_tmp_data + tid * tmp_data_thread_stride;
      float* trans_tmp_data = g_trans_tmp_data + tid * 64;
      float* trans_remain_tmp_data = g_trans_remain_tmp_data + tid * 64;
#elif defined(ARM_WITH_OMP)
      float* tmp_data =
          g_tmp_data + omp_get_thread_num() * tmp_data_thread_stride;
      float* trans_tmp_data = g_trans_tmp_data + omp_get_thread_num() * 64;
      float* trans_remain_tmp_data =
          g_trans_remain_tmp_data + omp_get_thread_num() * 64;
#else
      float* tmp_data = g_tmp_data;
      float* trans_tmp_data = g_trans_tmp_data;
      float* trans_remain_tmp_data = g_trans_remain_tmp_data;
#endif
      int tile_index = tbi * tile_block;
      int tile_remain = size_tile - tile_index;
      int tile_count = tile_remain > tile_block ? tile_block : tile_remain;

      // input trans
      int c_gi_stride = tile_count * oc_4 * 4;
      int b_gi_stride = tile_count * ic_4 * 4;
      //*
      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int src_x = tw_index + tw_index;
        int src_y = th_index + th_index;
        int ex = src_x + 4 > w_pad ? w_pad - src_x : 4;
        int ey = src_y + 4 > h_pad ? h_pad - src_y : 4;

        float* dst_ptr = tmp_data + ti * 4;
        const float* src_ptr = input_c4 + (src_y * w_pad + src_x) * 4;

        if (ex == 4 && ey == 4) {
          // trans input
          for (int ci = 0; ci < ic_4; ++ci) {
            const float* src_ci = src_ptr + ci * ic_4_stride;
            float* dst_ci = dst_ptr + ci * tile_count * 4;
            input_trans_c4_4x4(
                src_ci, 4, w_pad * 4, dst_ci, b_gi_stride, b_gi_stride * 4);
          }
        } else {
          // trans remain input
          int x_size = ex;
          for (int ci = 0; ci < ic_4; ++ci) {
            const float* src_ci = src_ptr + ci * ic_4_stride;
            // pad
            memset(trans_remain_tmp_data, 0, 64 * sizeof(float));
            if (x_size > 0) {
              for (int yi = 0; yi < ey; ++yi) {
                float* dst_yi = trans_remain_tmp_data + yi * 16;
                const float* src_yi = src_ci + w_pad * yi * 4;
                memcpy(dst_yi, src_yi, x_size * sizeof(float) * 4);
              }
            }

            // trans
            float* dst_ci = dst_ptr + ci * tile_count * 4;
            input_trans_c4_4x4(trans_remain_tmp_data,
                               4,
                               16,
                               dst_ci,
                               b_gi_stride,
                               b_gi_stride * 4);
          }  // for ci_4
        }
      }
      //*/
      // input trans end
      // *begin compute dot
      // *
      //*
      float* dst_temp_data = tmp_data + tile_block * ic_4 * 64;
      float* b_ptr = tmp_data;
      int w_gi_stride = ic_4 * oc_4 * 16;
      for (int gi = 0; gi < 16; ++gi) {
        float* origin_C = dst_temp_data + gi * c_gi_stride;
        float* origin_B = b_ptr + gi * b_gi_stride;
        const float* origin_A = weight + gi * w_gi_stride;
        sgemm_prepack_c4_small(
            oc_4 * 4, tile_count, ic_4 * 4, origin_A, origin_B, origin_C, ctx);
      }
      //*/
      //*
      // output trans
      float bias_value[4];
      memset(bias_value, 0, 4 * sizeof(float));

      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int dst_x = tw_index * 2;
        int dst_y = th_index * 2;

        int ex = dst_x + 2 > wout ? wout - dst_x : 2;
        int ey = dst_y + 2 > hout ? hout - dst_y : 2;

        float* dst_ptr = output + (dst_y * wout + dst_x) * 4;
        float* src_ptr = dst_temp_data + ti * 4;

        if (ex == 2) {
          // trans output
          for (int ci = 0; ci < oc_4; ++ci) {
            if (param.bias) {
              if (ci * 4 + 4 < chout) {
                bias_value[0] = bias[ci * 4];
                bias_value[1] = bias[ci * 4 + 1];
                bias_value[2] = bias[ci * 4 + 2];
                bias_value[3] = bias[ci * 4 + 3];
              } else {
                for (int p = 0; p < 4 && ci * 4 + p < chout; p++) {
                  bias_value[p] = bias[ci * 4 + p];
                }
              }
            }

            float* dst_ci = dst_ptr + ci * oc_4_stride;
            float* src_ci = src_ptr + ci * tile_count * 4;
            output_trans_c4_post_2x4(src_ci,
                                     c_gi_stride,
                                     c_gi_stride * 4,
                                     trans_remain_tmp_data,
                                     4,
                                     8,
                                     bias_value,
                                     param.fuse_relu);
            write_to_output_c4_fp32(trans_remain_tmp_data,
                                    output_ptr,
                                    ci * 4,
                                    ci * 4 + 4,
                                    dst_y,
                                    dst_y + ey,
                                    dst_x,
                                    dst_x + ex,
                                    chout,
                                    hout,
                                    wout,
                                    false,
                                    zero_ptr,
                                    &act_param);
          }
        } else {
          for (int ci = 0; ci < oc_4; ++ci) {
            if (param.bias) {
              if (ci * 4 + 4 < chout) {
                bias_value[0] = bias[ci * 4];
                bias_value[1] = bias[ci * 4 + 1];
                bias_value[2] = bias[ci * 4 + 2];
                bias_value[3] = bias[ci * 4 + 3];
              } else {
                for (int p = 0; p < 4 && ci * 4 + p < chout; p++) {
                  bias_value[p] = bias[ci * 4 + p];
                }
              }
            }

            // trans output
            float* dst_ci = dst_ptr + ci * oc_4_stride;
            float* src_ci = src_ptr + ci * tile_count * 4;
            output_trans_c4_post_2x4(src_ci,
                                     c_gi_stride,
                                     c_gi_stride * 4,
                                     trans_remain_tmp_data,
                                     4,
                                     8,
                                     bias_value,
                                     param.fuse_relu);
            // copy to dest
            memset(trans_tmp_data, 0, 16 * sizeof(float));
            for (int i = 0; i < ey; ++i) {
              memcpy(trans_tmp_data + i * ex * 4,
                     trans_remain_tmp_data + i * 8,
                     ex * sizeof(float) * 4);
            }
            write_to_output_c4_fp32(trans_tmp_data,
                                    output_ptr,
                                    ci * 4,
                                    ci * 4 + 4,
                                    dst_y,
                                    dst_y + ey,
                                    dst_x,
                                    dst_x + ex,
                                    chout,
                                    hout,
                                    wout,
                                    false,
                                    zero_ptr,
                                    &act_param);
          }
        }
      }
      //*/
    }  // for block_count
    LITE_PARALLEL_END();
  }  // for num
}  // conv_compute

void conv_compute_2x2_3x3_small(const float* input,
                                float* output,
                                int num,
                                int chout,
                                int hout,
                                int wout,
                                int chin,
                                int hin,
                                int win,
                                const float* weight,
                                const float* bias,
                                const operators::ConvParam& param,
                                ARMContext* ctx) {
  auto act_param = param.activation_param;
  const int pad_h0 = (*param.paddings)[0];
  const int pad_h1 = (*param.paddings)[1];
  const int pad_w0 = (*param.paddings)[2];
  const int pad_w1 = (*param.paddings)[3];
  float* tmp_work_space =
      ctx->workspace_data<float>() + ctx->llc_size() / sizeof(float);

  int in_n_stride = chin * hin * win;
  int out_n_stride = chout * hout * wout;
  int ic_stride = win * hin;
  int oc_stride = wout * hout;
  int ic_4 = (chin + 3) / 4;
  int oc_4 = (chout + 3) / 4;

  int tile_w = (wout + 1) / 2;
  int tile_h = (hout + 1) / 2;
  int size_tile = tile_h * tile_w;

  int w_pad = win + pad_w0 + pad_w1;
  int h_pad = hin + pad_h0 + pad_h1;

  const int zero_len = w_pad;
  float zero_ptr[zero_len];  // NOLINT
  memset(zero_ptr, 0, zero_len * sizeof(float));

  float* input_c4 = tmp_work_space;
  int new_h_stride = w_pad * 4;
  int new_c_stride = new_h_stride * h_pad;

  int ic_4_stride = w_pad * h_pad * 4;
  int oc_4_stride = wout * hout * 4;

  int tile_block = 8;
  int block_count = (size_tile + tile_block - 1) / tile_block;

  int threads = ctx->threads();
  float* g_tmp_data = tmp_work_space + ic_4 * new_c_stride;
  int tmp_data_thread_stride = tile_block * (oc_4 + ic_4) * 64;
  memset(g_tmp_data, 0, tmp_data_thread_stride * sizeof(float));
  float* g_trans_tmp_data = g_tmp_data + tmp_data_thread_stride;
  float* g_trans_remain_tmp_data = g_trans_tmp_data + 64;

  // begin compute
  for (int ni = 0; ni < num; ++ni) {
    // trans input to c4

    for (int i = 0; i < ic_4; ++i) {
      prepack_input_nxwc4_dw(input + ni * in_n_stride,
                             input_c4 + i * new_c_stride,
                             i * 4,
                             -pad_h0,
                             hin + pad_h1,
                             -pad_w0,
                             win + pad_w1,
                             chin,
                             win,
                             hin,
                             zero_ptr);
    }
    float* output_ptr = output + ni * out_n_stride;

    const float* weight_ptr = weight;
    const float* bias_ptr = bias;
    for (int tbi = 0; tbi < block_count; ++tbi) {
      float* tmp_data = g_tmp_data;
      float* trans_tmp_data = g_trans_tmp_data;
      float* trans_remain_tmp_data = g_trans_remain_tmp_data;
      int tile_index = tbi * tile_block;
      int tile_remain = size_tile - tile_index;
      int tile_count = tile_remain > tile_block ? tile_block : tile_remain;

      // input trans
      int c_gi_stride = tile_count * oc_4 * 4;
      int b_gi_stride = tile_count * ic_4 * 4;
      //*
      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int src_x = tw_index + tw_index;
        int src_y = th_index + th_index;
        int ex = src_x + 4 > w_pad ? w_pad - src_x : 4;
        int ey = src_y + 4 > h_pad ? h_pad - src_y : 4;

        float* dst_ptr = tmp_data + ti * 4;
        const float* src_ptr = input_c4 + (src_y * w_pad + src_x) * 4;

        if (ex == 4 && ey == 4) {
          // trans input
          for (int ci = 0; ci < ic_4; ++ci) {
            const float* src_ci = src_ptr + ci * ic_4_stride;
            float* dst_ci = dst_ptr + ci * tile_count * 4;
            input_trans_c4_4x4(
                src_ci, 4, w_pad * 4, dst_ci, b_gi_stride, b_gi_stride * 4);
          }
        } else {
          // trans remain input
          int x_size = ex;
          for (int ci = 0; ci < ic_4; ++ci) {
            const float* src_ci = src_ptr + ci * ic_4_stride;
            // pad
            memset(trans_remain_tmp_data, 0, 64 * sizeof(float));
            if (x_size > 0) {
              for (int yi = 0; yi < ey; ++yi) {
                float* dst_yi = trans_remain_tmp_data + yi * 16;
                const float* src_yi = src_ci + w_pad * yi * 4;
                memcpy(dst_yi, src_yi, x_size * sizeof(float) * 4);
              }
            }

            float* dst_ci = dst_ptr + ci * tile_count * 4;
            input_trans_c4_4x4(trans_remain_tmp_data,
                               4,
                               16,
                               dst_ci,
                               b_gi_stride,
                               b_gi_stride * 4);
          }  // for ci_4
        }
      }
      //*/
      // input trans end
      // *begin compute dot
      // *
      //*
      float* dst_temp_data = tmp_data + tile_block * ic_4 * 64;
      float* b_ptr = tmp_data;
      int w_gi_stride = ic_4 * oc_4 * 16;
      LITE_PARALLEL_BEGIN(gi, tid, 16) {
        float* origin_C = dst_temp_data + gi * c_gi_stride;
        float* origin_B = b_ptr + gi * b_gi_stride;
        const float* origin_A = weight + gi * w_gi_stride;
        sgemm_prepack_c4_small(
            oc_4 * 4, tile_count, ic_4 * 4, origin_A, origin_B, origin_C, ctx);
      }
      LITE_PARALLEL_END();
      //*/
      //*
      // output trans
      float bias_value[4];
      memset(bias_value, 0, 4 * sizeof(float));

      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int dst_x = tw_index * 2;
        int dst_y = th_index * 2;

        int ex = dst_x + 2 > wout ? wout - dst_x : 2;
        int ey = dst_y + 2 > hout ? hout - dst_y : 2;

        float* dst_ptr = output + (dst_y * wout + dst_x) * 4;
        float* src_ptr = dst_temp_data + ti * 4;

        if (ex == 2) {
          // trans output
          for (int ci = 0; ci < oc_4; ++ci) {
            if (param.bias) {
              if (ci * 4 + 4 < chout) {
                bias_value[0] = bias[ci * 4];
                bias_value[1] = bias[ci * 4 + 1];
                bias_value[2] = bias[ci * 4 + 2];
                bias_value[3] = bias[ci * 4 + 3];
              } else {
                for (int p = 0; p < 4 && ci * 4 + p < chout; p++) {
                  bias_value[p] = bias[ci * 4 + p];
                }
              }
            }

            float* dst_ci = dst_ptr + ci * oc_4_stride;
            float* src_ci = src_ptr + ci * tile_count * 4;

            output_trans_c4_post_2x4(src_ci,
                                     c_gi_stride,
                                     c_gi_stride * 4,
                                     trans_remain_tmp_data,
                                     4,
                                     8,
                                     bias_value,
                                     param.fuse_relu);
            write_to_output_c4_fp32(trans_remain_tmp_data,
                                    output_ptr,
                                    ci * 4,
                                    ci * 4 + 4,
                                    dst_y,
                                    dst_y + ey,
                                    dst_x,
                                    dst_x + ex,
                                    chout,
                                    hout,
                                    wout,
                                    false,
                                    zero_ptr,
                                    &act_param);
          }
        } else {
          for (int ci = 0; ci < oc_4; ++ci) {
            if (param.bias) {
              if (ci * 4 + 4 < chout) {
                bias_value[0] = bias[ci * 4];
                bias_value[1] = bias[ci * 4 + 1];
                bias_value[2] = bias[ci * 4 + 2];
                bias_value[3] = bias[ci * 4 + 3];
              } else {
                for (int p = 0; p < 4 && ci * 4 + p < chout; p++) {
                  bias_value[p] = bias[ci * 4 + p];
                }
              }
            }
            // trans output
            float* dst_ci = dst_ptr + ci * oc_4_stride;
            float* src_ci = src_ptr + ci * tile_count * 4;
            output_trans_c4_post_2x4(src_ci,
                                     c_gi_stride,
                                     c_gi_stride * 4,
                                     trans_remain_tmp_data,
                                     4,
                                     8,
                                     bias_value,
                                     param.fuse_relu);
            // copy to dest
            memset(trans_tmp_data, 0, 16 * sizeof(float));
            for (int i = 0; i < ey; ++i) {
              memcpy(trans_tmp_data + i * ex * 4,
                     trans_remain_tmp_data + i * 8,
                     ex * sizeof(float) * 4);
            }
            write_to_output_c4_fp32(trans_tmp_data,
                                    output_ptr,
                                    ci * 4,
                                    ci * 4 + 4,
                                    dst_y,
                                    dst_y + ey,
                                    dst_x,
                                    dst_x + ex,
                                    chout,
                                    hout,
                                    wout,
                                    false,
                                    zero_ptr,
                                    &act_param);
          }
        }
      }
      //*/
    }  // for block_count
  }    // for num
}  // conv_compute

/*
AT = [
    1   1   1   1   1   1     1     0
    0   1   -1  2   -2  1/2   -1/2  0
    0   1   1   4   4   1/4   1/4   0
    0   1   -1  8   -8  1/8   -1/8  0
    0   1   1   16  16  1/16  1/16  0
    0   1   -1  32  -32 1/32  -1/32 0
]
*/
void output_trans_c4_6x8(const float* src,
                         int src_stride,
                         float* dest,
                         int dest_stride) {
  const float32x4_t src0 = vld1q_f32(src);
  const float32x4_t src1 = vld1q_f32(src + src_stride);
  const float32x4_t src2 = vld1q_f32(src + src_stride * 2);
  const float32x4_t src3 = vld1q_f32(src + src_stride * 3);
  const float32x4_t src4 = vld1q_f32(src + src_stride * 4);
  const float32x4_t src5 = vld1q_f32(src + src_stride * 5);
  const float32x4_t src6 = vld1q_f32(src + src_stride * 6);
  const float32x4_t src7 = vld1q_f32(src + src_stride * 7);

  float32x4_t tmp024a = vaddq_f32(src1, src2);
  float32x4_t tmp135a = vsubq_f32(src1, src2);
  float32x4_t tmp024b = vaddq_f32(src3, src4);
  float32x4_t tmp135b = vsubq_f32(src3, src4);
  float32x4_t tmp024c = vaddq_f32(src5, src6);
  float32x4_t tmp135c = vsubq_f32(src5, src6);

  float32x4_t dest0 =
      vaddq_f32(vaddq_f32(vaddq_f32(src0, tmp024a), tmp024b), tmp024c);
  float32x4_t dest2 = vaddq_f32(vaddq_f32(tmp024a, vmulq_n_f32(tmp024b, 4)),
                                vmulq_n_f32(tmp024c, 0.25f));
  float32x4_t dest4 = vaddq_f32(vaddq_f32(tmp024a, vmulq_n_f32(tmp024b, 16)),
                                vmulq_n_f32(tmp024c, 0.0625f));

  float32x4_t dest1 = vaddq_f32(vaddq_f32(tmp135a, vmulq_n_f32(tmp135b, 2)),
                                vmulq_n_f32(tmp135c, 0.5f));
  float32x4_t dest3 = vaddq_f32(vaddq_f32(tmp135a, vmulq_n_f32(tmp135b, 8)),
                                vmulq_n_f32(tmp135c, 0.125f));
  float32x4_t dest5 =
      vaddq_f32(src7,
                vaddq_f32(vaddq_f32(tmp135a, vmulq_n_f32(tmp135b, 32)),
                          vmulq_n_f32(tmp135c, 0.03125f)));

  vst1q_f32(dest, dest0);
  vst1q_f32(dest + dest_stride, dest1);
  vst1q_f32(dest + dest_stride * 2, dest2);
  vst1q_f32(dest + dest_stride * 3, dest3);
  vst1q_f32(dest + dest_stride * 4, dest4);
  vst1q_f32(dest + dest_stride * 5, dest5);
}

void output_trans_c4_post_6x8(const float* src,
                              int src_stride,
                              float* dest,
                              int dest_stride,
                              float* bias_value,
                              bool has_relu = false) {
  const float32x4_t src0 = vld1q_f32(src);
  const float32x4_t src1 = vld1q_f32(src + src_stride);
  const float32x4_t src2 = vld1q_f32(src + src_stride * 2);
  const float32x4_t src3 = vld1q_f32(src + src_stride * 3);
  const float32x4_t src4 = vld1q_f32(src + src_stride * 4);
  const float32x4_t src5 = vld1q_f32(src + src_stride * 5);
  const float32x4_t src6 = vld1q_f32(src + src_stride * 6);
  const float32x4_t src7 = vld1q_f32(src + src_stride * 7);

  float32x4_t tmp024a = vaddq_f32(src1, src2);
  float32x4_t tmp135a = vsubq_f32(src1, src2);
  float32x4_t tmp024b = vaddq_f32(src3, src4);
  float32x4_t tmp135b = vsubq_f32(src3, src4);
  float32x4_t tmp024c = vaddq_f32(src5, src6);
  float32x4_t tmp135c = vsubq_f32(src5, src6);

  float32x4_t dest0 =
      vaddq_f32(vaddq_f32(vaddq_f32(src0, tmp024a), tmp024b), tmp024c);
  float32x4_t dest2 = vaddq_f32(vaddq_f32(tmp024a, vmulq_n_f32(tmp024b, 4)),
                                vmulq_n_f32(tmp024c, 0.25f));
  float32x4_t dest4 = vaddq_f32(vaddq_f32(tmp024a, vmulq_n_f32(tmp024b, 16)),
                                vmulq_n_f32(tmp024c, 0.0625f));

  float32x4_t dest1 = vaddq_f32(vaddq_f32(tmp135a, vmulq_n_f32(tmp135b, 2)),
                                vmulq_n_f32(tmp135c, 0.5f));
  float32x4_t dest3 = vaddq_f32(vaddq_f32(tmp135a, vmulq_n_f32(tmp135b, 8)),
                                vmulq_n_f32(tmp135c, 0.125f));
  float32x4_t dest5 =
      vaddq_f32(src7,
                vaddq_f32(vaddq_f32(tmp135a, vmulq_n_f32(tmp135b, 32)),
                          vmulq_n_f32(tmp135c, 0.03125f)));

  float32x4_t bias = vld1q_f32(bias_value);
  dest0 = vaddq_f32(dest0, bias);
  dest1 = vaddq_f32(dest1, bias);
  dest2 = vaddq_f32(dest2, bias);
  dest3 = vaddq_f32(dest3, bias);
  dest4 = vaddq_f32(dest4, bias);
  dest5 = vaddq_f32(dest5, bias);

  vst1q_f32(dest, dest0);
  vst1q_f32(dest + dest_stride, dest1);
  vst1q_f32(dest + dest_stride * 2, dest2);
  vst1q_f32(dest + dest_stride * 3, dest3);
  vst1q_f32(dest + dest_stride * 4, dest4);
  vst1q_f32(dest + dest_stride * 5, dest5);
}

/*
AT = [
    1   1   1   1   1   0
    0   1   -1  2   -2  0
    0   1   1   4   4   0
    0   1   -1  8   -8  0
]
*/
void output_trans_c4_4x6(const float* src,
                         int src_stride,
                         float* dest,
                         int dest_stride) {
  const float32x4_t src0 = vld1q_f32(src);
  const float32x4_t src1 = vld1q_f32(src + src_stride);
  const float32x4_t src2 = vld1q_f32(src + src_stride * 2);
  const float32x4_t src3 = vld1q_f32(src + src_stride * 3);
  const float32x4_t src4 = vld1q_f32(src + src_stride * 4);
  const float32x4_t src5 = vld1q_f32(src + src_stride * 5);

  float32x4_t tmp02a = vaddq_f32(src1, src2);
  float32x4_t tmp13a = vsubq_f32(src1, src2);
  float32x4_t tmp02b = vaddq_f32(src3, src4);
  float32x4_t tmp13b = vsubq_f32(src3, src4);

  float32x4_t dest0 = vaddq_f32(vaddq_f32(src0, tmp02a), tmp02b);
  float32x4_t dest2 = vaddq_f32(tmp02a, vmulq_n_f32(tmp02b, 4));
  float32x4_t dest1 = vaddq_f32(tmp13a, vmulq_n_f32(tmp13b, 2));
  float32x4_t dest3 =
      vaddq_f32(vaddq_f32(tmp13a, vmulq_n_f32(tmp13b, 8)), src5);

  vst1q_f32(dest, dest0);
  vst1q_f32(dest + dest_stride, dest1);
  vst1q_f32(dest + dest_stride * 2, dest2);
  vst1q_f32(dest + dest_stride * 3, dest3);
}

void output_trans_c4_post_4x6(const float* src,
                              int src_stride,
                              float* dest,
                              int dest_stride,
                              float* bias_value,
                              bool has_relu = false) {
  const float32x4_t src0 = vld1q_f32(src);
  const float32x4_t src1 = vld1q_f32(src + src_stride);
  const float32x4_t src2 = vld1q_f32(src + src_stride * 2);
  const float32x4_t src3 = vld1q_f32(src + src_stride * 3);
  const float32x4_t src4 = vld1q_f32(src + src_stride * 4);
  const float32x4_t src5 = vld1q_f32(src + src_stride * 5);

  float32x4_t tmp02a = vaddq_f32(src1, src2);
  float32x4_t tmp13a = vsubq_f32(src1, src2);
  float32x4_t tmp02b = vaddq_f32(src3, src4);
  float32x4_t tmp13b = vsubq_f32(src3, src4);

  float32x4_t dest0 = vaddq_f32(vaddq_f32(src0, tmp02a), tmp02b);
  float32x4_t dest2 = vaddq_f32(tmp02a, vmulq_n_f32(tmp02b, 4));
  float32x4_t dest1 = vaddq_f32(tmp13a, vmulq_n_f32(tmp13b, 2));
  float32x4_t dest3 =
      vaddq_f32(vaddq_f32(tmp13a, vmulq_n_f32(tmp13b, 8)), src5);

  float32x4_t bias = vld1q_f32(bias_value);
  dest0 = vaddq_f32(dest0, bias);
  dest1 = vaddq_f32(dest1, bias);
  dest2 = vaddq_f32(dest2, bias);
  dest3 = vaddq_f32(dest3, bias);

  vst1q_f32(dest, dest0);
  vst1q_f32(dest + dest_stride, dest1);
  vst1q_f32(dest + dest_stride * 2, dest2);
  vst1q_f32(dest + dest_stride * 3, dest3);
}

/*
BT = [
   1    0     -21/4   0     -21/4     0     -1  0
   0    1     1     -17/4   -17/4     1     1   0
   0    -1    1     17/4    -17/4     -1    1   0
   0    1/2   1/4   -5/2    -5/4      2     1   0
   0    -1/2  1/4   5/2     -5/4      -2    1   0
   0    2     4   -5/2      -5        1/2   1   0
   0    -2    4     5/2     -5        -1/2  1   0
   0    -1    0     21/4    0         -21/4 0   1
]
*/
void input_trans_c4_8x8(const float* src,
                        int src_stride,
                        float* dest,
                        int dest_stride) {
  float32x4_t src0 = vld1q_f32(src);
  float32x4_t src1 = vld1q_f32(src + src_stride);
  float32x4_t src2 = vld1q_f32(src + src_stride * 2);
  float32x4_t src3 = vld1q_f32(src + src_stride * 3);
  float32x4_t src4 = vld1q_f32(src + src_stride * 4);
  float32x4_t src5 = vld1q_f32(src + src_stride * 5);
  float32x4_t src6 = vld1q_f32(src + src_stride * 6);
  float32x4_t src7 = vld1q_f32(src + src_stride * 7);

  float32x4_t dst0 = vaddq_f32(vsubq_f32(src0, src6),
                               vmulq_n_f32(vsubq_f32(src4, src2), 5.25));
  float32x4_t dst7 = vaddq_f32(vsubq_f32(src7, src1),
                               vmulq_n_f32(vsubq_f32(src3, src5), 5.25));

  float32x4_t tmp12a =
      vsubq_f32(vaddq_f32(src2, src6), vmulq_n_f32(src4, 4.25));
  float32x4_t tmp12b =
      vsubq_f32(vaddq_f32(src1, src5), vmulq_n_f32(src3, 4.25));
  float32x4_t dst1 = vaddq_f32(tmp12a, tmp12b);
  float32x4_t dst2 = vsubq_f32(tmp12a, tmp12b);

  float32x4_t tmp34a = vsubq_f32(vaddq_f32(src6, vmulq_n_f32(src2, 0.25)),
                                 vmulq_n_f32(src4, 1.25));
  float32x4_t tmp34b =
      vaddq_f32(vsubq_f32(vmulq_n_f32(src1, 0.5), vmulq_n_f32(src3, 2.5)),
                vmulq_n_f32(src5, 2));
  float32x4_t dst3 = vaddq_f32(tmp34a, tmp34b);
  float32x4_t dst4 = vsubq_f32(tmp34a, tmp34b);

  float32x4_t tmp56a =
      vaddq_f32(src6, vmulq_n_f32(vsubq_f32(src2, vmulq_n_f32(src4, 1.25)), 4));
  float32x4_t tmp56b =
      vaddq_f32(vsubq_f32(vmulq_n_f32(src1, 2), vmulq_n_f32(src3, 2.5)),
                vmulq_n_f32(src5, 0.5));
  float32x4_t dst5 = vaddq_f32(tmp56a, tmp56b);
  float32x4_t dst6 = vsubq_f32(tmp56a, tmp56b);

  vst1q_f32(dest, dst0);
  vst1q_f32(dest + dest_stride, dst1);
  vst1q_f32(dest + dest_stride * 2, dst2);
  vst1q_f32(dest + dest_stride * 3, dst3);
  vst1q_f32(dest + dest_stride * 4, dst4);
  vst1q_f32(dest + dest_stride * 5, dst5);
  vst1q_f32(dest + dest_stride * 6, dst6);
  vst1q_f32(dest + dest_stride * 7, dst7);
}

// BT[6][6] = {
//     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
//     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
//     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
//     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
//     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
//     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
// };
void input_trans_c4_6x6(const float* src,
                        int src_stride,
                        float* dest,
                        int dest_stride) {
  float32x4_t src0 = vld1q_f32(src);
  float32x4_t src1 = vld1q_f32(src + src_stride);
  float32x4_t src2 = vld1q_f32(src + src_stride * 2);
  float32x4_t src3 = vld1q_f32(src + src_stride * 3);
  float32x4_t src4 = vld1q_f32(src + src_stride * 4);
  float32x4_t src5 = vld1q_f32(src + src_stride * 5);

  float32x4_t dst0 =
      vaddq_f32(vsubq_f32(vmulq_n_f32(src0, 4), vmulq_n_f32(src2, 5)), src4);
  float32x4_t tmp1 = vsubq_f32(src4, vmulq_n_f32(src2, 4));
  float32x4_t tmp2 = vsubq_f32(vmulq_n_f32(src1, 4), src3);
  float32x4_t dst1 = vsubq_f32(tmp1, tmp2);
  float32x4_t dst2 = vaddq_f32(tmp1, tmp2);
  float32x4_t tmp3 = vsubq_f32(src4, src2);
  float32x4_t tmp4 = vmulq_n_f32(vsubq_f32(src1, src3), 2);
  float32x4_t dst3 = vsubq_f32(tmp3, tmp4);
  float32x4_t dst4 = vaddq_f32(tmp3, tmp4);
  float32x4_t dst5 =
      vaddq_f32(vsubq_f32(vmulq_n_f32(src1, 4), vmulq_n_f32(src3, 5)), src5);

  vst1q_f32(dest, dst0);
  vst1q_f32(dest + dest_stride, dst1);
  vst1q_f32(dest + dest_stride * 2, dst2);
  vst1q_f32(dest + dest_stride * 3, dst3);
  vst1q_f32(dest + dest_stride * 4, dst4);
  vst1q_f32(dest + dest_stride * 5, dst5);
}

// BT=[1, 0, -1, 0,
//    0, 1,  1, 0,
//    0, -1, 1, 0,
//    0, 1,  0, -1]
void input_trans_c4_4x4(const float* src,
                        int src_stride,
                        int src_h_stride,
                        float* dest,
                        int dest_stride,
                        int dest_h_stride) {
  float32x4_t src00 = vld1q_f32(src);
  float32x4_t src01 = vld1q_f32(src + src_stride);
  float32x4_t src02 = vld1q_f32(src + src_stride + src_stride);
  float32x4_t src03 = vld1q_f32(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  float32x4_t src10 = vld1q_f32(src);
  float32x4_t src11 = vld1q_f32(src + src_stride);
  float32x4_t src12 = vld1q_f32(src + src_stride + src_stride);
  float32x4_t src13 = vld1q_f32(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  float32x4_t src20 = vld1q_f32(src);
  float32x4_t src21 = vld1q_f32(src + src_stride);
  float32x4_t src22 = vld1q_f32(src + src_stride + src_stride);
  float32x4_t src23 = vld1q_f32(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  float32x4_t src30 = vld1q_f32(src);
  float32x4_t src31 = vld1q_f32(src + src_stride);
  float32x4_t src32 = vld1q_f32(src + src_stride + src_stride);
  float32x4_t src33 = vld1q_f32(src + src_stride + src_stride + src_stride);

  float32x4_t dst00 = vsubq_f32(src00, src02);
  float32x4_t dst10 = vaddq_f32(src01, src02);
  float32x4_t dst20 = vsubq_f32(src02, src01);
  float32x4_t dst30 = vsubq_f32(src01, src03);

  float32x4_t dst01 = vsubq_f32(src10, src12);
  float32x4_t dst11 = vaddq_f32(src11, src12);
  float32x4_t dst21 = vsubq_f32(src12, src11);
  float32x4_t dst31 = vsubq_f32(src11, src13);

  float32x4_t dst02 = vsubq_f32(src20, src22);
  float32x4_t dst12 = vaddq_f32(src21, src22);
  float32x4_t dst22 = vsubq_f32(src22, src21);
  float32x4_t dst32 = vsubq_f32(src21, src23);

  float32x4_t dst03 = vsubq_f32(src30, src32);
  float32x4_t dst13 = vaddq_f32(src31, src32);
  float32x4_t dst23 = vsubq_f32(src32, src31);
  float32x4_t dst33 = vsubq_f32(src31, src33);

  float32x4_t dest00 = vsubq_f32(dst00, dst02);
  float32x4_t dest10 = vaddq_f32(dst01, dst02);
  float32x4_t dest20 = vsubq_f32(dst02, dst01);
  float32x4_t dest30 = vsubq_f32(dst01, dst03);

  float32x4_t dest01 = vsubq_f32(dst10, dst12);
  float32x4_t dest11 = vaddq_f32(dst11, dst12);
  float32x4_t dest21 = vsubq_f32(dst12, dst11);
  float32x4_t dest31 = vsubq_f32(dst11, dst13);

  float32x4_t dest02 = vsubq_f32(dst20, dst22);
  float32x4_t dest12 = vaddq_f32(dst21, dst22);
  float32x4_t dest22 = vsubq_f32(dst22, dst21);
  float32x4_t dest32 = vsubq_f32(dst21, dst23);

  float32x4_t dest03 = vsubq_f32(dst30, dst32);
  float32x4_t dest13 = vaddq_f32(dst31, dst32);
  float32x4_t dest23 = vsubq_f32(dst32, dst31);
  float32x4_t dest33 = vsubq_f32(dst31, dst33);

  vst1q_f32(dest, dest00);
  vst1q_f32(dest + dest_stride, dest10);
  vst1q_f32(dest + dest_stride + dest_stride, dest20);
  vst1q_f32(dest + dest_stride + dest_stride + dest_stride, dest30);
  dest += dest_h_stride;
  vst1q_f32(dest, dest01);
  vst1q_f32(dest + dest_stride, dest11);
  vst1q_f32(dest + dest_stride + dest_stride, dest21);
  vst1q_f32(dest + dest_stride + dest_stride + dest_stride, dest31);
  dest += dest_h_stride;
  vst1q_f32(dest, dest02);
  vst1q_f32(dest + dest_stride, dest12);
  vst1q_f32(dest + dest_stride + dest_stride, dest22);
  vst1q_f32(dest + dest_stride + dest_stride + dest_stride, dest32);
  dest += dest_h_stride;
  vst1q_f32(dest, dest03);
  vst1q_f32(dest + dest_stride, dest13);
  vst1q_f32(dest + dest_stride + dest_stride, dest23);
  vst1q_f32(dest + dest_stride + dest_stride + dest_stride, dest33);
}

// AT=[1, 1,  1,  0,
//    0, 1, -1, -1]
void output_trans_c4_post_2x4(const float* src,
                              int src_stride,
                              int src_h_stride,
                              float* dest,
                              int dest_stride,
                              int dest_h_stride,
                              float* bias_value,
                              bool has_relu) {
  float32x4_t src00 = vld1q_f32(src);
  float32x4_t src01 = vld1q_f32(src + src_stride);
  float32x4_t src02 = vld1q_f32(src + src_stride + src_stride);
  float32x4_t src03 = vld1q_f32(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  float32x4_t src10 = vld1q_f32(src);
  float32x4_t src11 = vld1q_f32(src + src_stride);
  float32x4_t src12 = vld1q_f32(src + src_stride + src_stride);
  float32x4_t src13 = vld1q_f32(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  float32x4_t src20 = vld1q_f32(src);
  float32x4_t src21 = vld1q_f32(src + src_stride);
  float32x4_t src22 = vld1q_f32(src + src_stride + src_stride);
  float32x4_t src23 = vld1q_f32(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  float32x4_t src30 = vld1q_f32(src);
  float32x4_t src31 = vld1q_f32(src + src_stride);
  float32x4_t src32 = vld1q_f32(src + src_stride + src_stride);
  float32x4_t src33 = vld1q_f32(src + src_stride + src_stride + src_stride);

  float32x4_t dst00 = vaddq_f32(vaddq_f32(src00, src01), src02);
  float32x4_t dst10 = vsubq_f32(vsubq_f32(src01, src02), src03);
  float32x4_t dst01 = vaddq_f32(vaddq_f32(src10, src11), src12);
  float32x4_t dst11 = vsubq_f32(vsubq_f32(src11, src12), src13);
  float32x4_t dst02 = vaddq_f32(vaddq_f32(src20, src21), src22);
  float32x4_t dst12 = vsubq_f32(vsubq_f32(src21, src22), src23);
  float32x4_t dst03 = vaddq_f32(vaddq_f32(src30, src31), src32);
  float32x4_t dst13 = vsubq_f32(vsubq_f32(src31, src32), src33);

  float32x4_t dest00 = vaddq_f32(vaddq_f32(dst00, dst01), dst02);
  float32x4_t dest10 = vsubq_f32(vsubq_f32(dst01, dst02), dst03);
  float32x4_t dest01 = vaddq_f32(vaddq_f32(dst10, dst11), dst12);
  float32x4_t dest11 = vsubq_f32(vsubq_f32(dst11, dst12), dst13);

  float32x4_t bias = vld1q_f32(bias_value);
  dest00 = vaddq_f32(dest00, bias);
  dest10 = vaddq_f32(dest10, bias);
  dest01 = vaddq_f32(dest01, bias);
  dest11 = vaddq_f32(dest11, bias);

  vst1q_f32(dest, dest00);
  vst1q_f32(dest + dest_stride, dest10);
  dest += dest_h_stride;
  vst1q_f32(dest, dest01);
  vst1q_f32(dest + dest_stride, dest11);
}

void weight_trans_c4_8x8(
    float* dest, const float* din, int ch_in, int ch_out, void* workspace) {
  const float coeff[8][3] = {{1.0f, 0.0f, 0.0f},
                             {-2.0f / 9, -2.0f / 9, -2.0f / 9},
                             {-2.0f / 9, 2.0f / 9, -2.0f / 9},
                             {1.0f / 90, 1.0f / 45, 2.0f / 45},
                             {1.0f / 90, -1.0f / 45, 2.0f / 45},
                             {32.0f / 45, 16.0f / 45, 8.0f / 45},
                             {32.0f / 45, -16.0f / 45, 8.0f / 45},
                             {0.0f, 0.0f, 1.0f}};

  float* ptr_out = static_cast<float*>(workspace);

  for (int i = 0; i < ch_out; i++) {
    for (int j = 0; j < ch_in; j++) {
      const float* kernel0 =
          static_cast<const float*>(din) + (i * ch_in + j) * 9;
      float* ptr_channel = ptr_out + (i * ch_in + j) * 64;

      //! transform kernel, transposed
      const float* k0 = kernel0;
      const float* k1 = kernel0 + 3;
      const float* k2 = kernel0 + 6;

      //! h
      float tmp[8][3];
      for (int i = 0; i < 8; i++) {
        tmp[i][0] =
            k0[0] * coeff[i][0] + k0[1] * coeff[i][1] + k0[2] * coeff[i][2];
        tmp[i][1] =
            k1[0] * coeff[i][0] + k1[1] * coeff[i][1] + k1[2] * coeff[i][2];
        tmp[i][2] =
            k2[0] * coeff[i][0] + k2[1] * coeff[i][1] + k2[2] * coeff[i][2];
      }

      //! v
      for (int j = 0; j < 8; j++) {
        float* tmpp = &tmp[j][0];
        for (int i = 0; i < 8; i++) {
          ptr_channel[j * 8 + i] = tmpp[0] * coeff[i][0] +
                                   tmpp[1] * coeff[i][1] +
                                   tmpp[2] * coeff[i][2];
        }
      }
    }
  }

  int oc_pad = (ch_out + 3) / 4 * 4;
  int ic_pad = (ch_in + 3) / 4 * 4;
  int c_stride = ic_pad * oc_pad;
  for (int i = 0; i < ch_out * ch_in * 64; ++i) {
    int new_c = i % 64;
    int new_oc = i / ch_in / 64 / 4;
    int new_ic = i / 64 % ch_in;
    int new_inner = i / ch_in / 64 % 4;
    int dest_ind =
        new_c * c_stride + new_oc * ic_pad * 4 + new_ic * 4 + new_inner;
    dest[dest_ind] = ptr_out[i];
  }
}

// Input weight Layout: K*C*R*R (RR=3x3)
// Output weight Layout: G*G*[K/4]*[C]*4 (GG=6x6, [x] means round up to integer)
// Temp data Layout: K*C*G*G
void weight_trans_c4_6x6(
    float* dest, const float* din, int ch_in, int ch_out, void* workspace) {
  const float coeff[6][3] = {{0.25f, 0.0f, 0.0f},
                             {-1.0f / 6, -1.0f / 6, -1.0f / 6},
                             {-1.0f / 6, 1.0f / 6, -1.0f / 6},
                             {1.0f / 24, 1.0f / 12, 1.0f / 6},
                             {1.0f / 24, -1.0f / 12, 1.0f / 6},
                             {0.0f, 0.0f, 1.0f}};

  float* ptr_out = static_cast<float*>(workspace);

  for (int i = 0; i < ch_out; i++) {
    for (int j = 0; j < ch_in; j++) {
      const float* kernel0 =
          static_cast<const float*>(din) + (i * ch_in + j) * 9;
      float* ptr_channel = ptr_out + (i * ch_in + j) * 36;

      //! transform kernel, transposed
      const float* k0 = kernel0;
      const float* k1 = kernel0 + 3;
      const float* k2 = kernel0 + 6;

      //! h
      float tmp[6][3];
      for (int i = 0; i < 6; i++) {
        tmp[i][0] =
            k0[0] * coeff[i][0] + k0[1] * coeff[i][1] + k0[2] * coeff[i][2];
        tmp[i][1] =
            k1[0] * coeff[i][0] + k1[1] * coeff[i][1] + k1[2] * coeff[i][2];
        tmp[i][2] =
            k2[0] * coeff[i][0] + k2[1] * coeff[i][1] + k2[2] * coeff[i][2];
      }

      //! v
      for (int j = 0; j < 6; j++) {
        float* tmpp = &tmp[j][0];
        for (int i = 0; i < 6; i++) {
          ptr_channel[j * 6 + i] = tmpp[0] * coeff[i][0] +
                                   tmpp[1] * coeff[i][1] +
                                   tmpp[2] * coeff[i][2];
        }
      }
    }
  }

  int oc_pad = (ch_out + 3) / 4 * 4;
  int ic_pad = (ch_in + 3) / 4 * 4;
  int c_stride = ic_pad * oc_pad;
  for (int i = 0; i < ch_out * ch_in * 36; ++i) {
    int new_c = i % 36;
    int new_oc = i / ch_in / 36 / 4;
    int new_ic = i / 36 % ch_in;
    int new_inner = i / ch_in / 36 % 4;
    int dest_ind =
        new_c * c_stride + new_oc * ic_pad * 4 + new_ic * 4 + new_inner;
    dest[dest_ind] = ptr_out[i];
  }
}

void weight_trans_c4_4x4(
    float* dest, const float* din, int ch_in, int ch_out, void* workspace) {
  const float coeff[4][3] = {{1.0f, 0.0f, 0.0f},
                             {0.5f, 0.5f, 0.5f},
                             {0.5f, -0.5f, 0.5f},
                             {0.0f, 0.0f, 1.0f}};

  float* ptr_out = static_cast<float*>(workspace);

  for (int i = 0; i < ch_out; i++) {
    for (int j = 0; j < ch_in; j++) {
      const float* kernel0 =
          static_cast<const float*>(din) + (i * ch_in + j) * 9;
      float* ptr_channel = ptr_out + (i * ch_in + j) * 16;

      //! transform kernel, transposed
      const float* k0 = kernel0;
      const float* k1 = kernel0 + 3;
      const float* k2 = kernel0 + 6;

      //! h
      float tmp[4][3];
      for (int i = 0; i < 4; i++) {
        tmp[i][0] =
            k0[0] * coeff[i][0] + k0[1] * coeff[i][1] + k0[2] * coeff[i][2];
        tmp[i][1] =
            k1[0] * coeff[i][0] + k1[1] * coeff[i][1] + k1[2] * coeff[i][2];
        tmp[i][2] =
            k2[0] * coeff[i][0] + k2[1] * coeff[i][1] + k2[2] * coeff[i][2];
      }

      //! v
      for (int j = 0; j < 4; j++) {
        float* tmpp = &tmp[j][0];
        for (int i = 0; i < 4; i++) {
          ptr_channel[j * 4 + i] = tmpp[0] * coeff[i][0] +
                                   tmpp[1] * coeff[i][1] +
                                   tmpp[2] * coeff[i][2];
        }
      }
    }
  }

  int oc_pad = (ch_out + 3) / 4 * 4;
  int ic_pad = (ch_in + 3) / 4 * 4;
  int c_stride = ic_pad * oc_pad;
  for (int i = 0; i < ch_out * ch_in * 16; ++i) {
    int new_c = i % 16;
    int new_oc = i / ch_in / 16 / 4;
    int new_ic = i / 16 % ch_in;
    int new_inner = i / ch_in / 16 % 4;
    int dest_ind =
        new_c * c_stride + new_oc * ic_pad * 4 + new_ic * 4 + new_inner;
    dest[dest_ind] = ptr_out[i];
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
