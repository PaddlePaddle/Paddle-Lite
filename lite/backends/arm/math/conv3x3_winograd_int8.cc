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
void input_trans_c8_4x4_int8(const int8_t* src,
                             int src_stride,
                             int src_h_stride,
                             int16_t* dest,
                             int dest_stride,
                             int dest_h_stride);
void input_trans_c8_6x6_int8(const int8_t* src,
                             int src_stride,
                             int16_t* dest,
                             int dest_stride);
void input_trans_c8_post_6x6_int8(const int16_t* src,
                                  int src_stride,
                                  int16_t* dest,
                                  int dest_stride);
void output_trans_c8_post_2x4_int8(const int32_t* src,
                                   int src_stride,
                                   int src_h_stride,
                                   int32_t* dest,
                                   int dest_stride,
                                   int dest_h_stride);
void output_trans_c8_post_4x6_int8(const int32_t* src,
                                   int src_stride,
                                   int32_t* dest,
                                   int dest_stride);
void weight_trans_c8_4x4_int8(
    int16_t* dest, const int8_t* src, int ic, int oc, void* workspace);
void weight_trans_c8_6x6_int8(
    int16_t* dest, const int8_t* src, int ic, int oc, void* workspace);
// F(2,3)
template <typename Dtype>
void conv_compute_2x2_3x3_int8(const int8_t* input,
                               Dtype* output,
                               int num,
                               int chout,
                               int hout,
                               int wout,
                               int chin,
                               int hin,
                               int win,
                               const int16_t* weight,
                               const float* bias,
                               const float* scale,
                               const operators::ConvParam& param,
                               ARMContext* ctx) {
  auto act_param = param.activation_param;
  const int pad_h0 = (*param.paddings)[0];
  const int pad_h1 = (*param.paddings)[1];
  const int pad_w0 = (*param.paddings)[2];
  const int pad_w1 = (*param.paddings)[3];
  int8_t* tmp_work_space =
      ctx->workspace_data<int8_t>() + ctx->llc_size() / sizeof(int8_t);

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
  Dtype zero_ptr[zero_len];  // NOLINT
  memset(zero_ptr, 0, zero_len * sizeof(Dtype));

  int8_t* input_c8 = tmp_work_space;
  int new_h_stride = w_pad * 8;
  int new_c_stride = new_h_stride * h_pad;

  int ic_8_stride = w_pad * h_pad * 8;
  int oc_8_stride = wout * hout * 8;

  int tile_block = 8;
  int block_count = (size_tile + tile_block - 1) / tile_block;

  int threads = ctx->threads();
  int16_t* g_tmp_data =
      (int16_t*)(tmp_work_space + ic_8 * ic_8_stride +  // NOLINT
                 oc_8 * oc_8_stride * sizeof(int32_t));
  int tmp_input_thread_stride = tile_block * ic_8 * 128;
  int tmp_output_thread_stride = tile_block * oc_8 * 128;
  int tmp_data_thread_stride_size = tmp_input_thread_stride * sizeof(int16_t) +
                                    tmp_output_thread_stride * sizeof(int32_t);
  memset(g_tmp_data, 0, tmp_data_thread_stride_size);
  int8_t* g_trans_remain_tmp_data =
      (int8_t*)(g_tmp_data +  // NOLINT
                threads * (tmp_input_thread_stride +
                           tmp_output_thread_stride * sizeof(int32_t) /
                               sizeof(int16_t)));
  int32_t* g_trans_tmp_data =
      (int32_t*)(g_trans_remain_tmp_data + threads * 128);  // NOLINT
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[12] = {
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  bool flag_bias = (bias == nullptr) ? false : true;
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 4;
      for (int i = 0; i < 4; i++) {
        alpha[i] = 1.f / act_param.hard_swish_scale;
        alpha[i + 4] = act_param.hard_swish_offset;
        alpha[i + 8] = act_param.hard_swish_threshold;
      }
    }
  }
  // begin compute
  for (int ni = 0; ni < num; ++ni) {
    // trans input to c8
    for (int i = 0; i < ic_8; ++i) {
      prepack_input_nxwc8_int8_dw(input + ni * in_n_stride,
                                  input_c8 + i * new_c_stride,
                                  i * 8,
                                  -pad_h0,
                                  hin + pad_h1,
                                  -pad_w0,
                                  win + pad_w1,
                                  chin,
                                  win,
                                  hin);
    }
    int32_t* output_c8 = (int32_t*)(input_c8 + ic_8 * ic_8_stride);  // NOLINT
    Dtype* output_ptr = output + ni * out_n_stride;

    const int16_t* weight_ptr = weight;
    LITE_PARALLEL_BEGIN(tbi, tid, block_count) {
#ifdef LITE_USE_THREAD_POOL
      int16_t* tmp_data =
          g_tmp_data + tid * tmp_data_thread_stride_size / sizeof(int16_t);
      int32_t* trans_tmp_data = g_trans_tmp_data + tid * 32;
      int8_t* trans_remain_tmp_data = g_trans_remain_tmp_data + tid * 128;
#elif defined(ARM_WITH_OMP)
      int16_t* tmp_data =
          g_tmp_data +
          omp_get_thread_num() * tmp_data_thread_stride_size / sizeof(int16_t);
      int32_t* trans_tmp_data = g_trans_tmp_data + omp_get_thread_num() * 32;
      int8_t* trans_remain_tmp_data =
          g_trans_remain_tmp_data + omp_get_thread_num() * 128;
#else
      int16_t* tmp_data = g_tmp_data;
      int32_t* trans_tmp_data = g_trans_tmp_data;
      int8_t* trans_remain_tmp_data = g_trans_remain_tmp_data;
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

        int16_t* dst_ptr = tmp_data + ti * 8;
        const int8_t* src_ptr = input_c8 + (src_y * w_pad + src_x) * 8;

        if (ex == 4 && ey == 4) {
          // trans input
          for (int ci = 0; ci < ic_8; ++ci) {
            const int8_t* src_ci = src_ptr + ci * ic_8_stride;
            int16_t* dst_ci = dst_ptr + ci * tile_count * 8;
            input_trans_c8_4x4_int8(
                src_ci, 8, w_pad * 8, dst_ci, b_gi_stride, b_gi_stride * 4);
          }
        } else {
          // trans remain input
          int x_size = ex;
          for (int ci = 0; ci < ic_8; ++ci) {
            const int8_t* src_ci = src_ptr + ci * ic_8_stride;
            // pad
            memset(trans_remain_tmp_data, 0, 128 * sizeof(int8_t));
            if (x_size > 0) {
              for (int yi = 0; yi < ey; ++yi) {
                int8_t* dst_yi = trans_remain_tmp_data + yi * 32;
                const int8_t* src_yi = src_ci + w_pad * yi * 8;
                memcpy(dst_yi, src_yi, x_size * sizeof(int8_t) * 8);
              }
            }

            // trans
            int16_t* dst_ci = dst_ptr + ci * tile_count * 8;
            input_trans_c8_4x4_int8(trans_remain_tmp_data,
                                    8,
                                    32,
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
      int32_t* dst_temp_data =
          (int32_t*)(tmp_data + tmp_input_thread_stride);  // NOLINT
      int16_t* b_ptr = tmp_data;
      int w_gi_stride = ic_8 * oc_8 * 64;
      for (int gi = 0; gi < 16; ++gi) {
        int32_t* origin_C = dst_temp_data + gi * c_gi_stride;
        int16_t* origin_B = b_ptr + gi * b_gi_stride;
        const int16_t* origin_A = weight + gi * w_gi_stride;
        sgemm_prepack_c8_int16_small(
            oc_8 * 8, tile_count, ic_8 * 8, origin_A, origin_B, origin_C, ctx);
      }
      //*/
      //*
      // output trans
      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index / tile_w;

        int dst_x = tw_index * 2;
        int dst_y = th_index * 2;

        int ex = dst_x + 2 > wout ? wout - dst_x : 2;
        int ey = dst_y + 2 > hout ? hout - dst_y : 2;

        int32_t* src_ptr = dst_temp_data + ti * 8;
        int32_t* trans_remain_tmp_i32_data =
            (int32_t*)(trans_remain_tmp_data);  // NOLINT
        int32_t* dst_ptr = output_c8 + (dst_y * wout + dst_x) * 8;

        if (ex == 2 && ey == 2) {
          // trans output
          for (int ci = 0; ci < oc_8; ++ci) {
            int cur_ind = ci * 8;

            int32_t* src_ci = src_ptr + ci * tile_count * 8;
            int32_t* dst_ci = dst_ptr + ci * oc_8_stride;
            output_trans_c8_post_2x4_int8(
                src_ci, c_gi_stride, c_gi_stride * 4, dst_ci, 8, wout * 8);
          }
        } else {
          for (int ci = 0; ci < oc_8; ++ci) {
            int cur_ind = ci * 8;
            // trans output
            int32_t* src_ci = src_ptr + ci * tile_count * 8;
            output_trans_c8_post_2x4_int8(src_ci,
                                          c_gi_stride,
                                          c_gi_stride * 4,
                                          trans_remain_tmp_i32_data,
                                          8,
                                          16);
            // copy to dest
            int32_t* dst_ci = dst_ptr + ci * oc_8_stride;
            for (int i = 0; i < ey; ++i) {
              memcpy(dst_ci + i * wout * 8,
                     trans_remain_tmp_i32_data + i * 16,
                     ex * sizeof(int32_t) * 8);
            }
          }
        }
      }
      //*/
    }  // for block_count
    LITE_PARALLEL_END();
    for (int ci = 0; ci < oc_8; ++ci) {
      write_int32_nchwc8_to_nchw(output_c8 + ci * oc_8_stride,
                                 output_ptr,
                                 ci * 8,
                                 ci * 8 + 8,
                                 0,
                                 hout,
                                 0,
                                 wout,
                                 chout,
                                 hout,
                                 wout,
                                 flag_act,
                                 alpha,
                                 bias + ci * 8,
                                 flag_bias,
                                 zero_ptr,
                                 scale + ci * 8);
    }
  }  // for num
}  // conv compute
template void conv_compute_2x2_3x3_int8<int8_t>(
    const int8_t* input,
    int8_t* output,
    int num,
    int chout,
    int hout,
    int wout,
    int chin,
    int hin,
    int win,
    const int16_t* weight,
    const float* bias,
    const float* scale,
    const operators::ConvParam& param,
    ARMContext* ctx);
template void conv_compute_2x2_3x3_int8<float>(
    const int8_t* input,
    float* output,
    int num,
    int chout,
    int hout,
    int wout,
    int chin,
    int hin,
    int win,
    const int16_t* weight,
    const float* bias,
    const float* scale,
    const operators::ConvParam& param,
    ARMContext* ctx);

template <typename Dtype>
void conv_compute_4x4_3x3_int8(const int8_t* input,
                               Dtype* output,
                               int num,
                               int chout,
                               int hout,
                               int wout,
                               int chin,
                               int hin,
                               int win,
                               const int16_t* weight,
                               const float* bias,
                               const float* scale,
                               const operators::ConvParam& param,
                               ARMContext* ctx) {
  auto act_param = param.activation_param;
  const int pad_h0 = (*param.paddings)[0];
  const int pad_h1 = (*param.paddings)[1];
  const int pad_w0 = (*param.paddings)[2];
  const int pad_w1 = (*param.paddings)[3];
  int8_t* tmp_work_space =
      ctx->workspace_data<int8_t>() + ctx->llc_size() / sizeof(int8_t);

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
  Dtype zero_ptr[zero_len];                  // NOLINT
  memset(zero_ptr, 0, zero_len * sizeof(Dtype));

  int8_t* input_c8 = tmp_work_space;  // input_c8 for input layout transform
  int new_h_stride = w_pad * 8;       // 8 is c8
  int new_c_stride = new_h_stride * h_pad;  // in stride w_pad*h_pad*8

  int ic_8_stride = w_pad * h_pad * 8;
  int oc_8_stride = wout * hout * 8;

  int tile_block = 8;
  int block_count = (size_tile + tile_block - 1) / tile_block;

  int threads = ctx->threads();
  int16_t* g_tmp_data = reinterpret_cast<int16_t*>(
      tmp_work_space + ic_8 * ic_8_stride * sizeof(int8_t) +  // NOLINT
      oc_8 * oc_8_stride * sizeof(int32_t));
  int tmp_input_thread_stride =
      tile_block * ic_8 * 288;  // 128 = 8*4*4, 8*6*6=288
  // tmp_output_thread_stride is batched gemm result
  int tmp_output_thread_stride =
      tile_block * oc_8 * 288;  // 128 = 8*4*4, 8*6*6=288
  int tmp_data_thread_stride_size = tmp_input_thread_stride * sizeof(int16_t) +
                                    tmp_output_thread_stride * sizeof(int32_t);
  memset(g_tmp_data, 0, tmp_data_thread_stride_size);
  int16_t* g_trans_tmp_data =
      (int16_t*)(g_tmp_data +  // NOLINT
                 threads * (tmp_input_thread_stride * sizeof(int16_t) +
                            tmp_output_thread_stride * sizeof(int32_t)) /
                     sizeof(int16_t));
  // 128=4*4*c8, 288=6*6*c8
  int8_t* g_trans_remain_tmp_data =
      (int8_t*)(g_trans_tmp_data + threads * 288);  // NOLINT
  int32_t* g_trans_tmp_output_data = reinterpret_cast<int32_t*>(
      g_trans_remain_tmp_data + threads * 288);  // 4x4x8=128
  int32_t* g_trans_remain_tmp_output_data =
      g_trans_tmp_output_data + threads * 192;
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[12] = {
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  bool flag_bias = (bias == nullptr) ? false : true;
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 4;
      for (int i = 0; i < 4; i++) {
        alpha[i] = 1.f / act_param.hard_swish_scale;
        alpha[i + 4] = act_param.hard_swish_offset;
        alpha[i + 8] = act_param.hard_swish_threshold;
      }
    }
  }
  // begin compute
  for (int ni = 0; ni < num; ++ni) {
    // trans input to c8
    for (int i = 0; i < ic_8; ++i) {
      prepack_input_nxwc8_int8_dw(input + ni * in_n_stride,
                                  input_c8 + i * new_c_stride,
                                  i * 8,
                                  -pad_h0,
                                  hin + pad_h1,
                                  -pad_w0,
                                  win + pad_w1,
                                  chin,
                                  win,
                                  hin);
    }
    int32_t* output_c8 = (int32_t*)(input_c8 + ic_8 * ic_8_stride);  // NOLINT
    Dtype* output_ptr = output + ni * out_n_stride;
    const int16_t* weight_ptr = weight;
    LITE_PARALLEL_BEGIN(tbi, tid, block_count) {
#ifdef LITE_USE_THREAD_POOL
      int16_t* tmp_data =
          g_tmp_data + tid * tmp_data_thread_stride_size / sizeof(int16_t);
      int16_t* trans_tmp_data = g_trans_tmp_data + tid * 288;
      int8_t* trans_remain_tmp_data = g_trans_remain_tmp_data + tid * 288;
      int32_t* trans_tmp_output_data = g_trans_tmp_output_data + tid * 192;
      int32_t* trans_remain_tmp_output_data =
          g_trans_remain_tmp_output_data + tid * 128;
#elif defined(ARM_WITH_OMP)
      int16_t* tmp_data =
          g_tmp_data +
          omp_get_thread_num() * tmp_data_thread_stride_size / sizeof(int16_t);
      int16_t* trans_tmp_data = g_trans_tmp_data + omp_get_thread_num() * 288;
      int8_t* trans_remain_tmp_data =
          g_trans_remain_tmp_data + omp_get_thread_num() * 288;
      int32_t* trans_tmp_output_data =
          g_trans_tmp_output_data + omp_get_thread_num() * 192;
      int32_t* trans_remain_tmp_output_data =
          g_trans_remain_tmp_output_data + omp_get_thread_num() * 128;
#else
      int16_t* tmp_data = g_tmp_data;
      int16_t* trans_tmp_data = g_trans_tmp_data;
      int8_t* trans_remain_tmp_data = g_trans_remain_tmp_data;
      int32_t* trans_tmp_output_data = g_trans_tmp_output_data;
      int32_t* trans_remain_tmp_output_data = g_trans_remain_tmp_output_data;
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

        int16_t* dst_ptr = tmp_data + ti * 8;  // 8 is c8
        const int8_t* src_ptr = input_c8 + (src_y * w_pad + src_x) * 8;

        if (ex == 6 && ey == 6) {
          // trans input
          for (int ci = 0; ci < ic_8; ++ci) {
            const int8_t* src_ci = src_ptr + ci * ic_8_stride;
            int16_t* dst_ci = dst_ptr + ci * tile_count * 8;
            for (int i = 0; i < 6; ++i) {
              const int8_t* ci_ptr = src_ci + i * w_pad * 8;
              input_trans_c8_6x6_int8(ci_ptr, 8, trans_tmp_data + i * 8, 48);
            }
            for (int i = 0; i < 6; ++i) {
              input_trans_c8_post_6x6_int8(trans_tmp_data + i * 48,
                                           8,
                                           dst_ci + i * b_gi_stride * 6,
                                           b_gi_stride);
            }
          }
        } else {
          // trans remain input
          int x_size = ex;
          for (int ci = 0; ci < ic_8; ++ci) {
            const int8_t* src_ci = src_ptr + ci * ic_8_stride;
            // pad
            memset(
                trans_remain_tmp_data, 0, 288 * sizeof(int8_t));  // 288=6*6*c8
            if (x_size > 0) {
              for (int yi = 0; yi < ey; ++yi) {
                int8_t* dst_yi = trans_remain_tmp_data +
                                 yi * 48;  // 32=4(4x4)*c8, 48=6(6x6)*c8
                const int8_t* src_yi = src_ci + w_pad * yi * 8;
                memcpy(dst_yi, src_yi, x_size * sizeof(int8_t) * 8);
              }
            }

            // trans
            for (int i = 0; i < 6; ++i) {
              int8_t* ci_ptr = trans_remain_tmp_data + i * 48;
              input_trans_c8_6x6_int8(ci_ptr, 8, trans_tmp_data + i * 8, 48);
            }
            int16_t* dst_ci = dst_ptr + ci * tile_count * 8;
            for (int i = 0; i < 6; ++i) {
              input_trans_c8_post_6x6_int8(trans_tmp_data + i * 48,
                                           8,
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
      // dst_temp_data layout size: tile_block * oc_8 * 6*6 * c8
      int32_t* dst_temp_data =
          (int32_t*)(tmp_data + tmp_input_thread_stride);  // NOLINT
      // b_ptr layout size: tile_block * ic_8 * 6*6 * c8
      int16_t* b_ptr = tmp_data;
      int w_gi_stride = ic_8 * oc_8 * 64;  // 64=c8*c8
      for (int gi = 0; gi < 36; ++gi) {
        int32_t* origin_C = dst_temp_data + gi * c_gi_stride;
        int16_t* origin_B = b_ptr + gi * b_gi_stride;
        const int16_t* origin_A = weight + gi * w_gi_stride;
        int col_idx = gi / 6;
        int row_idx = gi % 6;
        if (col_idx == 5 || row_idx == 5) {
          sgemm_prepack_c8_int16_small(oc_8 * 8,
                                       tile_count,
                                       ic_8 * 8,
                                       origin_A,
                                       origin_B,
                                       origin_C,
                                       ctx,
                                       24);
        } else {
          sgemm_prepack_c8_int16_small(oc_8 * 8,
                                       tile_count,
                                       ic_8 * 8,
                                       origin_A,
                                       origin_B,
                                       origin_C,
                                       ctx);
        }
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

        int32_t* src_ptr = dst_temp_data + ti * 8;
        int32_t* dst_ptr = output_c8 + (dst_y * wout + dst_x) * 8;
        if (ex == 4) {
          // trans output
          for (int ci = 0; ci < oc_8; ++ci) {
            int32_t* src_ci = src_ptr + ci * tile_count * 8;
            int32_t* dst_ci = dst_ptr + ci * oc_8_stride;
            for (int i = 0; i < 6; ++i) {
              output_trans_c8_post_4x6_int8(src_ci + i * c_gi_stride * 6,
                                            c_gi_stride,
                                            trans_tmp_output_data + i * 8,
                                            48);  // 6*c8=48
            }
            for (int i = 0; i < ey; ++i) {
              output_trans_c8_post_4x6_int8(
                  trans_tmp_output_data + i * 48, 8, dst_ci + i * wout * 8, 8);
            }
          }
        } else {
          for (int ci = 0; ci < oc_8; ++ci) {
            // trans output
            int32_t* src_ci = src_ptr + ci * tile_count * 8;
            for (int i = 0; i < 6; ++i) {
              output_trans_c8_post_4x6_int8(src_ci + i * c_gi_stride * 6,
                                            c_gi_stride,
                                            trans_tmp_output_data + i * 8,
                                            48);
            }
            for (int i = 0; i < ey; ++i) {
              output_trans_c8_post_4x6_int8(
                  trans_tmp_output_data + i * 48,
                  8,  // 4(4x4)*c8=32
                  trans_remain_tmp_output_data + i * 32,
                  8);
            }
            // copy to dest
            int32_t* dst_ci = dst_ptr + ci * oc_8_stride;
            for (int i = 0; i < ey; ++i) {
              memcpy(dst_ci + i * wout * 8,                  // 8 is c8
                     trans_remain_tmp_output_data + i * 32,  // 32=c8*4(4x4)
                     ex * sizeof(int32_t) * 8);  // 4*sizeof(int32_t) * 8*4=512
            }
          }
        }
      }
      //*/
    }  // for block_count
    LITE_PARALLEL_END();
    for (int ci = 0; ci < oc_8; ++ci) {
      write_int32_nchwc8_to_nchw(output_c8 + ci * oc_8_stride,
                                 output_ptr,
                                 ci * 8,
                                 ci * 8 + 8,
                                 0,
                                 hout,
                                 0,
                                 wout,
                                 chout,
                                 hout,
                                 wout,
                                 flag_act,
                                 alpha,
                                 bias + ci * 8,
                                 flag_bias,
                                 zero_ptr,
                                 scale + ci * 8);
    }
  }  // for num
}  // conv compute
template void conv_compute_4x4_3x3_int8<int8_t>(
    const int8_t* input,
    int8_t* output,
    int num,
    int chout,
    int hout,
    int wout,
    int chin,
    int hin,
    int win,
    const int16_t* weight,
    const float* bias,
    const float* scale,
    const operators::ConvParam& param,
    ARMContext* ctx);
template void conv_compute_4x4_3x3_int8<float>(
    const int8_t* input,
    float* output,
    int num,
    int chout,
    int hout,
    int wout,
    int chin,
    int hin,
    int win,
    const int16_t* weight,
    const float* bias,
    const float* scale,
    const operators::ConvParam& param,
    ARMContext* ctx);

// BT=[1, 0, -1, 0,
//    0, 1,  1, 0,
//    0, -1, 1, 0,
//    0, 1,  0, -1]
void input_trans_c8_4x4_int8(const int8_t* src,
                             int src_stride,
                             int src_h_stride,
                             int16_t* dest,
                             int dest_stride,
                             int dest_h_stride) {
  int8x8_t src00 = vld1_s8(src);
  int8x8_t src01 = vld1_s8(src + src_stride);
  int8x8_t src02 = vld1_s8(src + src_stride + src_stride);
  int8x8_t src03 = vld1_s8(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  int8x8_t src10 = vld1_s8(src);
  int8x8_t src11 = vld1_s8(src + src_stride);
  int8x8_t src12 = vld1_s8(src + src_stride + src_stride);
  int8x8_t src13 = vld1_s8(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  int8x8_t src20 = vld1_s8(src);
  int8x8_t src21 = vld1_s8(src + src_stride);
  int8x8_t src22 = vld1_s8(src + src_stride + src_stride);
  int8x8_t src23 = vld1_s8(src + src_stride + src_stride + src_stride);
  src += src_h_stride;
  int8x8_t src30 = vld1_s8(src);
  int8x8_t src31 = vld1_s8(src + src_stride);
  int8x8_t src32 = vld1_s8(src + src_stride + src_stride);
  int8x8_t src33 = vld1_s8(src + src_stride + src_stride + src_stride);

  int16x8_t dst00 = vsubl_s8(src00, src02);
  int16x8_t dst10 = vaddl_s8(src01, src02);
  int16x8_t dst20 = vsubl_s8(src02, src01);
  int16x8_t dst30 = vsubl_s8(src01, src03);

  int16x8_t dst01 = vsubl_s8(src10, src12);
  int16x8_t dst11 = vaddl_s8(src11, src12);
  int16x8_t dst21 = vsubl_s8(src12, src11);
  int16x8_t dst31 = vsubl_s8(src11, src13);

  int16x8_t dst02 = vsubl_s8(src20, src22);
  int16x8_t dst12 = vaddl_s8(src21, src22);
  int16x8_t dst22 = vsubl_s8(src22, src21);
  int16x8_t dst32 = vsubl_s8(src21, src23);

  int16x8_t dst03 = vsubl_s8(src30, src32);
  int16x8_t dst13 = vaddl_s8(src31, src32);
  int16x8_t dst23 = vsubl_s8(src32, src31);
  int16x8_t dst33 = vsubl_s8(src31, src33);

  int16x8_t dest00 = vsubq_s16(dst00, dst02);
  int16x8_t dest10 = vaddq_s16(dst01, dst02);
  int16x8_t dest20 = vsubq_s16(dst02, dst01);
  int16x8_t dest30 = vsubq_s16(dst01, dst03);

  int16x8_t dest01 = vsubq_s16(dst10, dst12);
  int16x8_t dest11 = vaddq_s16(dst11, dst12);
  int16x8_t dest21 = vsubq_s16(dst12, dst11);
  int16x8_t dest31 = vsubq_s16(dst11, dst13);

  int16x8_t dest02 = vsubq_s16(dst20, dst22);
  int16x8_t dest12 = vaddq_s16(dst21, dst22);
  int16x8_t dest22 = vsubq_s16(dst22, dst21);
  int16x8_t dest32 = vsubq_s16(dst21, dst23);

  int16x8_t dest03 = vsubq_s16(dst30, dst32);
  int16x8_t dest13 = vaddq_s16(dst31, dst32);
  int16x8_t dest23 = vsubq_s16(dst32, dst31);
  int16x8_t dest33 = vsubq_s16(dst31, dst33);

  vst1q_s16(dest, dest00);
  vst1q_s16(dest + dest_stride, dest10);
  vst1q_s16(dest + dest_stride + dest_stride, dest20);
  vst1q_s16(dest + dest_stride + dest_stride + dest_stride, dest30);
  dest += dest_h_stride;
  vst1q_s16(dest, dest01);
  vst1q_s16(dest + dest_stride, dest11);
  vst1q_s16(dest + dest_stride + dest_stride, dest21);
  vst1q_s16(dest + dest_stride + dest_stride + dest_stride, dest31);
  dest += dest_h_stride;
  vst1q_s16(dest, dest02);
  vst1q_s16(dest + dest_stride, dest12);
  vst1q_s16(dest + dest_stride + dest_stride, dest22);
  vst1q_s16(dest + dest_stride + dest_stride + dest_stride, dest32);
  dest += dest_h_stride;
  vst1q_s16(dest, dest03);
  vst1q_s16(dest + dest_stride, dest13);
  vst1q_s16(dest + dest_stride + dest_stride, dest23);
  vst1q_s16(dest + dest_stride + dest_stride + dest_stride, dest33);
}

// BT[6][6] = {
//     {4, 0, -5, 0, 1, 0},
//     {0,-4, -4, 1, 1, 0},
//     {0, 4, -4,-1, 1, 0},
//     {0,-2, -1, 2, 1, 0},
//     {0, 2, -1,-2, 1, 0},
//     {0, 4,  0,-5, 0, 1}
// };
void input_trans_c8_6x6_int8(const int8_t* src,
                             int src_stride,
                             int16_t* dest,
                             int dest_stride) {
  int8x8_t src0 = vld1_s8(src);
  int8x8_t src1 = vld1_s8(src + src_stride);
  int8x8_t src2 = vld1_s8(src + src_stride * 2);
  int8x8_t src3 = vld1_s8(src + src_stride * 3);
  int8x8_t src4 = vld1_s8(src + src_stride * 4);
  int8x8_t src5 = vld1_s8(src + src_stride * 5);

  int16x8_t dst0 =
      vaddq_s16(vshlq_n_s16(vsubl_s8(src0, src2), 2), vsubl_s8(src4, src2));
  int16x8_t dst1 =
      vsubq_s16(vaddl_s8(src3, src4), vshlq_n_s16(vaddl_s8(src1, src2), 2));
  int16x8_t dst2 =
      vaddq_s16(vsubl_s8(src4, src3), vshlq_n_s16(vsubl_s8(src1, src2), 2));
  int16x8_t dst3 =
      vaddq_s16(vsubl_s8(src4, src2), vshlq_n_s16(vsubl_s8(src3, src1), 1));
  int16x8_t dst4 =
      vaddq_s16(vsubl_s8(src4, src2), vshlq_n_s16(vsubl_s8(src1, src3), 1));
  int16x8_t dst5 =
      vaddq_s16(vshlq_n_s16(vsubl_s8(src1, src3), 2), vsubl_s8(src5, src3));
  vst1q_s16(dest, dst0);
  vst1q_s16(dest + dest_stride, dst1);
  vst1q_s16(dest + dest_stride * 2, dst2);
  vst1q_s16(dest + dest_stride * 3, dst3);
  vst1q_s16(dest + dest_stride * 4, dst4);
  vst1q_s16(dest + dest_stride * 5, dst5);
}

// BT[6][6] = {
//     {4, 0, -5, 0, 1, 0},
//     {0,-4, -4, 1, 1, 0},
//     {0, 4, -4,-1, 1, 0},
//     {0,-2, -1, 2, 1, 0},
//     {0, 2, -1,-2, 1, 0},
//     {0, 4,  0,-5, 0, 1}
// };
void input_trans_c8_post_6x6_int8(const int16_t* src,
                                  int src_stride,
                                  int16_t* dest,
                                  int dest_stride) {
  int16x8_t src0 = vld1q_s16(src);
  int16x8_t src1 = vld1q_s16(src + src_stride);
  int16x8_t src2 = vld1q_s16(src + src_stride * 2);
  int16x8_t src3 = vld1q_s16(src + src_stride * 3);
  int16x8_t src4 = vld1q_s16(src + src_stride * 4);
  int16x8_t src5 = vld1q_s16(src + src_stride * 5);

  int16x8_t dst0 =
      vaddq_s16(vshlq_n_s16(vsubq_s16(src0, src2), 2), vsubq_s16(src4, src2));
  int16x8_t dst1 =
      vsubq_s16(vaddq_s16(src3, src4), vshlq_n_s16(vaddq_s16(src1, src2), 2));
  int16x8_t dst2 =
      vaddq_s16(vsubq_s16(src4, src3), vshlq_n_s16(vsubq_s16(src1, src2), 2));
  int16x8_t dst3 =
      vaddq_s16(vsubq_s16(src4, src2), vshlq_n_s16(vsubq_s16(src3, src1), 1));
  int16x8_t dst4 =
      vaddq_s16(vsubq_s16(src4, src2), vshlq_n_s16(vsubq_s16(src1, src3), 1));
  int16x8_t dst5 =
      vaddq_s16(vshlq_n_s16(vsubq_s16(src1, src3), 2), vsubq_s16(src5, src3));
  vst1q_s16(dest, dst0);
  vst1q_s16(dest + dest_stride, dst1);
  vst1q_s16(dest + dest_stride * 2, dst2);
  vst1q_s16(dest + dest_stride * 3, dst3);
  vst1q_s16(dest + dest_stride * 4, dst4);
  vst1q_s16(dest + dest_stride * 5, dst5);
}

// AT=[1, 1,  1,  0,
//    0, 1, -1, -1]
void output_trans_c8_post_2x4_int8(const int32_t* src,
                                   int src_stride,
                                   int src_h_stride,
                                   int32_t* dest,
                                   int dest_stride,
                                   int dest_h_stride) {
  int32x4_t src400 = vld1q_s32(src);
  int32x4_t src800 = vld1q_s32(src + 4);
  src += src_stride;
  int32x4_t src401 = vld1q_s32(src);
  int32x4_t src801 = vld1q_s32(src + 4);
  src += src_stride;
  int32x4_t src402 = vld1q_s32(src);
  int32x4_t src802 = vld1q_s32(src + 4);
  src += src_stride;
  int32x4_t src403 = vld1q_s32(src);
  int32x4_t src803 = vld1q_s32(src + 4);

  src += src_h_stride - 3 * src_stride;

  int32x4_t src410 = vld1q_s32(src);
  int32x4_t src810 = vld1q_s32(src + 4);
  src += src_stride;
  int32x4_t src411 = vld1q_s32(src);
  int32x4_t src811 = vld1q_s32(src + 4);
  src += src_stride;
  int32x4_t src412 = vld1q_s32(src);
  int32x4_t src812 = vld1q_s32(src + 4);
  src += src_stride;
  int32x4_t src413 = vld1q_s32(src);
  int32x4_t src813 = vld1q_s32(src + 4);

  src += src_h_stride - 3 * src_stride;

  int32x4_t src420 = vld1q_s32(src);
  int32x4_t src820 = vld1q_s32(src + 4);
  src += src_stride;
  int32x4_t src421 = vld1q_s32(src);
  int32x4_t src821 = vld1q_s32(src + 4);
  src += src_stride;
  int32x4_t src422 = vld1q_s32(src);
  int32x4_t src822 = vld1q_s32(src + 4);
  src += src_stride;
  int32x4_t src423 = vld1q_s32(src);
  int32x4_t src823 = vld1q_s32(src + 4);

  src += src_h_stride - 3 * src_stride;

  int32x4_t src430 = vld1q_s32(src);
  int32x4_t src830 = vld1q_s32(src + 4);
  src += src_stride;
  int32x4_t src431 = vld1q_s32(src);
  int32x4_t src831 = vld1q_s32(src + 4);
  src += src_stride;
  int32x4_t src432 = vld1q_s32(src);
  int32x4_t src832 = vld1q_s32(src + 4);
  src += src_stride;
  int32x4_t src433 = vld1q_s32(src);
  int32x4_t src833 = vld1q_s32(src + 4);

  int32x4_t dst400 = vaddq_s32(vaddq_s32(src400, src401), src402);
  int32x4_t dst410 = vsubq_s32(vsubq_s32(src401, src402), src403);
  int32x4_t dst401 = vaddq_s32(vaddq_s32(src410, src411), src412);
  int32x4_t dst411 = vsubq_s32(vsubq_s32(src411, src412), src413);
  int32x4_t dst402 = vaddq_s32(vaddq_s32(src420, src421), src422);
  int32x4_t dst412 = vsubq_s32(vsubq_s32(src421, src422), src423);
  int32x4_t dst403 = vaddq_s32(vaddq_s32(src430, src431), src432);
  int32x4_t dst413 = vsubq_s32(vsubq_s32(src431, src432), src433);

  int32x4_t dst800 = vaddq_s32(vaddq_s32(src800, src801), src802);
  int32x4_t dst810 = vsubq_s32(vsubq_s32(src801, src802), src803);
  int32x4_t dst801 = vaddq_s32(vaddq_s32(src810, src811), src812);
  int32x4_t dst811 = vsubq_s32(vsubq_s32(src811, src812), src813);
  int32x4_t dst802 = vaddq_s32(vaddq_s32(src820, src821), src822);
  int32x4_t dst812 = vsubq_s32(vsubq_s32(src821, src822), src823);
  int32x4_t dst803 = vaddq_s32(vaddq_s32(src830, src831), src832);
  int32x4_t dst813 = vsubq_s32(vsubq_s32(src831, src832), src833);

  int32x4_t dest400 = vaddq_s32(vaddq_s32(dst400, dst401), dst402);
  int32x4_t dest410 = vsubq_s32(vsubq_s32(dst401, dst402), dst403);
  int32x4_t dest401 = vaddq_s32(vaddq_s32(dst410, dst411), dst412);
  int32x4_t dest411 = vsubq_s32(vsubq_s32(dst411, dst412), dst413);

  int32x4_t dest800 = vaddq_s32(vaddq_s32(dst800, dst801), dst802);
  int32x4_t dest810 = vsubq_s32(vsubq_s32(dst801, dst802), dst803);
  int32x4_t dest801 = vaddq_s32(vaddq_s32(dst810, dst811), dst812);
  int32x4_t dest811 = vsubq_s32(vsubq_s32(dst811, dst812), dst813);

  vst1q_s32(dest, dest400);
  vst1q_s32(dest + 4, dest800);
  dest += dest_stride;
  vst1q_s32(dest, dest410);
  vst1q_s32(dest + 4, dest810);
  dest += dest_h_stride - dest_stride;
  vst1q_s32(dest, dest401);
  vst1q_s32(dest + 4, dest801);
  dest += dest_stride;
  vst1q_s32(dest, dest411);
  vst1q_s32(dest + 4, dest811);
}

// AT = [1  1  1  1  1  0
//      0  1 -1  2 -2  0
//      0  1  1  4  4  0
//      0  1 -1  8 -8  1]
void output_trans_c8_post_4x6_int8(const int32_t* src,
                                   int src_stride,
                                   int32_t* dest,
                                   int dest_stride) {
  const int32x4_t src40 = vld1q_s32(src);
  const int32x4_t src80 = vld1q_s32(src + 4);
  const int32x4_t src41 = vld1q_s32(src + src_stride);
  const int32x4_t src81 = vld1q_s32(src + src_stride + 4);
  const int32x4_t src42 = vld1q_s32(src + src_stride * 2);
  const int32x4_t src82 = vld1q_s32(src + src_stride * 2 + 4);
  const int32x4_t src43 = vld1q_s32(src + src_stride * 3);
  const int32x4_t src83 = vld1q_s32(src + src_stride * 3 + 4);
  const int32x4_t src44 = vld1q_s32(src + src_stride * 4);
  const int32x4_t src84 = vld1q_s32(src + src_stride * 4 + 4);
  const int32x4_t src45 = vld1q_s32(src + src_stride * 5);
  const int32x4_t src85 = vld1q_s32(src + src_stride * 5 + 4);

  int32x4_t tmp402a = vaddq_s32(src41, src42);
  int32x4_t tmp413a = vsubq_s32(src41, src42);
  int32x4_t tmp402b = vaddq_s32(src43, src44);
  int32x4_t tmp413b = vsubq_s32(src43, src44);
  int32x4_t tmp802a = vaddq_s32(src81, src82);
  int32x4_t tmp813a = vsubq_s32(src81, src82);
  int32x4_t tmp802b = vaddq_s32(src83, src84);
  int32x4_t tmp813b = vsubq_s32(src83, src84);
  int32x4_t dest40 = vaddq_s32(vaddq_s32(src40, tmp402a), tmp402b);
  int32x4_t dest42 = vaddq_s32(tmp402a, vshlq_n_s32(tmp402b, 2));
  int32x4_t dest41 = vaddq_s32(tmp413a, vshlq_n_s32(tmp413b, 1));
  int32x4_t dest43 =
      vaddq_s32(vaddq_s32(tmp413a, vshlq_n_s32(tmp413b, 3)), src45);
  int32x4_t dest80 = vaddq_s32(vaddq_s32(src80, tmp802a), tmp802b);
  int32x4_t dest82 = vaddq_s32(tmp802a, vshlq_n_s32(tmp802b, 2));
  int32x4_t dest81 = vaddq_s32(tmp813a, vshlq_n_s32(tmp813b, 1));
  int32x4_t dest83 =
      vaddq_s32(vaddq_s32(tmp813a, vshlq_n_s32(tmp813b, 3)), src85);
  vst1q_s32(dest, dest40);
  vst1q_s32(dest + 4, dest80);
  vst1q_s32(dest + dest_stride, dest41);
  vst1q_s32(dest + dest_stride + 4, dest81);
  vst1q_s32(dest + dest_stride * 2, dest42);
  vst1q_s32(dest + dest_stride * 2 + 4, dest82);
  vst1q_s32(dest + dest_stride * 3, dest43);
  vst1q_s32(dest + dest_stride * 3 + 4, dest83);
}

void weight_trans_c8_4x4_int8(
    int16_t* dest, const int8_t* din, int ch_in, int ch_out, void* workspace) {
  const int16_t coeff[4][3] = {{2, 0, 0}, {1, 1, 1}, {1, -1, 1}, {0, 0, 2}};

  int16_t* ptr_out = static_cast<int16_t*>(workspace);

  for (int i = 0; i < ch_out; i++) {
    for (int j = 0; j < ch_in; j++) {
      const int8_t* kernel0 =
          static_cast<const int8_t*>(din) + (i * ch_in + j) * 9;
      int16_t* ptr_channel = ptr_out + (i * ch_in + j) * 16;

      //! transform kernel, transposed
      const int8_t* k0 = kernel0;
      const int8_t* k1 = kernel0 + 3;
      const int8_t* k2 = kernel0 + 6;

      //! h
      int16_t tmp[4][3];
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
        int16_t* tmpp = &tmp[j][0];
        for (int i = 0; i < 4; i++) {
          ptr_channel[j * 4 + i] = tmpp[0] * coeff[i][0] +
                                   tmpp[1] * coeff[i][1] +
                                   tmpp[2] * coeff[i][2];
        }
      }
    }
  }

  int oc_pad = (ch_out + 7) / 8 * 8;
  int ic_pad = (ch_in + 7) / 8 * 8;
  int c_stride = ic_pad * oc_pad;
  for (int i = 0; i < ch_out * ch_in * 16; ++i) {
    int new_c = i % 16;
    int new_oc = i / ch_in / 16 / 8;
    int new_ic = i / 16 % ch_in;
    int new_inner = i / ch_in / 16 % 8;
    int dest_ind =
        new_c * c_stride + new_oc * ic_pad * 8 + new_ic * 8 + new_inner;
    dest[dest_ind] = ptr_out[i];
  }
}

// Input weight Layout: K*C*R*R (RR=3x3)
// Output weight Layout: G*G*[K/4]*[C]*4 (GG=6x6, [x] means round up to integer)
// Temp data Layout: K*C*G*G
void weight_trans_c8_6x6_int8(
    int16_t* dest, const int8_t* din, int ch_in, int ch_out, void* workspace) {
  const int32_t coeff[6][3] = {
      {6, 0, 0}, {-4, -4, -4}, {-4, 4, -4}, {1, 2, 4}, {1, -2, 4}, {0, 0, 24}};
  // const float coeff[6][3] = {{0.25f, 0.0f, 0.0f},
  //                            {-1.0f / 6, -1.0f / 6, -1.0f / 6},
  //                            {-1.0f / 6, 1.0f / 6, -1.0f / 6},
  //                            {1.0f / 24, 1.0f / 12, 1.0f / 6},
  //                            {1.0f / 24, -1.0f / 12, 1.0f / 6},
  //                            {0.0f, 0.0f, 1.0f}};

  // int16_t* ptr_out = static_cast<int16_t*>(workspace);
  int32_t* ptr_out = static_cast<int32_t*>(workspace);
  for (int i = 0; i < ch_out; i++) {
    for (int j = 0; j < ch_in; j++) {
      const int8_t* kernel0 =
          static_cast<const int8_t*>(din) + (i * ch_in + j) * 9;
      // int16_t* ptr_channel = ptr_out + (i * ch_in + j) * 36;
      int32_t* ptr_channel = ptr_out + (i * ch_in + j) * 36;

      //! transform kernel, transposed
      const int8_t* k0 = kernel0;
      const int8_t* k1 = kernel0 + 3;
      const int8_t* k2 = kernel0 + 6;

      //! h
      int32_t tmp[6][3];
      for (int i = 0; i < 6; i++) {
        tmp[i][0] = static_cast<int32_t>(k0[0]) * coeff[i][0] +
                    static_cast<int32_t>(k0[1]) * coeff[i][1] +
                    static_cast<int32_t>(k0[2]) * coeff[i][2];
        tmp[i][1] = static_cast<int32_t>(k1[0]) * coeff[i][0] +
                    static_cast<int32_t>(k1[1]) * coeff[i][1] +
                    static_cast<int32_t>(k1[2]) * coeff[i][2];
        tmp[i][2] = static_cast<int32_t>(k2[0]) * coeff[i][0] +
                    static_cast<int32_t>(k2[1]) * coeff[i][1] +
                    static_cast<int32_t>(k2[2]) * coeff[i][2];
      }

      //! v
      for (int j = 0; j < 6; j++) {
        int32_t* tmpp = &tmp[j][0];
        for (int i = 0; i < 6; i++) {
          ptr_channel[j * 6 + i] = tmpp[0] * coeff[i][0] +
                                   tmpp[1] * coeff[i][1] +
                                   tmpp[2] * coeff[i][2];
          if (i == 5 || j == 5) ptr_channel[j * 6 + i] /= 24;
        }
      }
    }
  }
  int oc_pad = (ch_out + 7) / 8 * 8;  // c8
  int ic_pad = (ch_in + 7) / 8 * 8;   // c8
  int c_stride = ic_pad * oc_pad;
  for (int i = 0; i < ch_out * ch_in * 36; ++i) {
    int new_c = i % 36;
    int new_oc = i / ch_in / 36 / 8;
    int new_ic = i / 36 % ch_in;
    int new_inner = i / ch_in / 36 % 8;
    int dest_ind =
        new_c * c_stride + new_oc * ic_pad * 8 + new_ic * 8 + new_inner;
    dest[dest_ind] = static_cast<int16_t>(ptr_out[i]);
  }
}
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
