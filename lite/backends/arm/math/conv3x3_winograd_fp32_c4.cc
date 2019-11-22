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
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif
#include <arm_neon.h>

namespace paddle {
namespace lite {
namespace arm {
namespace math {
void input_trans_c4(const float* src,
                    int src_stride,
                    float* dest,
                    int dest_stride);
void output_trans_c4(const float* src,
                     int src_stride,
                     float* dest,
                     int dest_stride);
void output_trans_c4_post(const float* src,
                          int src_stride,
                          float* dest,
                          int dest_stride,
                          float* bias_value,
                          bool has_relu);
void weight_trans_c4(
    float* dest, const float* src, int ic, int oc, void* workspace);

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
  const int pad_h = param.paddings[0];
  const int pad_w = param.paddings[1];
  float* tmp_work_space =
      ctx->workspace_data<float>() + ctx->llc_size() / sizeof(float);

  int in_n_stride = chin * hin * win;
  int out_n_stride = chout * hout * wout;
  int ic_stride = win * hin;
  int oc_stride = wout * hout;
  int ic_4 = (chin + 3) / 4;
  int oc_4 = (chout + 3) / 4;
  int ic_remain = chin - ic_4 * 4;
  int oc_remain = chout - oc_4 * 4;

  int tile_w = (wout + 5) / 6;
  int tile_h = (hout + 5) / 6;
  int size_tile = tile_h * tile_w;
  int m_pad = (chout + 3) / 4 * 4;
  int m_remain = m_pad - chout;
  int maxc_4 = ic_4 > oc_4 ? ic_4 : oc_4;
  float zero_ptr[8];
  memset(zero_ptr, 0, 8 * sizeof(float));

  int w_pad = win + pad_w * 2;
  int h_pad = hin + pad_h * 2;
  float* input_c4 = tmp_work_space;
  int new_h_stride = w_pad * 4;
  int new_c_stride = new_h_stride * h_pad;

  int ic_4_stride = w_pad * h_pad * 4;
  int oc_4_stride = wout * hout * 4;

  int tile_block = 8;
#ifdef __aarch64__
  tile_block = 16;
#endif
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
                             -pad_h,
                             hin + pad_h,
                             -pad_w,
                             win + pad_w,
                             chin,
                             win,
                             hin,
                             zero_ptr);
    }
    const float* input_ptr = input + ni * in_n_stride;
    float* output_ptr = output + ni * out_n_stride;

    const float* weight_ptr = weight;
    const float* bias_ptr = bias;
#pragma omp parallel for num_threads(threads)
    for (int tbi = 0; tbi < block_count; ++tbi) {
#ifdef ARM_WITH_OMP
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
              input_trans_c4(ci_ptr, 4, trans_tmp_data + i * 4, 32);
            }
            float* dst_ci = dst_ptr + ci * tile_count * 4;
            for (int i = 0; i < 8; ++i) {
              input_trans_c4(trans_tmp_data + i * 32,
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
              input_trans_c4(ci_ptr, 4, trans_tmp_data + i * 4, 32);
            }
            float* dst_ci = dst_ptr + ci * tile_count * 4;
            for (int i = 0; i < 8; ++i) {
              input_trans_c4(trans_tmp_data + i * 32,
                             4,
                             dst_ci + i * b_gi_stride * 8,
                             b_gi_stride);
            }
          }  // for ci_4
        }
      }
      // input trans end
      // *begin compute dot
      // *

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

      // output trans
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

        float bias_value[4];
        memset(bias_value, 0, 4 * sizeof(float));

        if (ex == 6) {
          // trans output
          for (int ci = 0; ci < oc_4; ++ci) {
            if (param.bias) {
              bias_value[0] = bias[ci * 4];
              bias_value[1] = bias[ci * 4 + 1];
              bias_value[2] = bias[ci * 4 + 2];
              bias_value[3] = bias[ci * 4 + 3];
            }

            float* dst_ci = dst_ptr + ci * oc_4_stride;
            float* src_ci = src_ptr + ci * tile_count * 4;
            for (int i = 0; i < 8; ++i) {
              output_trans_c4(src_ci + i * c_gi_stride * 8,
                              c_gi_stride,
                              trans_tmp_data + i * 4,
                              32);
            }
            for (int i = 0; i < ey; ++i) {
              output_trans_c4_post(trans_tmp_data + i * 32,
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
                                    zero_ptr);
          }
        } else {
          for (int ci = 0; ci < oc_4; ++ci) {
            if (param.bias) {
              bias_value[0] = bias[ci * 4];
              bias_value[1] = bias[ci * 4 + 1];
              bias_value[2] = bias[ci * 4 + 2];
              bias_value[3] = bias[ci * 4 + 3];
            }
            // trans output
            float* dst_ci = dst_ptr + ci * oc_4_stride;
            float* src_ci = src_ptr + ci * tile_count * 4;
            for (int i = 0; i < 8; ++i) {
              output_trans_c4(src_ci + i * c_gi_stride * 8,
                              c_gi_stride,
                              trans_tmp_data + i * 4,
                              32);
            }
            for (int i = 0; i < ey; ++i) {
              output_trans_c4_post(trans_tmp_data + i * 32,
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
                                    zero_ptr);
          }
        }
      }
    }  // for block_count
  }    // for num
}  // conv_compute

void output_trans_c4(const float* src,
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
void output_trans_c4_post(const float* src,
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

  if (bias_value) {
    float32x4_t bias = vld1q_f32(bias_value);
    dest0 = vaddq_f32(dest0, bias);
    dest1 = vaddq_f32(dest1, bias);
    dest2 = vaddq_f32(dest2, bias);
    dest3 = vaddq_f32(dest3, bias);
    dest4 = vaddq_f32(dest4, bias);
    dest5 = vaddq_f32(dest5, bias);
  }

  if (has_relu) {
    float32x4_t zeros = vdupq_n_f32(0);
    dest0 = vmaxq_f32(dest0, zeros);
    dest1 = vmaxq_f32(dest1, zeros);
    dest2 = vmaxq_f32(dest2, zeros);
    dest3 = vmaxq_f32(dest3, zeros);
    dest4 = vmaxq_f32(dest4, zeros);
    dest5 = vmaxq_f32(dest5, zeros);
  }

  vst1q_f32(dest, dest0);
  vst1q_f32(dest + dest_stride, dest1);
  vst1q_f32(dest + dest_stride * 2, dest2);
  vst1q_f32(dest + dest_stride * 3, dest3);
  vst1q_f32(dest + dest_stride * 4, dest4);
  vst1q_f32(dest + dest_stride * 5, dest5);
}

void input_trans_c4(const float* src,
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
void weight_trans_c4(
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
    int new_ic = i / 64 % (ch_in * 4) % ch_in;
    int new_inner = i / ch_in / 64 % 4;
    int dest_ind =
        new_c * c_stride + new_oc * ic_pad * 4 + new_ic * 4 + new_inner;
    dest[dest_ind] = ptr_out[i];
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
