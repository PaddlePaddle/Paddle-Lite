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

#include "lite/backends/arm/math/sgemv.h"
#include <arm_neon.h>
#include <algorithm>
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void sgemv(const int M,
           const int N,
           const float *A,
           const float *x,
           float *y,
           float beta,
           bool flag_bias,
           const float *bias);

void sgemv_relu(const int M,
                const int N,
                const float *A,
                const float *x,
                float *y,
                float beta,
                bool flag_bias,
                const float *bias);

void sgemv_relu6(const int M,
                 const int N,
                 const float *A,
                 const float *x,
                 float *y,
                 float beta,
                 bool flag_bias,
                 const float *bias,
                 const float six);

void sgemv_leakey_relu(const int M,
                       const int N,
                       const float *A,
                       const float *x,
                       float *y,
                       float beta,
                       bool flag_bias,
                       const float *bias,
                       const float alpha);

void sgemv_trans(const int M,
                 const int N,
                 const float *A,
                 const float *x,
                 float *y,
                 float beta,
                 bool flag_bias,
                 const float *bias,
                 bool flag_act,
                 lite_api::ActivationType act,
                 const ARMContext *ctx,
                 float six,
                 float alpha);

bool sgemv(const float *A,
           const float *x,
           float *y,
           bool transA,
           int M,
           int N,
           float beta,
           bool is_bias,
           const float *bias,
           bool flag_act,
           lite_api::ActivationType act,
           const ARMContext *ctx,
           float six,
           float alpha) {
  if (transA) {
    sgemv_trans(
        M, N, A, x, y, beta, is_bias, bias, flag_act, act, ctx, six, alpha);
  } else {
    if (flag_act) {
      if (act == lite_api::ActivationType::kRelu) {
        sgemv_relu(M, N, A, x, y, beta, is_bias, bias);
      } else if (act == lite_api::ActivationType::kRelu6) {
        sgemv_relu6(M, N, A, x, y, beta, is_bias, bias, six);
      } else if (act == lite_api::ActivationType::kLeakyRelu) {
        sgemv_leakey_relu(M, N, A, x, y, beta, is_bias, bias, alpha);
      } else {
        LOG(FATAL)
            << "sgemv no transA only support relu, relu6, leakey relu fusion";
      }
    } else {
      sgemv(M, N, A, x, y, beta, is_bias, bias);
    }
  }
  return true;
}

#ifdef __aarch64__
void sgemv_trans(const int M,
                 const int N,
                 const float *A,
                 const float *x,
                 float *y,
                 float beta,
                 bool flag_bias,
                 const float *bias,
                 bool flag_act,
                 lite_api::ActivationType act,
                 const ARMContext *ctx,
                 float six,
                 float alpha) {
  int m_cnt16 = M >> 4;
  int m_cnt8 = (M & 15) >> 3;
  int m_cnt4 = (M & 15 & 7) >> 2;
  int m_remain = M & 15 & 7 & 3;
  int ths = ctx->threads();
  int valid_ths = std::min((N + 3) / 4, ths);
  int valid_block = std::max(4, (N / valid_ths + 3) / 4 * 4);
  valid_ths = (N + valid_block - 1) / valid_block;
  int block_cnt = valid_block / 4;
  float zero_buf[M];           // NOLINT
  float y_buf[valid_ths * M];  // NOLINT
  bool has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  memset(zero_buf, 0, M * sizeof(float));
  if (flag_bias) {
    memcpy(y_buf, bias, M * sizeof(float));
    memset(y_buf + M, 0, (valid_ths - 1) * M * sizeof(float));
  } else {
    memset(y_buf, 0, valid_ths * M * sizeof(float));
  }
#pragma omp parallel for
  for (int t = 0; t < valid_ths; ++t) {
    float *block_y = y_buf + t * M;
    const float *block_x = x + t * valid_block;
    const float *block_A = A + t * valid_block * M;
    for (int i = 0; i < block_cnt; ++i) {
      float *y_ptr = block_y;
      const float *x_ptr = block_x + i * 4;
      const float *in0_ptr = block_A + i * 4 * M;
      const float *in1_ptr = in0_ptr + M;
      const float *in2_ptr = in1_ptr + M;
      const float *in3_ptr = in2_ptr + M;
      int offset = t * valid_block + (i + 1) * 4 - N;
      if (offset > 0) {
        if (offset > 3) {
          in0_ptr = zero_buf;
          in1_ptr = zero_buf;
          in2_ptr = zero_buf;
          in3_ptr = zero_buf;
        } else {
          switch (offset) {
            case 3:
              in1_ptr = zero_buf;
            case 2:
              in2_ptr = zero_buf;
            case 1:
              in3_ptr = zero_buf;
            default:
              break;
          }
        }
      }
      // clang-format off
      if (m_cnt16 > 0) {
        int cnt16 = m_cnt16;
        asm volatile(
            "ld1  {v4.4s},  [%[x]]    \n"                               /* load x   to v4     */
            "ld1  {v5.4s,  v6.4s,  v7.4s,  v8.4s},   [%[in0]], #64 \n"  /* load in0 to v5,  v6,  v7,  v8  */
            "ld1  {v9.4s,  v10.4s, v11.4s, v12.4s},  [%[in1]], #64 \n"  /* load in1 to v9,  v10, v11, v12 */
            "ld1  {v13.4s, v14.4s, v15.4s, v16.4s},  [%[in2]], #64 \n"  /* load in2 to v13, v14, v15, v16 */
            "ld1  {v17.4s, v18.4s, v19.4s, v20.4s},  [%[in3]], #64 \n"  /* load in3 to v17, v18, v19, v20 */
            "1:\n"
            "ld1  {v0.4s, v1.4s, v2.4s, v3.4s},  [%[y]]    \n"        /*load y to v0, v1, v2, v3  */
            "fmla v0.4s,  v5.4s,  v4.s[0]     \n" /*  v0 += v5 * v4[0]  */
            "fmla v1.4s,  v6.4s,  v4.s[0]     \n" /*  v1 += v6 * v4[0]  */
            "fmla v2.4s,  v7.4s,  v4.s[0]     \n" /*  v2 += v7 * v4[0]  */
            "fmla v3.4s,  v8.4s,  v4.s[0]     \n" /*  v3 += v8 * v4[0]  */
            "ld1  {v5.4s, v6.4s,  v7.4s,  v8.4s},   [%[in0]], #64 \n" /* load in0 to v5,  v6,  v7,  v8  */
            "fmla v0.4s,  v9.4s,  v4.s[1]     \n" /*  v0 += v9  * v4[1]  */
            "fmla v1.4s,  v10.4s, v4.s[1]     \n" /*  v1 += v10 * v4[1]  */
            "fmla v2.4s,  v11.4s, v4.s[1]     \n" /*  v2 += v11 * v4[1]  */
            "fmla v3.4s,  v12.4s, v4.s[1]     \n" /*  v3 += v12 * v4[1]  */
            "ld1  {v9.4s, v10.4s, v11.4s, v12.4s},  [%[in1]], #64 \n" /* load in1 to v9,  v10, v11, v12 */
            "fmla v0.4s,  v13.4s, v4.s[2]     \n" /*  v0 += v13 * v4[2]  */
            "fmla v1.4s,  v14.4s, v4.s[2]     \n" /*  v1 += v14 * v4[2]  */
            "fmla v2.4s,  v15.4s, v4.s[2]     \n" /*  v2 += v15 * v4[2]  */
            "fmla v3.4s,  v16.4s, v4.s[2]     \n" /*  v3 += v16 * v4[2]  */
            "ld1  {v13.4s, v14.4s, v15.4s, v16.4s}, [%[in2]], #64 \n" /* load in2 to v13, v14, v15, v16 */
            "fmla v0.4s,  v17.4s, v4.s[3]     \n" /*  v0 += v17 * v4[3]  */
            "fmla v1.4s,  v18.4s, v4.s[3]     \n" /*  v1 += v18 * v4[3]  */
            "fmla v2.4s,  v19.4s, v4.s[3]     \n" /*  v2 += v19 * v4[3]  */
            "fmla v3.4s,  v20.4s, v4.s[3]     \n" /*  v3 += v20 * v4[3]  */
            "ld1  {v17.4s, v18.4s, v19.4s, v20.4s}, [%[in3]], #64 \n" /* load in3 to v17, v18, v19, v20 */
            "subs %w[cnt], %w[cnt], #1        \n" /*       sub cnt       */
            "st1  {v0.4s, v1.4s, v2.4s, v3.4s}, [%[y]], #64   \n"     /*  store v0, v1, v2, v3 to y */
            "bne  1b  \n"                     /*  branch to label 1 */
            "sub  %[in0], %[in0], #64     \n" /* restore in0 address */
            "sub  %[in1], %[in1], #64     \n" /* restore in1 address */
            "sub  %[in2], %[in2], #64     \n" /* restore in2 address */
            "sub  %[in3], %[in3], #64     \n" /* restore in3 address */
            : [cnt] "+r"(cnt16),
              [in0] "+r"(in0_ptr),
              [in1] "+r"(in1_ptr),
              [in2] "+r"(in2_ptr),
              [in3] "+r"(in3_ptr),
              [y] "+r"(y_ptr)
            : [x] "r"(x_ptr)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", 
              "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", 
              "v17", "v18", "v19", "v20", "cc", "memory"
        );
      }
      if (m_cnt8 > 0) {
        int cnt8 = m_cnt8;
        asm volatile(
            "ld1  {v2.4s},  [%[x]]                \n" /* load x   to v2     */
            "ld1  {v3.4s, v4.4s},  [%[in0]], #32  \n" /* load in0 to v3, v4 */
            "ld1  {v5.4s, v6.4s},  [%[in1]], #32  \n" /* load in1 to v5, v6 */
            "ld1  {v7.4s, v8.4s},  [%[in2]], #32  \n" /* load in2 to v7, v8 */
            "ld1  {v9.4s, v10.4s}, [%[in3]], #32  \n" /* load in3 to v9, v10*/
            "1:\n"
            "ld1  {v0.4s, v1.4s}, [%[y]]    \n" /*  load y to v0, v1  */
            "fmla v0.4s, v3.4s,   v2.s[0]   \n" /*  v0 += v3 * v2[0]  */
            "fmla v1.4s, v4.4s,   v2.s[0]   \n" /*  v1 += v4 * v2[0]  */
            "prfm pldl1keep,      [%[in0]]  \n" /*    preload in0     */
            "ld1  {v3.4s, v4.4s}, [%[in0]], #32 \n" /* load in0 to v3, v4 */
            "fmla v0.4s, v5.4s,   v2.s[1]   \n" /*  v0 += v5 * v2[1]  */
            "fmla v1.4s, v6.4s,   v2.s[1]   \n" /*  v1 += v6 * v2[1]  */
            "prfm pldl1keep,      [%[in1]]  \n" /*    preload in1     */
            "ld1  {v5.4s, v6.4s}, [%[in1]], #32 \n" /* load in0 to v5, v6 */
            "fmla v0.4s, v7.4s,   v2.s[2]   \n" /*  v0 += v7 * v2[2]  */
            "fmla v1.4s, v8.4s,   v2.s[2]   \n" /*  v1 += v8 * v2[2]  */
            "prfm pldl1keep,      [%[in2]]  \n" /*    preload in2     */
            "ld1  {v7.4s, v8.4s}, [%[in2]], #32 \n" /* load in0 to v7, v8 */
            "fmla v0.4s, v9.4s,   v2.s[3]   \n" /*  v0 += v9 * v2[3]  */
            "fmla v1.4s, v10.4s,  v2.s[3]   \n" /*  v1 += v10 * v2[3] */
            "subs %w[cnt], %w[cnt], #1      \n" /*      sub cnt       */
            "prfm pldl1keep,      [%[in3]]  \n" /*    preload in3     */
            "st1  {v0.4s, v1.4s}, [%[y]],   #32 \n" /*  store v0, v1 to y */
            "ld1  {v9.4s, v10.4s},[%[in3]], #32 \n" /* load in0 to v9, v10*/
            "bne  1b  \n"                       /*  branch to label 1 */
            "sub  %[in0], %[in0], #32     \n" /* restore in0 address */
            "sub  %[in1], %[in1], #32     \n" /* restore in1 address */
            "sub  %[in2], %[in2], #32     \n" /* restore in2 address */
            "sub  %[in3], %[in3], #32     \n" /* restore in3 address */
            : [cnt] "+r"(cnt8),
              [in0] "+r"(in0_ptr),
              [in1] "+r"(in1_ptr),
              [in2] "+r"(in2_ptr),
              [in3] "+r"(in3_ptr),
              [y] "+r"(y_ptr)
            : [x] "r"(x_ptr)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", 
              "v7", "v8", "v9", "v10", "cc", "memory"
        );
      }
      if (m_cnt4 > 0) {
        int cnt4 = m_cnt4;
        asm volatile(
            "ld1  {v1.4s},  [%[in0]], #16 \n" /* load in0 to v1  */
            "ld1  {v2.4s},  [%[in1]], #16 \n" /* load in1 to v2  */
            "ld1  {v3.4s},  [%[in2]], #16 \n" /* load in2 to v3  */
            "ld1  {v4.4s},  [%[in3]], #16 \n" /* load in3 to v4  */
            "ld1  {v5.4s},  [%[x]]        \n" /* load x   to v5  */
            "1:\n"
            "ld1  {v0.4s},  [%[y]]        \n" /*   load y to v0    */
            "fmla v0.4s, v1.4s, v5.s[0]   \n" /* v0 += v1 * v5[0]  */
            "prfm  pldl1keep,   [%[in0]]  \n" /*    preload in0    */
            "ld1  {v1.4s},  [%[in0]], #16 \n" /*  load in0 to v1   */
            "fmla v0.4s, v2.4s, v5.s[1]   \n" /* v0 += v2 * v5[1]  */
            "prfm  pldl1keep,  [%[in1]]   \n" /*    preload in1    */
            "ld1  {v2.4s},  [%[in1]], #16 \n" /*  load in1 to v2   */
            "fmla v0.4s, v3.4s, v5.s[2]   \n" /* v0 += v3 * v5[2]  */
            "prfm pldl1keep,  [%[in2]]    \n" /*    preload in2    */
            "ld1  {v3.4s},  [%[in2]], #16 \n" /*  load in2 to v3   */
            "fmla v0.4s, v4.4s, v5.s[3]   \n" /* v0 += v4 * v5[3]  */
            "subs %w[cnt], %w[cnt], #1    \n" /*      sub cnt      */
            "prfm pldl1keep,  [%[in3]]    \n" /*    preload in3    */
            "st1  {v0.4s},  [%[y]], #16   \n" /*  store v0 to y    */
            "ld1  {v4.4s},  [%[in3]], #16 \n" /*  load in3 to v4   */
            "bne  1b  \n"                     /* branch to label 1 */
            "sub  %[in0], %[in0], #16     \n" /* restore in0 address*/
            "sub  %[in1], %[in1], #16     \n" /* restore in1 address*/
            "sub  %[in2], %[in2], #16     \n" /* restore in2 address*/
            "sub  %[in3], %[in3], #16     \n" /* restore in3 address*/
            : [cnt] "+r"(cnt4),
              [in0] "+r"(in0_ptr),
              [in1] "+r"(in1_ptr),
              [in2] "+r"(in2_ptr),
              [in3] "+r"(in3_ptr),
              [y] "+r"(y_ptr)
            : [x] "r"(x_ptr)
            : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory"
        );
      }
      // clang-format on
      for (int r = 0; r < m_remain; ++r) {
        float val0 = x_ptr[0] * in0_ptr[r];
        float val1 = x_ptr[1] * in1_ptr[r];
        float val2 = x_ptr[2] * in2_ptr[r];
        float val3 = x_ptr[3] * in3_ptr[r];
        y_ptr[r] += val0 + val1 + val2 + val3;
      }
    }
  }
  int cnt4 = M >> 2;
  int remain = M & 3;
  //! do reduction
  int rdc_ths = valid_ths >> 1;
  while (rdc_ths > 0) {
#pragma omp parallel for
    for (int t = 0; t < rdc_ths; ++t) {
      float *y0 = y_buf + t * M;
      for (int i = t + rdc_ths; i < valid_ths; i += rdc_ths) {
        float *y0_ptr = y0;
        float *y_ptr = y_buf + i * M;
        for (int j = 0; j < cnt4; ++j) {
          float32x4_t val0 = vld1q_f32(y0_ptr + j * 4);
          float32x4_t val1 = vld1q_f32(y_ptr + j * 4);
          float32x4_t val = vaddq_f32(val0, val1);
          vst1q_f32(y0_ptr + j * 4, val);
        }
        y0_ptr += cnt4 * 4;
        y_ptr += cnt4 * 4;
        for (int j = 0; j < remain; ++j) {
          y0_ptr[j] += y_ptr[j];
        }
      }
    }
    valid_ths = rdc_ths;
    rdc_ths = rdc_ths >> 1;
  }
  if (has_beta) {
    if (flag_act) {
      float *in_y = y_buf;
      float32x4_t vzero = vdupq_n_f32(0.f);
      float32x4_t vbeta = vdupq_n_f32(beta);
      if (act == lite_api::ActivationType::kRelu) {
        if (cnt4 > 0) {
          int cnt = cnt4;
          asm volatile(
              "ld1  {v0.4s},  [%[in_y]], #16  \n" /*  load y to v0    */
              "1:\n"
              "fmax v1.4s, v0.4s, %[vzero].4s \n" /*      v0 relu     */
              "fadd v1.4s, v1.4s, %[vbeta].4s \n"
              "ld1  {v0.4s},  [%[in_y]], #16  \n" /*   load y to v0   */
              "subs %w[cnt],  %w[cnt], #1     \n" /*      sub cnt     */
              "st1  {v1.4s},  [%[out_y]], #16 \n" /*  store v1 to y   */
              "bne  1b                        \n" /* branch to label 1*/
              "sub  %[in_y],  %[in_y],  #16   \n" /*   restore in_y   */
              : [cnt] "+r"(cnt), [in_y] "+r"(in_y), [out_y] "+r"(y)
              : [vzero] "w"(vzero), [vbeta] "w"(vbeta)
              : "v0", "v1", "cc", "memory");
        }
        for (int r = 0; r < remain; ++r) {
          y[r] = beta * y[r] + in_y[r] > 0.f ? in_y[r] : 0.f;
        }
      } else if (act == lite_api::ActivationType::kRelu6) {
        float32x4_t vsix = vdupq_n_f32(six);
        if (cnt4 > 0) {
          int cnt = cnt4;
          asm volatile(
              "ld1  {v0.4s},  [%[in_y]], #16  \n" /*  load y to v0    */
              "1:\n"
              "fmax v1.4s, v0.4s, %[vzero].4s \n" /*      v0 relu6    */
              "fmin v1.4s, v1.4s, %[vsix].4s  \n" /*      v1 relu6    */
              "ld1  {v0.4s},  [%[in_y]], #16  \n" /*   load y to v0   */
              "fadd v1.4s, v1.4s, %[vbeta].4s \n"
              "subs %w[cnt],  %w[cnt], #1     \n" /*      sub cnt     */
              "st1  {v1.4s},  [%[out_y]], #16 \n" /*  store v1 to y   */
              "bne  1b                        \n" /* branch to label 1*/
              "sub  %[in_y],  %[in_y],  #16   \n" /*   restore in_y   */
              : [cnt] "+r"(cnt), [in_y] "+r"(in_y), [out_y] "+r"(y)
              : [vzero] "w"(vzero), [vsix] "w"(vsix), [vbeta] "w"(vbeta)
              : "v0", "v1", "cc", "memory");
        }
        for (int r = 0; r < remain; ++r) {
          float tmp = in_y[r] > 0.f ? in_y[r] : 0.f;
          y[r] = beta * y[r] + (tmp > six ? six : y[r]);
        }
      } else if (act == lite_api::ActivationType::kLeakyRelu) {
        float32x4_t valpha = vdupq_n_f32(alpha);
        if (cnt4 > 0) {
          int cnt = cnt4;
          asm volatile(
              "1:\n"
              "ld1   {v0.4s},  [%[in_y]],   #16 \n" /*   load y to v0   */
              "fcmge v4.4s, v0.4s,  %[vzero].4s \n" /*    vcgeq_f32     */
              "fmul  v5.4s, v0.4s, %[valpha].4s \n" /*    vmulq_f32     */
              "bif   v0.16b,   v5.16b,   v4.16b \n" /*      choose      */
              "fadd  v0.4s, v0.4s, %[vbeta].4s \n"
              "subs  %w[cnt],  %w[cnt], #1      \n" /*      sub cnt     */
              "st1   {v0.4s},  [%[out_y]], #16  \n" /*  store v0 to y   */
              "bne   1b                         \n" /* branch to label 1*/
              : [cnt] "+r"(cnt), [in_y] "+r"(in_y), [out_y] "+r"(y)
              : [vzero] "w"(vzero), [valpha] "w"(valpha), [vbeta] "w"(vbeta)
              : "v0", "v4", "v5", "cc", "memory");
        }
        for (int r = 0; r < remain; ++r) {
          y[r] = beta * y[r] + (in_y[r] < 0.f ? alpha * in_y[r] : in_y[r]);
        }
      }
    } else {
      for (int i = 0; i < M; i++) {
        y[i] = beta * y[i] + y_buf[i];
      }
    }
  } else {
    if (flag_act) {
      float *in_y = y_buf;
      float32x4_t vzero = vdupq_n_f32(0.f);
      if (act == lite_api::ActivationType::kRelu) {
        if (cnt4 > 0) {
          int cnt = cnt4;
          asm volatile(
              "ld1  {v0.4s},  [%[in_y]], #16  \n" /*  load y to v0    */
              "1:\n"
              "fmax v1.4s, v0.4s, %[vzero].4s \n" /*      v0 relu     */
              "ld1  {v0.4s},  [%[in_y]], #16  \n" /*   load y to v0   */
              "subs %w[cnt],  %w[cnt], #1     \n" /*      sub cnt     */
              "st1  {v1.4s},  [%[out_y]], #16 \n" /*  store v1 to y   */
              "bne  1b                        \n" /* branch to label 1*/
              "sub  %[in_y],  %[in_y],  #16   \n" /*   restore in_y   */
              : [cnt] "+r"(cnt), [in_y] "+r"(in_y), [out_y] "+r"(y)
              : [vzero] "w"(vzero)
              : "v0", "v1", "cc", "memory");
        }
        for (int r = 0; r < remain; ++r) {
          y[r] = in_y[r] > 0.f ? in_y[r] : 0.f;
        }
      } else if (act == lite_api::ActivationType::kRelu6) {
        float32x4_t vsix = vdupq_n_f32(six);
        if (cnt4 > 0) {
          int cnt = cnt4;
          asm volatile(
              "ld1  {v0.4s},  [%[in_y]], #16  \n" /*  load y to v0    */
              "1:\n"
              "fmax v1.4s, v0.4s, %[vzero].4s \n" /*      v0 relu6    */
              "fmin v1.4s, v1.4s, %[vsix].4s  \n" /*      v1 relu6    */
              "ld1  {v0.4s},  [%[in_y]], #16  \n" /*   load y to v0   */
              "subs %w[cnt],  %w[cnt], #1     \n" /*      sub cnt     */
              "st1  {v1.4s},  [%[out_y]], #16 \n" /*  store v1 to y   */
              "bne  1b                        \n" /* branch to label 1*/
              "sub  %[in_y],  %[in_y],  #16   \n" /*   restore in_y   */
              : [cnt] "+r"(cnt), [in_y] "+r"(in_y), [out_y] "+r"(y)
              : [vzero] "w"(vzero), [vsix] "w"(vsix)
              : "v0", "v1", "cc", "memory");
        }
        for (int r = 0; r < remain; ++r) {
          y[r] = in_y[r] > 0.f ? in_y[r] : 0.f;
          y[r] = y[r] > six ? six : y[r];
        }
      } else if (act == lite_api::ActivationType::kLeakyRelu) {
        float32x4_t valpha = vdupq_n_f32(alpha);
        if (cnt4 > 0) {
          int cnt = cnt4;
          asm volatile(
              "1:\n"
              "ld1   {v0.4s},  [%[in_y]],   #16 \n" /*   load y to v0   */
              "fcmge v4.4s, v0.4s,  %[vzero].4s \n" /*    vcgeq_f32     */
              "fmul  v5.4s, v0.4s, %[valpha].4s \n" /*    vmulq_f32     */
              "bif   v0.16b,   v5.16b,   v4.16b \n" /*      choose      */
              "subs  %w[cnt],  %w[cnt], #1      \n" /*      sub cnt     */
              "st1   {v0.4s},  [%[out_y]], #16  \n" /*  store v0 to y   */
              "bne   1b                         \n" /* branch to label 1*/
              : [cnt] "+r"(cnt), [in_y] "+r"(in_y), [out_y] "+r"(y)
              : [vzero] "w"(vzero), [valpha] "w"(valpha)
              : "v0", "v4", "v5", "cc", "memory");
        }
        for (int r = 0; r < remain; ++r) {
          y[r] = in_y[r] < 0.f ? alpha * in_y[r] : in_y[r];
        }
      }
    } else {
      memcpy(y, y_buf, M * sizeof(float));
    }
  }
}
#else
void sgemv_trans(const int M,
                 const int N,
                 const float *A,
                 const float *x,
                 float *y,
                 float beta,
                 bool flag_bias,
                 const float *bias,
                 bool flag_act,
                 lite_api::ActivationType act,
                 const ARMContext *ctx,
                 float six,
                 float alpha) {
  int m_cnt8 = M >> 3;
  int m_cnt4 = (M & 7) >> 2;
  int m_remain = M & 7 & 3;
  int ths = ctx->threads();
  int valid_ths = std::min((N + 3) / 4, ths);
  int valid_block = std::max(4, (N / valid_ths + 3) / 4 * 4);
  valid_ths = (N + valid_block - 1) / valid_block;
  int block_cnt = valid_block / 4;
  float zero_buf[M];           // NOLINT
  float y_buf[valid_ths * M];  // NOLINT
  memset(zero_buf, 0, M * sizeof(float));
  bool has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  if (flag_bias) {
    memcpy(y_buf, bias, M * sizeof(float));
    memset(y_buf + M, 0, (valid_ths - 1) * M * sizeof(float));
  } else {
    memset(y_buf, 0, valid_ths * M * sizeof(float));
  }
#pragma omp parallel for
  for (int t = 0; t < valid_ths; ++t) {
    float *block_y = y_buf + t * M;
    const float *block_x = x + t * valid_block;
    const float *block_A = A + t * valid_block * M;
    for (int i = 0; i < block_cnt; ++i) {
      float *y_ptr = block_y;
      const float *x_ptr = block_x + i * 4;
      const float *in0_ptr = block_A + i * 4 * M;
      const float *in1_ptr = in0_ptr + M;
      const float *in2_ptr = in1_ptr + M;
      const float *in3_ptr = in2_ptr + M;
      int offset = t * valid_block + (i + 1) * 4 - N;
      if (offset > 0) {
        if (offset > 3) {
          in0_ptr = zero_buf;
          in1_ptr = zero_buf;
          in2_ptr = zero_buf;
          in3_ptr = zero_buf;
        } else {
          switch (offset) {
            case 3:
              in1_ptr = zero_buf;
            case 2:
              in2_ptr = zero_buf;
            case 1:
              in3_ptr = zero_buf;
            default:
              break;
          }
        }
      }
      // clang-format off
      if (m_cnt8 > 0) {
        int cnt8 = m_cnt8;
        asm volatile(
            "vld1.32  {d4-d5},  [%[x]]    \n" /* load x   to q2     */
            "vld1.32  {d6-d9},  [%[in0]]! \n" /* load in0 to q3, q4 */
            "vld1.32  {d10-d13},[%[in1]]! \n" /* load in1 to q5, q6 */
            "vld1.32  {d14-d17},[%[in2]]! \n" /* load in2 to q7, q8 */
            "vld1.32  {d18-d21},[%[in3]]! \n" /* load in3 to q9, q10*/
            "1:\n"
            "vld1.32  {d0-d3},  [%[y]]    \n" /*  load y to q0, q1  */
            "vmla.f32 q0, q3,   d4[0]     \n" /*  q0 += q3 * q2[0]  */
            "vmla.f32 q1, q4,   d4[0]     \n" /*  q1 += q4 * q2[0]  */
            "pld  [%[in0]]                \n" /*    preload in0     */
            "vld1.32  {d6-d9},  [%[in0]]! \n" /* load in0 to q3, q4 */
            "vmla.f32 q0, q5,   d4[1]     \n" /*  q0 += q5 * q2[1]  */
            "vmla.f32 q1, q6,   d4[1]     \n" /*  q1 += q6 * q2[1]  */
            "pld  [%[in1]]                \n" /*    preload in1     */
            "vld1.32  {d10-d13},[%[in1]]! \n" /* load in0 to q5, q6 */
            "vmla.f32 q0, q7,   d5[0]     \n" /*  q0 += q7 * q2[2]  */
            "vmla.f32 q1, q8,   d5[0]     \n" /*  q1 += q8 * q2[2]  */
            "pld  [%[in2]]                \n" /*    preload in2     */
            "vld1.32  {d14-d17},[%[in2]]! \n" /* load in0 to q7, q8 */
            "vmla.f32 q0, q9,   d5[1]     \n" /*  q0 += q9 * q2[3]  */
            "vmla.f32 q1, q10,  d5[1]     \n" /*  q1 += q10 * q2[3] */
            "subs %[cnt], %[cnt], #1      \n" /*      sub cnt       */
            "pld  [%[in3]]                \n" /*    preload in3     */
            "vst1.32  {d0-d3},  [%[y]]!   \n" /*  store q0, q1 to y */
            "vld1.32  {d18-d21},[%[in3]]! \n" /* load in0 to q9, q10*/
            "pld  [%[y], #32] \n"             /*     preload y      */
            "bne  1b  \n"                     /*  branch to label 1 */
            "sub  %[in0], %[in0], #32     \n" /* restore in0 address */
            "sub  %[in1], %[in1], #32     \n" /* restore in1 address */
            "sub  %[in2], %[in2], #32     \n" /* restore in2 address */
            "sub  %[in3], %[in3], #32     \n" /* restore in3 address */
            : [cnt] "+r"(cnt8),
              [in0] "+r"(in0_ptr),
              [in1] "+r"(in1_ptr),
              [in2] "+r"(in2_ptr),
              [in3] "+r"(in3_ptr),
              [y] "+r"(y_ptr)
            : [x] "r"(x_ptr)
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", 
              "q7", "q8", "q9", "q10", "cc", "memory"
        );
      }
      if (m_cnt4 > 0) {
        int cnt4 = m_cnt4;
        asm volatile(
            "vld1.32  {d2-d3},  [%[in0]]! \n" /* load in0 to q1  */
            "vld1.32  {d4-d5},  [%[in1]]! \n" /* load in1 to q2  */
            "vld1.32  {d6-d7},  [%[in2]]! \n" /* load in2 to q3  */
            "vld1.32  {d8-d9},  [%[in3]]! \n" /* load in3 to q4  */
            "vld1.32  {d10-d11},[%[x]]    \n" /* load x   to q5  */
            "1:\n"
            "vld1.32  {d0-d1},  [%[y]]    \n" /*   load y to q0    */
            "vmla.f32 q0, q1,   d10[0]    \n" /* q0 += q1 * q5[0]  */
            "pld  [%[in0]]                \n" /*    preload in0    */
            "vld1.32  {d2-d3},  [%[in0]]! \n" /*  load in0 to q1   */
            "vmla.f32 q0, q2,   d10[1]    \n" /* q0 += q2 * q5[1]  */
            "pld  [%[in1]]                \n" /*    preload in1    */
            "vld1.32  {d4-d5},  [%[in1]]! \n" /*  load in0 to q2   */
            "vmla.f32 q0, q3,   d11[0]    \n" /* q0 += q3 * q5[2]  */
            "pld  [%[in2]]                \n" /*    preload in2    */
            "vld1.32  {d6-d7},  [%[in2]]! \n" /*  load in0 to q3   */
            "vmla.f32 q0, q4,   d11[1]    \n" /* q0 += q4 * q5[3]  */
            "subs %[cnt], %[cnt], #1      \n" /*      sub cnt      */
            "pld  [%[in3]]                \n" /*    preload in3    */
            "vst1.32  {d0-d1},  [%[y]]!   \n" /*  store q0 to y    */
            "vld1.32  {d8-d9},  [%[in3]]! \n" /*  load in0 to q4   */
            "bne  1b  \n"                     /*  branch to label 1 */
            "sub  %[in0], %[in0], #16     \n" /* restore in0 address*/
            "sub  %[in1], %[in1], #16     \n" /* restore in1 address*/
            "sub  %[in2], %[in2], #16     \n" /* restore in2 address*/
            "sub  %[in3], %[in3], #16     \n" /* restore in3 address*/
            : [cnt] "+r"(cnt4),
              [in0] "+r"(in0_ptr),
              [in1] "+r"(in1_ptr),
              [in2] "+r"(in2_ptr),
              [in3] "+r"(in3_ptr),
              [y] "+r"(y_ptr)
            : [x] "r"(x_ptr)
            : "q0", "q1", "q2", "q3", "q4", "q5", "cc", "memory"
        );
      }
      // clang-format on
      for (int r = 0; r < m_remain; ++r) {
        float val0 = x_ptr[0] * in0_ptr[r];
        float val1 = x_ptr[1] * in1_ptr[r];
        float val2 = x_ptr[2] * in2_ptr[r];
        float val3 = x_ptr[3] * in3_ptr[r];
        y_ptr[r] += val0 + val1 + val2 + val3;
      }
    }
  }
  //! do reduction
  int rdc_ths = valid_ths >> 1;
  while (rdc_ths > 0) {
#pragma omp parallel for
    for (int t = 0; t < rdc_ths; ++t) {
      float *y0 = y_buf + t * M;
      for (int i = t + rdc_ths; i < valid_ths; i += rdc_ths) {
        float *y0_ptr = y0;
        float *y_ptr = y_buf + i * M;
        for (int j = 0; j < m_cnt8; ++j) {
          float32x4_t val00 = vld1q_f32(y0_ptr + j * 8);
          float32x4_t val01 = vld1q_f32(y0_ptr + j * 8 + 4);
          float32x4_t val10 = vld1q_f32(y_ptr + j * 8);
          float32x4_t val11 = vld1q_f32(y_ptr + j * 8 + 4);
          float32x4_t val0 = vaddq_f32(val00, val10);
          float32x4_t val1 = vaddq_f32(val01, val11);
          vst1q_f32(y0_ptr + j * 8, val0);
          vst1q_f32(y0_ptr + j * 8 + 4, val1);
        }
        y0_ptr += m_cnt8 * 8;
        y_ptr += m_cnt8 * 8;
        for (int j = 0; j < m_cnt4; ++j) {
          float32x4_t val0 = vld1q_f32(y0_ptr + j * 4);
          float32x4_t val1 = vld1q_f32(y_ptr + j * 4);
          float32x4_t val = vaddq_f32(val0, val1);
          vst1q_f32(y0_ptr + j * 4, val);
        }
        y0_ptr += m_cnt4 * 4;
        y_ptr += m_cnt4 * 4;
        for (int j = 0; j < m_remain; ++j) {
          y0_ptr[j] += y_ptr[j];
        }
      }
    }
    valid_ths = rdc_ths;
    rdc_ths = rdc_ths >> 1;
  }
  // do activation
  if (has_beta) {
    if (flag_act) {
      float *in_y = y_buf;
      float32x4_t vzero = vdupq_n_f32(0.f);
      float32x4_t vbeta = vdupq_n_f32(beta);
      m_cnt4 = M >> 2;
      m_remain = M & 3;
      if (act == lite_api::ActivationType::kRelu) {
        if (m_cnt4 > 0) {
          int cnt4 = m_cnt4;
          asm volatile(
              "vld1.32  {d0-d1},  [%[in_y]]!  \n" /*  load y to q0    */
              "1:\n"
              "vmax.f32 q1, q0,   %q[vzero]   \n" /*      q0 relu     */
              "vld1.32  {d0-d1},  [%[in_y]]!  \n" /*   load y to q0   */
              "vadd.f32 q1, q1, %q[vbeta]     \n"
              "subs %[cnt], %[cnt], #1        \n" /*      sub cnt     */
              "vst1.32  {d2-d3},  [%[out_y]]! \n" /*  store q1 to y   */
              "bne  1b                        \n" /* branch to label 1*/
              "sub  %[in_y],  %[in_y],  #16   \n" /*   restore in_y   */
              : [cnt] "+r"(cnt4), [in_y] "+r"(in_y), [out_y] "+r"(y)
              : [vzero] "w"(vzero), [vbeta] "w"(vbeta)
              : "q0", "q1", "cc", "memory");
        }
        for (int r = 0; r < m_remain; ++r) {
          y[r] = beta * y[r] + (in_y[r] > 0.f ? in_y[r] : 0.f);
        }
      } else if (act == lite_api::ActivationType::kRelu6) {
        float32x4_t vsix = vdupq_n_f32(six);
        if (m_cnt4 > 0) {
          int cnt4 = m_cnt4;
          asm volatile(
              "vld1.32  {d0-d1},  [%[in_y]]!  \n" /*  load y to q0    */
              "1:\n"
              "vmax.f32 q1, q0,   %q[vzero]   \n" /*      q0 relu6    */
              "vld1.32  {d0-d1},  [%[in_y]]!  \n" /*   load y to q0   */
              "vmin.f32 q1, q1,   %q[vsix]    \n" /*      q0 relu6    */
              "vadd.f32 q1, q1, %q[vbeta]     \n"
              "subs %[cnt], %[cnt], #1        \n" /*      sub cnt     */
              "vst1.32  {d2-d3},  [%[out_y]]! \n" /*  store q1 to y   */
              "bne  1b                        \n" /* branch to label 1*/
              "sub  %[in_y],  %[in_y],  #16   \n" /*   restore in_y   */
              : [cnt] "+r"(cnt4), [in_y] "+r"(in_y), [out_y] "+r"(y)
              : [vzero] "w"(vzero), [vsix] "w"(vsix), [vbeta] "w"(vbeta)
              : "q0", "q1", "cc", "memory");
        }
        for (int r = 0; r < m_remain; ++r) {
          float tmp = in_y[r] > 0.f ? in_y[r] : 0.f;
          y[r] = beta * y[r] + (tmp > six ? six : y[r]);
        }
      } else if (act == lite_api::ActivationType::kLeakyRelu) {
        float32x4_t valpha = vdupq_n_f32(alpha);
        if (m_cnt4 > 0) {
          int cnt4 = m_cnt4;
          asm volatile(
              "1:\n"
              "vld1.32  {d0-d1}, [%[in_y]]!   \n" /*   load y to q0   */
              "vcge.f32 q3, q0,  %q[vzero]    \n" /*    vcgeq_f32     */
              "vmul.f32 q4, q0,  %q[valpha]   \n" /*    vmulq_f32     */
              "vbif q0, q4, q3                \n" /*      choose      */
              "vadd.f32 q0, q0, %q[vbeta]     \n"
              "subs %[cnt], %[cnt], #1        \n" /*      sub cnt     */
              "vst1.32  {d0-d1}, [%[out_y]]!  \n" /*  store q0 to y   */
              "bne  1b                        \n" /* branch to label 1*/
              : [cnt] "+r"(cnt4), [in_y] "+r"(in_y), [out_y] "+r"(y)
              : [vzero] "w"(vzero), [valpha] "w"(valpha), [vbeta] "w"(vbeta)
              : "q0", "q3", "q4", "cc", "memory");
        }
        for (int r = 0; r < m_remain; ++r) {
          y[r] = beta * y[r] + (in_y[r] < 0.f ? alpha * in_y[r] : in_y[r]);
        }
      }
    } else {
      for (int i = 0; i < M; i++) {
        y[i] = beta * y[i] + y_buf[i];
      }
    }
  } else {
    if (flag_act) {
      float *in_y = y_buf;
      float32x4_t vzero = vdupq_n_f32(0.f);
      m_cnt4 = M >> 2;
      m_remain = M & 3;
      if (act == lite_api::ActivationType::kRelu) {
        if (m_cnt4 > 0) {
          int cnt4 = m_cnt4;
          asm volatile(
              "vld1.32  {d0-d1},  [%[in_y]]!  \n" /*  load y to q0    */
              "1:\n"
              "vmax.f32 q1, q0,   %q[vzero]   \n" /*      q0 relu     */
              "vld1.32  {d0-d1},  [%[in_y]]!  \n" /*   load y to q0   */
              "subs %[cnt], %[cnt], #1        \n" /*      sub cnt     */
              "vst1.32  {d2-d3},  [%[out_y]]! \n" /*  store q1 to y   */
              "bne  1b                        \n" /* branch to label 1*/
              "sub  %[in_y],  %[in_y],  #16   \n" /*   restore in_y   */
              : [cnt] "+r"(cnt4), [in_y] "+r"(in_y), [out_y] "+r"(y)
              : [vzero] "w"(vzero)
              : "q0", "q1", "cc", "memory");
        }
        for (int r = 0; r < m_remain; ++r) {
          y[r] = in_y[r] > 0.f ? in_y[r] : 0.f;
        }
      } else if (act == lite_api::ActivationType::kRelu6) {
        float32x4_t vsix = vdupq_n_f32(six);
        if (m_cnt4 > 0) {
          int cnt4 = m_cnt4;
          asm volatile(
              "vld1.32  {d0-d1},  [%[in_y]]!  \n" /*  load y to q0    */
              "1:\n"
              "vmax.f32 q1, q0,   %q[vzero]   \n" /*      q0 relu6    */
              "vld1.32  {d0-d1},  [%[in_y]]!  \n" /*   load y to q0   */
              "vmin.f32 q1, q1,   %q[vsix]    \n" /*      q0 relu6    */
              "subs %[cnt], %[cnt], #1        \n" /*      sub cnt     */
              "vst1.32  {d2-d3},  [%[out_y]]! \n" /*  store q1 to y   */
              "bne  1b                        \n" /* branch to label 1*/
              "sub  %[in_y],  %[in_y],  #16   \n" /*   restore in_y   */
              : [cnt] "+r"(cnt4), [in_y] "+r"(in_y), [out_y] "+r"(y)
              : [vzero] "w"(vzero), [vsix] "w"(vsix)
              : "q0", "q1", "cc", "memory");
        }
        for (int r = 0; r < m_remain; ++r) {
          y[r] = in_y[r] > 0.f ? in_y[r] : 0.f;
          y[r] = y[r] > six ? six : y[r];
        }
      } else if (act == lite_api::ActivationType::kLeakyRelu) {
        float32x4_t valpha = vdupq_n_f32(alpha);
        if (m_cnt4 > 0) {
          int cnt4 = m_cnt4;
          asm volatile(
              "1:\n"
              "vld1.32  {d0-d1}, [%[in_y]]!   \n" /*   load y to q0   */
              "vcge.f32 q3, q0,  %q[vzero]    \n" /*    vcgeq_f32     */
              "vmul.f32 q4, q0,  %q[valpha]   \n" /*    vmulq_f32     */
              "vbif q0, q4, q3                \n" /*      choose      */
              "subs %[cnt], %[cnt], #1        \n" /*      sub cnt     */
              "vst1.32  {d0-d1}, [%[out_y]]!  \n" /*  store q0 to y   */
              "bne  1b                        \n" /* branch to label 1*/
              : [cnt] "+r"(cnt4), [in_y] "+r"(in_y), [out_y] "+r"(y)
              : [vzero] "w"(vzero), [valpha] "w"(valpha)
              : "q0", "q3", "q4", "cc", "memory");
        }
        for (int r = 0; r < m_remain; ++r) {
          y[r] = in_y[r] < 0.f ? alpha * in_y[r] : in_y[r];
        }
      }
    } else {
      memcpy(y, y_buf, M * sizeof(float));
    }
  }
}
#endif  // __aarch64__

// clang-format off
//! define compute kernel
#ifdef __aarch64__
#define SGEMV_IN_8                                    \
  "prfm  pldl1keep, [%[in]]   \n" /* preload din */   \
  "prfm  pldl1keep, [%[w0]]   \n" /* preload w0 */    \
  "prfm  pldl1keep, [%[w1]]   \n" /* preload w1 */    \
  "prfm  pldl1keep, [%[w2]]   \n" /* preload w2 */    \
  "prfm  pldl1keep, [%[w3]]   \n" /* preload w3 */    \
  "prfm  pldl1keep, [%[w4]]   \n" /* preload w4 */    \
  "prfm  pldl1keep, [%[w5]]   \n" /* preload w5 */    \
  "prfm  pldl1keep, [%[w6]]   \n" /* preload w6 */    \
  "prfm  pldl1keep, [%[w7]]   \n" /* preload w7 */    \
  "movi   v0.4s,  #0          \n" /* set out0 to 0 */ \
  "movi   v1.4s,  #0          \n" /* set out1 to 0 */ \
  "movi   v2.4s,  #0          \n" /* set out2 to 0 */ \
  "movi   v3.4s,  #0          \n" /* set out3 to 0 */ \
  "movi   v4.4s,  #0          \n" /* set out4 to 0 */ \
  "movi   v5.4s,  #0          \n" /* set out5 to 0 */ \
  "movi   v6.4s,  #0          \n" /* set out6 to 0 */ \
  "movi   v7.4s,  #0          \n" /* set out7 to 0 */

#define SGEMV_IN_8_BIAS                                    \
  "ldp   q8, q9, [%[bias_ptr]]\n" /* load bias to q8, q9*/ \
  "prfm  pldl1keep, [%[in]]   \n" /* preload din */        \
  "prfm  pldl1keep, [%[w0]]   \n" /* preload w0 */         \
  "prfm  pldl1keep, [%[w1]]   \n" /* preload w1 */         \
  "prfm  pldl1keep, [%[w2]]   \n" /* preload w2 */         \
  "prfm  pldl1keep, [%[w3]]   \n" /* preload w3 */         \
  "prfm  pldl1keep, [%[w4]]   \n" /* preload w4 */         \
  "prfm  pldl1keep, [%[w5]]   \n" /* preload w5 */         \
  "prfm  pldl1keep, [%[w6]]   \n" /* preload w6 */         \
  "prfm  pldl1keep, [%[w7]]   \n" /* preload w7 */         \
  "movi   v0.4s,  #0          \n" /* set out0 to 0 */      \
  "movi   v1.4s,  #0          \n" /* set out1 to 0 */      \
  "movi   v2.4s,  #0          \n" /* set out2 to 0 */      \
  "movi   v3.4s,  #0          \n" /* set out3 to 0 */      \
  "movi   v4.4s,  #0          \n" /* set out4 to 0 */      \
  "movi   v5.4s,  #0          \n" /* set out5 to 0 */      \
  "movi   v6.4s,  #0          \n" /* set out6 to 0 */      \
  "movi   v7.4s,  #0          \n" /* set out7 to 0 */      \
  "ins    v0.s[0], v8.s[0]    \n" /* out0 = bias0 */       \
  "ins    v1.s[0], v8.s[1]    \n" /* out1 = bias1 */       \
  "ins    v2.s[0], v8.s[2]    \n" /* out2 = bias2 */       \
  "ins    v3.s[0], v8.s[3]    \n" /* out3 = bias3 */       \
  "ins    v4.s[0], v9.s[0]    \n" /* out4 = bias4 */       \
  "ins    v5.s[0], v9.s[1]    \n" /* out5 = bias5 */       \
  "ins    v6.s[0], v9.s[2]    \n" /* out6 = bias6 */       \
  "ins    v7.s[0], v9.s[3]    \n" /* out7 = bias7 */

#define SGEMV_IN_1                                    \
  "prfm  pldl1keep, [%[in]]   \n" /* preload din */   \
  "prfm  pldl1keep, [%[w0]]   \n" /* preload w0 */    \
  "movi   v0.4s,  #0          \n" /* set out0 to 0 */ \
  "movi   v1.4s,  #0          \n" /* set out0 to 0 */

#define SGEMV_IN_1_BIAS                               \
  "prfm  pldl1keep, [%[in]]   \n" /* preload din */   \
  "prfm  pldl1keep, [%[w0]]   \n" /* preload w0 */    \
  "movi   v0.4s,  #0          \n" /* set out0 to 0 */ \
  "movi   v1.4s,  #0          \n" /* set out0 to 0 */ \
  "fmov   s0,  %w[bias0]      \n" /* set out0 = bias0 */

#define SGEMV_KERNEL_8                                                         \
  /* check main loop */                                                        \
  "cmp %w[cnt], #1            \n" /* check whether has main loop */            \
  "blt  2f                    \n" /* jump to tail */ /* main loop */           \
  "1:                         \n"                    /* main loop */           \
  "ldp q8, q9, [%[in]], #32   \n"                    /* load input 8 float */  \
  "ldp q10, q11, [%[w0]], #32 \n"                    /* load w0 8 float */     \
  "ldp q12, q13, [%[w1]], #32 \n"                    /* load w1 8 float */     \
  "ldp q14, q15, [%[w2]], #32 \n"                    /* load w2 8 float */     \
  "ldp q16, q17, [%[w3]], #32 \n"                    /* load w3 8 float */     \
  "ldp q18, q19, [%[w4]], #32 \n"                    /* load w4 8 float */     \
  "ldp q20, q21, [%[w5]], #32 \n"                    /* load w5 8 float */     \
  "ldp q22, q23, [%[w6]], #32 \n"                    /* load w6 8 float */     \
  "ldp q24, q25, [%[w7]], #32 \n"                    /* load w7 8 float */     \
  "fmla v0.4s, v8.4s, v10.4s  \n"                    /* mul + add*/            \
  "fmla v1.4s, v8.4s, v12.4s  \n"                    /* mul + add*/            \
  "fmla v2.4s, v8.4s, v14.4s  \n"                    /* mul + add*/            \
  "fmla v3.4s, v8.4s, v16.4s  \n"                    /* mul + add*/            \
  "fmla v4.4s, v8.4s, v18.4s  \n"                    /* mul + add*/            \
  "fmla v5.4s, v8.4s, v20.4s  \n"                    /* mul + add*/            \
  "fmla v6.4s, v8.4s, v22.4s  \n"                    /* mul + add*/            \
  "fmla v7.4s, v8.4s, v24.4s  \n"                    /* mul + add*/            \
  "subs %w[cnt], %w[cnt], #1  \n"                    /* sub main loop count */ \
  "fmla v0.4s, v9.4s, v11.4s  \n"                    /* mul + add*/            \
  "fmla v1.4s, v9.4s, v13.4s  \n"                    /* mul + add*/            \
  "fmla v2.4s, v9.4s, v15.4s  \n"                    /* mul + add*/            \
  "fmla v3.4s, v9.4s, v17.4s  \n"                    /* mul + add*/            \
  "fmla v4.4s, v9.4s, v19.4s  \n"                    /* mul + add*/            \
  "fmla v5.4s, v9.4s, v21.4s  \n"                    /* mul + add*/            \
  "fmla v6.4s, v9.4s, v23.4s  \n"                    /* mul + add*/            \
  "fmla v7.4s, v9.4s, v25.4s  \n"                    /* mul + add*/            \
  "bne 1b                     \n" /* jump to main loop */                      \
  /* pair add to final result */                                               \
  "2:                         \n"  /* reduce to scale */                       \
  "faddp  v16.4s, v0.4s, v0.4s\n"  /* pair add to vector */                    \
  "faddp  s8, v16.2s          \n"  /* pair add to scale */                     \
  "faddp  v17.4s, v1.4s, v1.4s\n"  /* pair add to vector */                    \
  "faddp  s9, v17.2s          \n"  /* pair add to scale */                     \
  "faddp  v18.4s, v2.4s, v2.4s\n"  /* pair add to vector */                    \
  "faddp  s10, v18.2s         \n"  /* pair add to scale */                     \
  "faddp  v19.4s, v3.4s, v3.4s\n"  /* pair add to vector */                    \
  "faddp  s11, v19.2s         \n"  /* pair add to scale */                     \
  "faddp  v20.4s, v4.4s, v4.4s\n"  /* pair add to vector */                    \
  "faddp  s12, v20.2s         \n"  /* pair add to scale */                     \
  "faddp  v21.4s, v5.4s, v5.4s\n"  /* pair add to vector */                    \
  "faddp  s13, v21.2s         \n"  /* pair add to scale */                     \
  "faddp  v22.4s, v6.4s, v6.4s\n"  /* pair add to vector */                    \
  "faddp  s14, v22.2s          \n" /* pair add to scale */                     \
  "faddp  v23.4s, v7.4s, v7.4s\n"  /* pair add to vector */                    \
  "faddp  s15, v23.2s          \n" /* pair add to scale */                     \
  "cmp %w[tail], #1           \n"  /* check whether has tail */                \
  "blt  4f                    \n"  /* jump to end */                           \
  "3:                         \n"  /* tail loop */                             \
  "ldr     s16, [%[in]], #4   \n"  /* load in, 1 float */                      \
  "ldr     s17, [%[w0]], #4   \n"  /* load w0, 1 float */                      \
  "ldr     s18, [%[w1]], #4   \n"  /* load w1, 1 float */                      \
  "ldr     s19, [%[w2]], #4   \n"  /* load w2, 1 float */                      \
  "ldr     s20, [%[w3]], #4   \n"  /* load w3, 1 float */                      \
  "ldr     s21, [%[w4]], #4   \n"  /* load w4, 1 float */                      \
  "ldr     s22, [%[w5]], #4   \n"  /* load w5, 1 float */                      \
  "ldr     s23, [%[w6]], #4   \n"  /* load w6, 1 float */                      \
  "ldr     s24, [%[w7]], #4   \n"  /* load w7, 1 float */                      \
  "fmadd   s8, s16, s17, s8   \n"  /* mul + add */                             \
  "fmadd   s9, s16, s18, s9   \n"  /* mul + add */                             \
  "fmadd   s10, s16, s19, s10 \n"  /* mul + add */                             \
  "fmadd   s11, s16, s20, s11 \n"  /* mul + add */                             \
  "fmadd   s12, s16, s21, s12 \n"  /* mul + add */                             \
  "fmadd   s13, s16, s22, s13 \n"  /* mul + add */                             \
  "fmadd   s14, s16, s23, s14 \n"  /* mul + add */                             \
  "fmadd   s15, s16, s24, s15 \n"  /* mul + add */                             \
  "subs %w[tail], %w[tail], #1\n"  /* sub tail loop count */                   \
  "bne 3b                     \n"  /* jump to tail loop */

#define SGEMV_KERNEL_1                                                         \
  /* check main loop */                                                        \
  "cmp %w[cnt], #1            \n" /* check whether has main loop */            \
  "blt  2f                    \n" /* jump to tail */                           \
  "1:                         \n" /* main loop */                              \
  "ldp q8, q9, [%[in]], #32   \n" /* load input 8 float */                     \
  "ldp q10, q11, [%[w0]], #32 \n" /* load w0 8 float */                        \
  "fmla v0.4s, v8.4s, v10.4s  \n" /* mul + add*/                               \
  "subs %w[cnt], %w[cnt], #1  \n" /* sub main loop count */                    \
  "fmla v1.4s, v9.4s, v11.4s  \n" /* mul + add*/                               \
  "bne 1b                     \n" /* jump to main loop */                      \
  /* pair add to final result */                                               \
  "2:                         \n" /* reduce to scale */                        \
  "fadd   v9.4s, v0.4s, v1.4s \n" /* add 2 vector */                           \
  "faddp  v10.4s, v9.4s, v9.4s\n" /* pair add to vector */                     \
  "faddp  s8, v10.2s          \n" /* pair add to scale */                      \
  "cmp %w[tail], #1           \n" /* check whether has tail */                 \
  "blt  4f                    \n" /* jump to end */                            \
  "3:                         \n" /* tail loop */                              \
  "ldr     s16, [%[in]], #4   \n" /* load in, 1 float */                       \
  "ldr     s17, [%[w0]], #4   \n" /* load w0, 1 float */                       \
  "fmadd   s8, s16, s17, s8   \n" /* mul + add */                              \
  "subs %w[tail], %w[tail], #1\n" /* sub tail loop count */                    \
  "bne 3b                     \n" /* jump to tail loop */

#define SGEMV_OUT_8                                      \
  /* end */                                              \
  "4:                          \n" /* end */             \
  "mov v8.s[1], v9.s[0]        \n" /* ins s9 to  v8[1]*/ \
  "mov v8.s[2], v10.s[0]       \n" /* ins s10 to v8[2]*/ \
  "mov v8.s[3], v11.s[0]       \n" /* ins s11 to v8[3]*/ \
  "mov v9.s[0], v12.s[0]       \n" /* ins s12 to v9[0]*/ \
  "mov v9.s[1], v13.s[0]       \n" /* ins s13 to v9[1]*/ \
  "mov v9.s[2], v14.s[0]       \n" /* ins s14 to v9[2]*/ \
  "mov v9.s[3], v15.s[0]       \n" /* ins s15 to v9[3]*/ \
  "stp q8, q9, [%[out]]        \n" /* save result */

#define SGEMV_OUT_8_RELU                                   \
  /* end */                                                \
  "4:                          \n" /* end */               \
  "mov v8.s[1], v9.s[0]        \n" /* ins s9 to  v8[1]*/   \
  "mov v8.s[2], v10.s[0]       \n" /* ins s10 to v8[2]*/   \
  "mov v8.s[3], v11.s[0]       \n" /* ins s11 to v8[3]*/   \
  "mov v9.s[0], v12.s[0]       \n" /* ins s12 to v9[0]*/   \
  "mov v9.s[1], v13.s[0]       \n" /* ins s13 to v9[1]*/   \
  "mov v9.s[2], v14.s[0]       \n" /* ins s14 to v9[2]*/   \
  "mov v9.s[3], v15.s[0]       \n" /* ins s15 to v9[3]*/   \
  "movi   v2.4s, #0            \n" /* zero data for relu */\
  "fmax   v8.4s, v8.4s, v2.4s  \n" /* relu */              \
  "fmax   v9.4s, v9.4s, v2.4s  \n" /* relu */              \
  "stp q8, q9, [%[out]]        \n" /* save result */

#define SGEMV_OUT_8_RELU6                                       \
  /* end */                                                     \
  "4:                              \n" /* end */                \
  "mov v8.s[1], v9.s[0]            \n" /* ins s9 to  v8[1]*/    \
  "mov v8.s[2], v10.s[0]           \n" /* ins s10 to v8[2]*/    \
  "mov v8.s[3], v11.s[0]           \n" /* ins s11 to v8[3]*/    \
  "mov v9.s[0], v12.s[0]           \n" /* ins s12 to v9[0]*/    \
  "mov v9.s[1], v13.s[0]           \n" /* ins s13 to v9[1]*/    \
  "mov v9.s[2], v14.s[0]           \n" /* ins s14 to v9[2]*/    \
  "mov v9.s[3], v15.s[0]           \n" /* ins s15 to v9[3]*/    \
  "movi   v2.4s, #0                \n" /* zero data for relu6 */\
  "fmax   v8.4s, v8.4s, v2.4s      \n" /* relu6 */              \
  "fmax   v9.4s, v9.4s, v2.4s      \n" /* relu6 */              \
  "fmin   v8.4s, v8.4s, %[vsix].4s \n" /* relu */               \
  "fmin   v9.4s, v9.4s, %[vsix].4s \n" /* relu */               \
  "stp q8, q9, [%[out]]            \n" /* save result */

#define SGEMV_OUT_8_LEAKEY_RELU                                         \
  /* end */                                                             \
  "4:                               \n" /* end */                       \
  "mov v8.s[1], v9.s[0]             \n" /* ins s9 to  v8[1]*/           \
  "mov v8.s[2], v10.s[0]            \n" /* ins s10 to v8[2]*/           \
  "mov v8.s[3], v11.s[0]            \n" /* ins s11 to v8[3]*/           \
  "mov v9.s[0], v12.s[0]            \n" /* ins s12 to v9[0]*/           \
  "mov v9.s[1], v13.s[0]            \n" /* ins s13 to v9[1]*/           \
  "mov v9.s[2], v14.s[0]            \n" /* ins s14 to v9[2]*/           \
  "mov v9.s[3], v15.s[0]            \n" /* ins s15 to v9[3]*/           \
  "movi   v2.4s, #0                 \n" /* zero data for leakey relu */ \
  "fcmge v4.4s, v8.4s,  v2.4s       \n" /* vcgeq_f32 */                 \
  "fmul v5.4s, v8.4s,  %[valpha].4s \n" /* vmulq_f32 */                 \
  "fcmge v6.4s, v9.4s,  v2.4s       \n" /* vcgeq_f32 */                 \
  "fmul v7.4s, v9.4s,  %[valpha].4s \n" /* vmulq_f32 */                 \
  "bif v8.16b, v5.16b, v4.16b       \n" /* choose*/                     \
  "bif v9.16b, v7.16b, v6.16b       \n" /* choose*/                     \
  "stp q8, q9, [%[out]]             \n" /* save result */

#define SGEMV_OUT_1                                 \
  /* end */                                         \
  "4:                         \n" /* end */         \
  "str s8, [%[out]]           \n" /* save result */

#define SGEMV_OUT_1_RELU                                   \
  /* end */                                                \
  "4:                         \n" /* end */                \
  "movi   d1, #0              \n" /* zero data for relu */ \
  "fmax   s8, s8, s1          \n" /* relu */               \
  "str s8, [%[out]]           \n" /* save result */

#define SGEMV_OUT_1_RELU6                                   \
  /* end */                                                 \
  "4:                         \n" /* end */                 \
  "movi   d1, #0              \n" /* zero data for relu6 */ \
  "fmov   s2, %w[six]         \n" /* mov six to s2  */      \
  "fmax   s8, s8, s1          \n" /* relu6 */               \
  "fmin   s8, s8, s2          \n" /* relu6 */               \
  "str s8, [%[out]]           \n" /* save result */

#define SGEMV_OUT_1_LEAKEY_RELU                             \
  /* end */                                                 \
  "4:                           \n" /* end */               \
  "fmov   s1, %w[alpha]         \n" /* mov alpha to s1  */  \
  "fcmp   s8, #0.0              \n" /* cmp with zero*/      \
  "bge    5f                    \n" /* if ge zero */        \
  "fmul   s8, s8, s1            \n" /* out * alpha */       \
  "5:                           \n" /* leakey relu label */ \
  "str s8, [%[out]]             \n" /* save result */

#define SGEMV_OUT_8_BETA                                 \
  /* end */                                              \
  "4:                          \n" /* end */             \
  "mov v8.s[1], v9.s[0]        \n" /* ins s9 to  v8[1]*/ \
  "ldp q0, q1, [%[out]]        \n"                       \
  "mov v8.s[2], v10.s[0]       \n" /* ins s10 to v8[2]*/ \
  "mov v8.s[3], v11.s[0]       \n" /* ins s11 to v8[3]*/ \
  "mov v9.s[0], v12.s[0]       \n" /* ins s12 to v9[0]*/ \
  "mov v9.s[1], v13.s[0]       \n" /* ins s13 to v9[1]*/ \
  "fmla v8.4s, v0.4s, %[vbeta].4s\n"                    \
  "mov v9.s[2], v14.s[0]       \n" /* ins s14 to v9[2]*/ \
  "mov v9.s[3], v15.s[0]       \n" /* ins s15 to v9[3]*/ \
  "fmla v9.4s, v1.4s, %[vbeta].4s\n"                    \
  "stp q8, q9, [%[out]]        \n" /* save result */

#define SGEMV_OUT_8_RELU_BETA                              \
  /* end */                                                \
  "4:                          \n" /* end */               \
  "mov v8.s[1], v9.s[0]        \n" /* ins s9 to  v8[1]*/   \
  "ldp q0, q1, [%[out]]        \n"                         \
  "mov v8.s[2], v10.s[0]       \n" /* ins s10 to v8[2]*/   \
  "mov v8.s[3], v11.s[0]       \n" /* ins s11 to v8[3]*/   \
  "mov v9.s[0], v12.s[0]       \n" /* ins s12 to v9[0]*/   \
  "mov v9.s[1], v13.s[0]       \n" /* ins s13 to v9[1]*/   \
  "mov v9.s[2], v14.s[0]       \n" /* ins s14 to v9[2]*/   \
  "mov v9.s[3], v15.s[0]       \n" /* ins s15 to v9[3]*/   \
  "movi   v2.4s, #0            \n" /* zero data for relu */\
  "fmax   v8.4s, v8.4s, v2.4s  \n" /* relu */              \
  "fmax   v9.4s, v9.4s, v2.4s  \n" /* relu */              \
  "fmla v8.4s, v0.4s, %[vbeta].4s\n"                      \
  "fmla v9.4s, v1.4s, %[vbeta].4s\n"                      \
  "stp q8, q9, [%[out]]        \n" /* save result */

#define SGEMV_OUT_8_RELU6_BETA                                  \
  /* end */                                                     \
  "4:                              \n" /* end */                \
  "mov v8.s[1], v9.s[0]            \n" /* ins s9 to  v8[1]*/    \
  "ldp q0, q1, [%[out]]        \n"                              \
  "mov v8.s[2], v10.s[0]           \n" /* ins s10 to v8[2]*/    \
  "mov v8.s[3], v11.s[0]           \n" /* ins s11 to v8[3]*/    \
  "mov v9.s[0], v12.s[0]           \n" /* ins s12 to v9[0]*/    \
  "mov v9.s[1], v13.s[0]           \n" /* ins s13 to v9[1]*/    \
  "mov v9.s[2], v14.s[0]           \n" /* ins s14 to v9[2]*/    \
  "mov v9.s[3], v15.s[0]           \n" /* ins s15 to v9[3]*/    \
  "movi   v2.4s, #0                \n" /* zero data for relu6 */\
  "fmax   v8.4s, v8.4s, v2.4s      \n" /* relu6 */              \
  "fmax   v9.4s, v9.4s, v2.4s      \n" /* relu6 */              \
  "fmin   v8.4s, v8.4s, %[vsix].4s \n" /* relu */               \
  "fmin   v9.4s, v9.4s, %[vsix].4s \n" /* relu */               \
  "fmla v8.4s, v0.4s, %[vbeta].4s\n"                           \
  "fmla v9.4s, v1.4s, %[vbeta].4s\n"                           \
  "stp q8, q9, [%[out]]            \n" /* save result */

#define SGEMV_OUT_8_LEAKEY_RELU_BETA                                    \
  /* end */                                                             \
  "4:                               \n" /* end */                       \
  "mov v8.s[1], v9.s[0]             \n" /* ins s9 to  v8[1]*/           \
  "ldp q0, q1, [%[out]]        \n"                                      \
  "mov v8.s[2], v10.s[0]            \n" /* ins s10 to v8[2]*/           \
  "mov v8.s[3], v11.s[0]            \n" /* ins s11 to v8[3]*/           \
  "mov v9.s[0], v12.s[0]            \n" /* ins s12 to v9[0]*/           \
  "mov v9.s[1], v13.s[0]            \n" /* ins s13 to v9[1]*/           \
  "mov v9.s[2], v14.s[0]            \n" /* ins s14 to v9[2]*/           \
  "mov v9.s[3], v15.s[0]            \n" /* ins s15 to v9[3]*/           \
  "movi   v2.4s, #0                 \n" /* zero data for leakey relu */ \
  "fcmge v4.4s, v8.4s,  v2.4s       \n" /* vcgeq_f32 */                 \
  "fmul v5.4s, v8.4s,  %[valpha].4s \n" /* vmulq_f32 */                 \
  "fcmge v6.4s, v9.4s,  v2.4s       \n" /* vcgeq_f32 */                 \
  "fmul v7.4s, v9.4s,  %[valpha].4s \n" /* vmulq_f32 */                 \
  "bif v8.16b, v5.16b, v4.16b       \n" /* choose*/                     \
  "bif v9.16b, v7.16b, v6.16b       \n" /* choose*/                     \
  "fmla v8.4s, v0.4s, %[vbeta].4s\n"                                   \
  "fmla v9.4s, v1.4s, %[vbeta].4s\n"                                   \
  "stp q8, q9, [%[out]]             \n" /* save result */

#define SGEMV_OUT_1_BETA                            \
  /* end */                                         \
  "4:                         \n" /* end */         \
  "ldr s1, [%[out]]           \n"                   \
  "fmov   s2, %w[beta]        \n"                   \
  "fmul s1, s1, s2            \n"                   \
  "fadd s8, s8, s1            \n"                   \
  "str s8, [%[out]]           \n" /* save result */

#define SGEMV_OUT_1_RELU_BETA                              \
  /* end */                                                \
  "4:                         \n" /* end */                \
  "movi   d1, #0              \n" /* zero data for relu */ \
  "ldr s4, [%[out]]           \n"                          \
  "fmov   s5, %w[beta]        \n"                          \
  "fmax   s8, s8, s1          \n" /* relu */               \
  "fmul s4, s4, s5            \n"                          \
  "fadd s8, s8, s4            \n"                          \
  "str s8, [%[out]]           \n" /* save result */

#define SGEMV_OUT_1_RELU6_BETA                              \
  /* end */                                                 \
  "4:                         \n" /* end */                 \
  "movi   d1, #0              \n" /* zero data for relu6 */ \
  "fmov   s2, %w[six]         \n" /* mov six to s2  */      \
  "ldr s4, [%[out]]           \n"                           \
  "fmax   s8, s8, s1          \n" /* relu6 */               \
  "fmov   s5, %w[beta]        \n"                           \
  "fmin   s8, s8, s2          \n" /* relu6 */               \
  "fmul s4, s4, s5            \n"                           \
  "fadd s8, s8, s4            \n"                           \
  "str s8, [%[out]]           \n" /* save result */

#define SGEMV_OUT_1_LEAKEY_RELU_BETA                        \
  /* end */                                                 \
  "4:                           \n" /* end */               \
  "fmov   s1, %w[alpha]         \n" /* mov alpha to s1  */  \
  "ldr s4, [%[out]]           \n"                           \
  "fcmp   s8, #0.0              \n" /* cmp with zero*/      \
  "fmov   s5, %w[beta]        \n"                           \
  "bge    5f                    \n" /* if ge zero */        \
  "fmul   s8, s8, s1            \n" /* out * alpha */       \
  "5:                           \n" /* leakey relu label */ \
  "fmul s4, s4, s5            \n"                           \
  "fadd s8, s8, s4            \n"                           \
  "str s8, [%[out]]             \n" /* save result */

#else  // __aarch64__

#define SGEMV_IN_4                                                    \
  "pld [%[in]]                    @ preload cache line, input\n"      \
  "pld [%[w0]]                    @ preload cache line, weights r0\n" \
  "pld [%[w1]]                    @ preload cache line, weights r1\n" \
  "pld [%[w2]]                    @ preload cache line, weights r2\n" \
  "pld [%[w3]]                    @ preload cache line, weights r3\n" \
  "vmov.u32 q0, #0                @ set q0 to 0\n"                    \
  "vmov.u32 q1, #0                @ set q1 to 0\n"                    \
  "vmov.u32 q2, #0                @ set q2 to 0\n"                    \
  "vmov.u32 q3, #0                @ set q3 to 0\n"                    \
  "pld [%[w0], #64]               @ preload cache line, weights r0\n" \
  "pld [%[w1], #64]               @ preload cache line, weights r1\n" \
  "pld [%[w2], #64]               @ preload cache line, weights r2\n" \
  "pld [%[w3], #64]               @ preload cache line, weights r3\n"

#define SGEMV_IN_4_BIAS                                               \
  "pld [%[in]]                    @ preload cache line, input\n"      \
  "pld [%[w0]]                    @ preload cache line, weights r0\n" \
  "pld [%[w1]]                    @ preload cache line, weights r1\n" \
  "pld [%[w2]]                    @ preload cache line, weights r2\n" \
  "pld [%[w3]]                    @ preload cache line, weights r3\n" \
  "vmov.u32 q0, #0                @ set q0 to 0\n"                    \
  "vmov.u32 q1, #0                @ set q1 to 0\n"                    \
  "vmov.u32 q2, #0                @ set q2 to 0\n"                    \
  "vmov.u32 q3, #0                @ set q3 to 0\n"                    \
  "vmov s0, %[bias0]              @ set q0 to bias0\n"                \
  "vmov s4, %[bias1]              @ set q1 to bias1\n"                \
  "vmov s8, %[bias2]              @ set q2 to bias2\n"                \
  "vmov s12,%[bias3]              @ set q3 to bias3\n"                \
  "pld [%[w0], #64]               @ preload cache line, weights r0\n" \
  "pld [%[w1], #64]               @ preload cache line, weights r1\n" \
  "pld [%[w2], #64]               @ preload cache line, weights r2\n" \
  "pld [%[w3], #64]               @ preload cache line, weights r3\n"

#define SGEMV_IN_1                                                        \
  "pld [%[in]]                        @ preload cache line, input\n"      \
  "pld [%[w0]]                        @ preload cache line, weights r0\n" \
  "vmov.u32 q0, #0                    @ set q0 to 0\n"

#define SGEMV_IN_1_BIAS                                                   \
  "pld [%[in]]                        @ preload cache line, input\n"      \
  "pld [%[w0]]                        @ preload cache line, weights r0\n" \
  "vmov.u32 q0, #0                    @ set q0 to 0\n"                    \
  "vmov s0, %[bias0]                  @ set q0 to 0\n"

#define SGEMV_KERNEL_4                                                         \
  /* check main loop */                                                        \
  "cmp %[cnt], #1                 @ check whether has main loop\n"             \
  "blt  2f                        @ jump to tail\n"                            \
  "1:                             @ main loop\n"                               \
  "vld1.32 {d8-d11}, [%[in]]!     @ load input, q4, q5\n"                      \
  "vld1.32 {d12-d15}, [%[w0]]!    @ load weights r0, q6,q7\n"                  \
  "vld1.32 {d16-d19}, [%[w1]]!    @ load weights r1, q8,q9\n"                  \
  "vmla.f32 q0, q4, q6            @ mul add\n"                                 \
  "vld1.32 {d20-d23}, [%[w2]]!    @ load weights r2, q10,q11\n"                \
  "vmla.f32 q1, q4, q8            @ mul add\n"                                 \
  "vld1.32 {d24-d27}, [%[w3]]!    @ load weights r3, q12,q13\n"                \
  /*"vmla.f32 q0, q4, q6            @ mul add\n" */                            \
  /*"vmla.f32 q1, q4, q8            @ mul add\n" */                            \
  "vmla.f32 q2, q4, q10           @ mul add\n"                                 \
  "vmla.f32 q3, q4, q12           @ mul add\n"                                 \
  "subs %[cnt], #1                @ sub loop count \n"                         \
  "vmla.f32 q0, q5, q7            @ mul add\n"                                 \
  "vmla.f32 q1, q5, q9            @ mul add\n"                                 \
  "vmla.f32 q2, q5, q11           @ mul add\n"                                 \
  "vmla.f32 q3, q5, q13           @ mul add\n"                                 \
  "bne 1b                         @ jump to main loop\n"                       \
  "2:                             @ pair add \n"                               \
  "vpadd.f32 d8, d0, d1           @ pair add, first step\n"                    \
  "vpadd.f32 d9, d2, d3           @ pair add, first step\n"                    \
  "vpadd.f32 d10, d4, d5          @ pair add, first step\n"                    \
  "vpadd.f32 d11, d6, d7          @ pair add, first step\n"                    \
  "vpadd.f32 d0, d8, d9           @ pair add, second step\n"                   \
  "vpadd.f32 d1, d10, d11         @ pair add, second step\n"                   \
  "cmp %[tail], #1                @ check whether has tail\n"                  \
  "blt  4f                        @ jump to end\n"                             \
  "3:                             @ tail loop\n"                               \
  "vldm     %[in]!, {s16}         @ load 1 float\n"                            \
  "vldm     %[w0]!, {s17}         @ load 1 float\n"                            \
  "vldm     %[w1]!, {s18}         @ load 1 float\n"                            \
  "vldm     %[w2]!, {s19}         @ load 1 float\n"                            \
  "vldm     %[w3]!, {s20}         @ load 1 float\n"                            \
  "vmla.f32   s0, s16, s17        @ mul + add\n"                               \
  "vmla.f32   s1, s16, s18        @ mul + add\n"                               \
  "vmla.f32   s2, s16, s19        @ mul + add\n"                               \
  "vmla.f32   s3, s16, s20        @ mul + add\n"                               \
  "subs %[tail], #1               @ sub loop count \n"                         \
  "bne 3b                         @ jump to tail loop\n"

#define SGEMV_KERNEL_1                                                         \
  "cmp %[cnt], #1                     @ check whether has main loop\n"         \
  "blt  2f                            @ jump to tail\n"                        \
  "1:                                 @ main loop\n"                           \
  "vld1.32 {d24-d27}, [%[in]]!        @ load input, q12,q13\n"                 \
  "vld1.32 {d28-d31}, [%[w0]]!        @ load weights r0, q14, q15\n"           \
  "vmla.f32 q0, q12, q14              @ mul add\n"                             \
  "vmla.f32 q0, q13, q15              @ mul add\n"                             \
  "subs %[cnt] , #1                   @ sub loop count \n"                     \
  "bne 1b                             @ jump to main loop\n"                   \
  "2:                                 @ end processing\n"                      \
  "vpadd.f32 d2, d0, d1               @ pair add, first step\n"                \
  "vpadd.f32 d0, d2, d2               @ pair add, final step\n"                \
  "cmp %[tail], #1                    @ check whether has mid cols\n"          \
  "blt  4f                            @ jump to end\n"                         \
  "3:                                 @ tail loop\n"                           \
  "vldm     %[in]!, {s16}             @ load 1 float\n"                        \
  "vldm     %[w0]!, {s17}             @ load 1 float\n"                        \
  "vmla.f32   s0, s16, s17            @ mul + add\n"                           \
  "subs %[tail], #1                   @ sub loop count \n"                     \
  "bne 3b                             @ jump to tail loop\n"

#define SGEMV_OUT_4                        \
  /* end */                                \
  "4:                             @ end\n" \
  "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

#define SGEMV_OUT_4_RELU                             \
  /* end */                                          \
  "4:                             @ end\n"           \
  "vmov.i32   q1, #0              @ zero for relu\n" \
  "vmax.f32   q0, q0, q1          @ relu\n"          \
  "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

#define SGEMV_OUT_4_RELU6                             \
  /* end */                                           \
  "4:                             @ end\n"            \
  "vmov.i32   q1, #0              @ zero for relu6\n" \
  "vdup.f32   q2, %[six]          @ six for relu6\n"  \
  "vmax.f32   q0, q0, q1          @ relu6\n"          \
  "vmin.f32   q0, q0, q2          @ relu6\n"          \
  "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

#define SGEMV_OUT_4_LEAKEY_RELU                              \
  /* end */                                                  \
  "4:                             @ end\n"                   \
  "vmov.i32   q1, #0              @ zero for leakey relu\n"  \
  "vdup.f32   q2, %[alpha]        @ alpha for leakey relu\n" \
  "vcge.f32   q3, q0, q1          @ vcgeq_f32 \n"            \
  "vmul.f32   q4, q0, q2          @ vmulq_f32 \n"            \
  "vbif q0,   q4, q3              @ choose \n"               \
  "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

#define SGEMV_OUT_1                        \
  /* end */                                \
  "4:                             @ end\n" \
  "vst1.32 {d0[0]}, [%[out]]      @ save result\n"

#define SGEMV_OUT_1_RELU                             \
  /* end */                                          \
  "4:                             @ end\n"           \
  "vmov.i32   d1, #0              @ zero for relu\n" \
  "vmax.f32   d0, d0, d1          @ relu\n"          \
  "vst1.32 {d0[0]}, [%[out]]      @ save result\n"

#define SGEMV_OUT_1_RELU6                             \
  /* end */                                           \
  "4:                             @ end\n"            \
  "vmov.i32   d1, #0              @ zero for relu6\n" \
  "vdup.f32   d4, %[six]          @ six  for relu6\n" \
  "vmax.f32   d0, d0, d1          @ relu6\n"          \
  "vmin.f32   d0, d0, d4          @ relu6\n"          \
  "vst1.32 {d0[0]}, [%[out]]      @ save result\n"

#define SGEMV_OUT_1_LEAKEY_RELU                                \
  /* end */                                                    \
  "4:                               @ end\n"                   \
  "vmov.i32   d2, #0                @ zero  for leakey relu\n" \
  "vdup.f32   d3, %[alpha]          @ alpha for leakey relu\n" \
  "vcge.f32   d6, d0, d2            @ vcgeq_f32 \n"            \
  "vmul.f32   d8, d0, d3            @ vmulq_f32 \n"            \
  "vbif d0,   d8, d6                @ choose \n"               \
  "vst1.32 {d0[0]}, [%[out]]        @ save result\n"

#define SGEMV_OUT_4_BETA                   \
  /* end */                                \
  "4:                             @ end\n" \
  "vld1.32 {d2-d3}, [%[out]]      \n"      \
  "vmla.f32 q0, q1, %q[vbeta]      \n"      \
  "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

#define SGEMV_OUT_4_RELU_BETA                        \
  /* end */                                          \
  "4:                             @ end\n"           \
  "vmov.i32   q1, #0              @ zero for relu\n" \
  "vld1.32 {d4-d5}, [%[out]]      \n"                \
  "vmax.f32   q0, q0, q1          @ relu\n"          \
  "vmla.f32 q0, q2, %q[vbeta]      \n"                \
  "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

#define SGEMV_OUT_4_RELU6_BETA                        \
  /* end */                                           \
  "4:                             @ end\n"            \
  "vmov.i32   q1, #0              @ zero for relu6\n" \
  "vdup.f32   q2, %[six]          @ six for relu6\n"  \
  "vld1.32 {d6-d7}, [%[out]]      \n"                 \
  "vmax.f32   q0, q0, q1          @ relu6\n"          \
  "vmin.f32   q0, q0, q2          @ relu6\n"          \
  "vmla.f32 q0, q3, %q[vbeta]      \n"                 \
  "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

#define SGEMV_OUT_4_LEAKEY_RELU_BETA                         \
  /* end */                                                  \
  "4:                             @ end\n"                   \
  "vmov.i32   q1, #0              @ zero for leakey relu\n"  \
  "vdup.f32   q2, %[alpha]        @ alpha for leakey relu\n" \
  "vld1.32 {d10-d11}, [%[out]]      \n"                      \
  "vcge.f32   q3, q0, q1          @ vcgeq_f32 \n"            \
  "vmul.f32   q4, q0, q2          @ vmulq_f32 \n"            \
  "vbif q0,   q4, q3              @ choose \n"               \
  "vmla.f32 q0, q5, %q[vbeta]      \n"                        \
  "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

#define SGEMV_OUT_1_BETA                   \
  /* end */                                \
  "4:                             @ end\n" \
  "vld1.32 {d2}, [%[out]]         \n"      \
  "vdup.f32   d4, %[beta]         \n"      \
  "vmla.f32 d0, d2, d4            \n"      \
  "vst1.32 {d0[0]}, [%[out]]      @ save result\n"

#define SGEMV_OUT_1_RELU_BETA                        \
  /* end */                                          \
  "4:                             @ end\n"           \
  "vmov.i32   d1, #0              @ zero for relu\n" \
  "vld1.32 {d2}, [%[out]]         \n"                \
  "vmax.f32   d0, d0, d1          @ relu\n"          \
  "vdup.f32   d4, %[beta]         \n"                \
  "vmla.f32 d0, d2, d4           \n"                 \
  "vst1.32 {d0[0]}, [%[out]]      @ save result\n"

#define SGEMV_OUT_1_RELU6_BETA                        \
  /* end */                                           \
  "4:                             @ end\n"            \
  "vmov.i32   d1, #0              @ zero for relu6\n" \
  "vdup.f32   d4, %[six]          @ six  for relu6\n" \
  "vld1.32 {d2}, [%[out]]         \n"                 \
  "vmax.f32   d0, d0, d1          @ relu6\n"          \
  "vdup.f32   d6, %[beta]         \n"                 \
  "vmin.f32   d0, d0, d4          @ relu6\n"          \
  "vmla.f32 d0, d2, d6             \n"                \
  "vst1.32 {d0[0]}, [%[out]]      @ save result\n"

#define SGEMV_OUT_1_LEAKEY_RELU_BETA                           \
  /* end */                                                    \
  "4:                               @ end\n"                   \
  "vmov.i32   d2, #0                @ zero  for leakey relu\n" \
  "vdup.f32   d3, %[alpha]          @ alpha for leakey relu\n" \
  "vld1.32 {d4}, [%[out]]           \n"                        \
  "vcge.f32   d6, d0, d2            @ vcgeq_f32 \n"            \
  "vmul.f32   d8, d0, d3            @ vmulq_f32 \n"            \
  "vdup.f32   d2, %[beta]         \n"                \
  "vbif d0,   d8, d6                @ choose \n"               \
  "vmla.f32 d0, d4, d2        \n"                        \
  "vst1.32 {d0[0]}, [%[out]]        @ save result\n"

#endif
// clang-format on

void sgemv(const int M,
           const int N,
           const float *A,
           const float *x,
           float *y,
           float beta,
           bool flag_bias,
           const float *bias) {
  float *data_out = y;
  const float *data_in = x;
  const float *weights_ptr = A;

  int cnt = N >> 3;
  int tail = N & 7;
  bool has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  float32x4_t vbeta = vdupq_n_f32(beta);

#ifdef __aarch64__
  int out_cnt = M >> 3;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    const float *ptr_w4 = ptr_w3 + N;
    const float *ptr_w5 = ptr_w4 + N;
    const float *ptr_w6 = ptr_w5 + N;
    const float *ptr_w7 = ptr_w6 + N;
    const float *bias_ptr = bias + out_idx;
    float bias_local[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (flag_bias) {
      bias_local[0] = bias_ptr[0];
      bias_local[1] = bias_ptr[1];
      bias_local[2] = bias_ptr[2];
      bias_local[3] = bias_ptr[3];
      bias_local[4] = bias_ptr[4];
      bias_local[5] = bias_ptr[5];
      bias_local[6] = bias_ptr[6];
      bias_local[7] = bias_ptr[7];
    }
    int cnt_loop = cnt;
    int tail_loop = tail;
    // clang-format off
    if (has_beta) {
      asm volatile(SGEMV_IN_8_BIAS SGEMV_KERNEL_8 SGEMV_OUT_8_BETA
                 : [in] "+r"(ptr_in),
                   [w0] "+r"(ptr_w0),
                   [w1] "+r"(ptr_w1),
                   [w2] "+r"(ptr_w2),
                   [w3] "+r"(ptr_w3),
                   [w4] "+r"(ptr_w4),
                   [w5] "+r"(ptr_w5),
                   [w6] "+r"(ptr_w6),
                   [w7] "+r"(ptr_w7),
                   [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out), [bias_ptr] "r"(bias_local), [vbeta] "w" (vbeta)
                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                   "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                   "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                   "v24", "v25", "cc", "memory");
    } else {
      asm volatile(SGEMV_IN_8_BIAS SGEMV_KERNEL_8 SGEMV_OUT_8
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [w4] "+r"(ptr_w4),
                    [w5] "+r"(ptr_w5),
                    [w6] "+r"(ptr_w6),
                    [w7] "+r"(ptr_w7),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out), [bias_ptr] "r"(bias_local)
                  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                    "v24", "v25", "cc", "memory");
    }
    // clang-format on
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 8; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = 0.f;
    if (flag_bias) {
      bias0 = bias[j];
    }
    if (has_beta) {
      asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_BETA
                   : [in] "+r"(ptr_in),
                     [w0] "+r"(ptr_w0),
                     [cnt] "+r"(cnt_loop),
                     [tail] "+r"(tail_loop)
                   : [out] "r"(ptr_out), [bias0] "r"(bias0), [beta] "r"(beta)
                   : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "cc");
    } else {
      asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1
                   : [in] "+r"(ptr_in),
                     [w0] "+r"(ptr_w0),
                     [cnt] "+r"(cnt_loop),
                     [tail] "+r"(tail_loop)
                   : [out] "r"(ptr_out), [bias0] "r"(bias0)
                   : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "cc");
    }
  }
#else  // __aarch64__
  int out_cnt = M >> 2;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 4;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    float bias0 = 0.f;
    float bias1 = 0.f;
    float bias2 = 0.f;
    float bias3 = 0.f;
    if (flag_bias) {
      bias0 = bias[out_idx];
      bias1 = bias[out_idx + 1];
      bias2 = bias[out_idx + 2];
      bias3 = bias[out_idx + 3];
    }
    int cnt_loop = cnt;
    int tail_loop = tail;
    // clang-format off
    if (has_beta) {
        asm volatile(SGEMV_IN_4_BIAS SGEMV_KERNEL_4 SGEMV_OUT_4
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out),
                    [bias0] "r"(bias0),
                    [bias1] "r"(bias1),
                    [bias2] "r"(bias2),
                    [bias3] "r"(bias3),
                    [vbeta] "w" (vbeta)
                  : "q0", "q1", "q2", "q3", "q4",
                    "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "cc",
                    "memory");
    } else {
        asm volatile(SGEMV_IN_4_BIAS SGEMV_KERNEL_4 SGEMV_OUT_4
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out),
                    [bias0] "r"(bias0),
                    [bias1] "r"(bias1),
                    [bias2] "r"(bias2),
                    [bias3] "r"(bias3)
                  : "q0", "q1", "q2", "q3", "q4",
                    "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "cc",
                    "memory");
    }
    // clang-format on
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 4; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = 0.f;
    if (flag_bias) {
      bias0 = bias[j];
    }
    if (has_beta) {
      asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_BETA
                   : [in] "+r"(ptr_in),
                     [w0] "+r"(ptr_w0),
                     [cnt] "+r"(cnt_loop),
                     [tail] "+r"(tail_loop)
                   : [out] "r"(ptr_out), [bias0] "r"(bias0), [beta] "r"(beta)
                   : "q0", "q1", "q12", "q13", "q14", "q15", "cc", "memory");
    } else {
      asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1
                   : [in] "+r"(ptr_in),
                     [w0] "+r"(ptr_w0),
                     [cnt] "+r"(cnt_loop),
                     [tail] "+r"(tail_loop)
                   : [out] "r"(ptr_out), [bias0] "r"(bias0)
                   : "q0", "q1", "q12", "q13", "q14", "q15", "cc", "memory");
    }
  }
#endif  // __aarch64__
}

void sgemv_relu(const int M,
                const int N,
                const float *A,
                const float *x,
                float *y,
                float beta,
                bool flag_bias,
                const float *bias) {
  float *data_out = y;
  const float *data_in = x;
  const float *weights_ptr = A;

  int cnt = N >> 3;
  int tail = N & 7;
  bool has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  float32x4_t vbeta = vdupq_n_f32(beta);

#ifdef __aarch64__
  int out_cnt = M >> 3;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    const float *ptr_w4 = ptr_w3 + N;
    const float *ptr_w5 = ptr_w4 + N;
    const float *ptr_w6 = ptr_w5 + N;
    const float *ptr_w7 = ptr_w6 + N;
    const float *bias_ptr = bias + out_idx;
    float bias_local[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (flag_bias) {
      bias_local[0] = bias_ptr[0];
      bias_local[1] = bias_ptr[1];
      bias_local[2] = bias_ptr[2];
      bias_local[3] = bias_ptr[3];
      bias_local[4] = bias_ptr[4];
      bias_local[5] = bias_ptr[5];
      bias_local[6] = bias_ptr[6];
      bias_local[7] = bias_ptr[7];
    }
    int cnt_loop = cnt;
    int tail_loop = tail;
    // clang-format off
    if (has_beta) {
      asm volatile(SGEMV_IN_8_BIAS SGEMV_KERNEL_8 SGEMV_OUT_8_RELU_BETA
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [w4] "+r"(ptr_w4),
                    [w5] "+r"(ptr_w5),
                    [w6] "+r"(ptr_w6),
                    [w7] "+r"(ptr_w7),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out), [bias_ptr] "r"(bias_local), [vbeta] "w"(vbeta)
                  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                    "v24", "v25", "cc", "memory");
    } else {
      asm volatile(SGEMV_IN_8_BIAS SGEMV_KERNEL_8 SGEMV_OUT_8_RELU
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [w4] "+r"(ptr_w4),
                    [w5] "+r"(ptr_w5),
                    [w6] "+r"(ptr_w6),
                    [w7] "+r"(ptr_w7),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out), [bias_ptr] "r"(bias_local)
                  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                    "v24", "v25", "cc", "memory");
    }
    // clang-format on
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 8; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = 0.f;
    if (flag_bias) {
      bias0 = bias[j];
    }
    if (has_beta) {
      asm volatile(
          SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_RELU_BETA
          : [in] "+r"(ptr_in),
            [w0] "+r"(ptr_w0),
            [cnt] "+r"(cnt_loop),
            [tail] "+r"(tail_loop)
          : [out] "r"(ptr_out), [bias0] "r"(bias0), [beta] "r"(beta)
          : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "cc", "memory");
    } else {
      asm volatile(
          SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_RELU
          : [in] "+r"(ptr_in),
            [w0] "+r"(ptr_w0),
            [cnt] "+r"(cnt_loop),
            [tail] "+r"(tail_loop)
          : [out] "r"(ptr_out), [bias0] "r"(bias0)
          : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "cc", "memory");
    }
  }
#else  // __aarch64__
  int out_cnt = M >> 2;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 4;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    float bias0 = 0.f;
    float bias1 = 0.f;
    float bias2 = 0.f;
    float bias3 = 0.f;
    if (flag_bias) {
      bias0 = bias[out_idx];
      bias1 = bias[out_idx + 1];
      bias2 = bias[out_idx + 2];
      bias3 = bias[out_idx + 3];
    }
    int cnt_loop = cnt;
    int tail_loop = tail;
    // clang-format off
    if (has_beta) {
        asm volatile(SGEMV_IN_4_BIAS SGEMV_KERNEL_4 SGEMV_OUT_4_RELU_BETA
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out),
                    [bias0] "r"(bias0),
                    [bias1] "r"(bias1),
                    [bias2] "r"(bias2),
                    [bias3] "r"(bias3),
                    [vbeta] "w"(vbeta)
                  : "q0", "q1", "q2", "q3", "q4",
                    "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "cc",
                    "memory");
    } else {
      asm volatile(SGEMV_IN_4_BIAS SGEMV_KERNEL_4 SGEMV_OUT_4_RELU
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out),
                    [bias0] "r"(bias0),
                    [bias1] "r"(bias1),
                    [bias2] "r"(bias2),
                    [bias3] "r"(bias3)
                  : "q0", "q1", "q2", "q3", "q4",
                    "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "cc",
                    "memory");
    }
    // clang-format on
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 4; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = 0.f;
    if (flag_bias) {
      bias0 = bias[j];
    }
    if (has_beta) {
      asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_RELU_BETA
                   : [in] "+r"(ptr_in),
                     [w0] "+r"(ptr_w0),
                     [cnt] "+r"(cnt_loop),
                     [tail] "+r"(tail_loop)
                   : [out] "r"(ptr_out), [bias0] "r"(bias0), [beta] "r"(beta)
                   : "q0", "q1", "q12", "q13", "q14", "q15", "cc", "memory");
    } else {
      asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_RELU
                   : [in] "+r"(ptr_in),
                     [w0] "+r"(ptr_w0),
                     [cnt] "+r"(cnt_loop),
                     [tail] "+r"(tail_loop)
                   : [out] "r"(ptr_out), [bias0] "r"(bias0)
                   : "q0", "q1", "q12", "q13", "q14", "q15", "cc", "memory");
    }
  }
#endif  // __aarch64__
}

void sgemv_relu6(const int M,
                 const int N,
                 const float *A,
                 const float *x,
                 float *y,
                 float beta,
                 bool flag_bias,
                 const float *bias,
                 const float six) {
  float *data_out = y;
  const float *data_in = x;
  const float *weights_ptr = A;

  int cnt = N >> 3;
  int tail = N & 7;
  float32x4_t vsix = vdupq_n_f32(six);
  bool has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  float32x4_t vbeta = vdupq_n_f32(beta);

#ifdef __aarch64__
  int out_cnt = M >> 3;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    const float *ptr_w4 = ptr_w3 + N;
    const float *ptr_w5 = ptr_w4 + N;
    const float *ptr_w6 = ptr_w5 + N;
    const float *ptr_w7 = ptr_w6 + N;
    const float *bias_ptr = bias + out_idx;
    float bias_local[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (flag_bias) {
      bias_local[0] = bias_ptr[0];
      bias_local[1] = bias_ptr[1];
      bias_local[2] = bias_ptr[2];
      bias_local[3] = bias_ptr[3];
      bias_local[4] = bias_ptr[4];
      bias_local[5] = bias_ptr[5];
      bias_local[6] = bias_ptr[6];
      bias_local[7] = bias_ptr[7];
    }
    int cnt_loop = cnt;
    int tail_loop = tail;
    // clang-format off
    if (has_beta) {
      asm volatile(SGEMV_IN_8_BIAS SGEMV_KERNEL_8 SGEMV_OUT_8_RELU6_BETA
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [w4] "+r"(ptr_w4),
                    [w5] "+r"(ptr_w5),
                    [w6] "+r"(ptr_w6),
                    [w7] "+r"(ptr_w7),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out), [bias_ptr] "r"(bias_local), 
                    [vsix] "w" (vsix), [vbeta] "w"(vbeta)
                  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                    "v24", "v25", "cc", "memory");
    } else {
      asm volatile(SGEMV_IN_8_BIAS SGEMV_KERNEL_8 SGEMV_OUT_8_RELU6
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [w4] "+r"(ptr_w4),
                    [w5] "+r"(ptr_w5),
                    [w6] "+r"(ptr_w6),
                    [w7] "+r"(ptr_w7),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out), [bias_ptr] "r"(bias_local), 
                    [vsix] "w" (vsix)
                  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                    "v24", "v25", "cc", "memory");
    }
    // clang-format on
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 8; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = 0.f;
    if (flag_bias) {
      bias0 = bias[j];
    }
    if (has_beta) {
      asm volatile(
          SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_RELU6_BETA
          : [in] "+r"(ptr_in),
            [w0] "+r"(ptr_w0),
            [cnt] "+r"(cnt_loop),
            [tail] "+r"(tail_loop)
          : [out] "r"(ptr_out),
            [bias0] "r"(bias0),
            [six] "r"(six),
            [beta] "r"(beta)
          : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "cc", "memory");
    } else {
      asm volatile(
          SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_RELU6
          : [in] "+r"(ptr_in),
            [w0] "+r"(ptr_w0),
            [cnt] "+r"(cnt_loop),
            [tail] "+r"(tail_loop)
          : [out] "r"(ptr_out), [bias0] "r"(bias0), [six] "r"(six)
          : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "cc", "memory");
    }
  }
#else  // __aarch64__
  int out_cnt = M >> 2;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 4;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    float bias0 = 0.f;
    float bias1 = 0.f;
    float bias2 = 0.f;
    float bias3 = 0.f;
    if (flag_bias) {
      bias0 = bias[out_idx];
      bias1 = bias[out_idx + 1];
      bias2 = bias[out_idx + 2];
      bias3 = bias[out_idx + 3];
    }
    int cnt_loop = cnt;
    int tail_loop = tail;
    // clang-format off
    if (has_beta) {
      asm volatile(SGEMV_IN_4_BIAS SGEMV_KERNEL_4 SGEMV_OUT_4_RELU6_BETA
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out),
                    [bias0] "r"(bias0),
                    [bias1] "r"(bias1),
                    [bias2] "r"(bias2),
                    [bias3] "r"(bias3),
                    [six] "r" (six),
                    [vbeta] "w"(vbeta)
                  : "q0", "q1", "q2", "q3", "q4",
                    "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "cc",
                    "memory");
    } else {
      asm volatile(SGEMV_IN_4_BIAS SGEMV_KERNEL_4 SGEMV_OUT_4_RELU6
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out),
                    [bias0] "r"(bias0),
                    [bias1] "r"(bias1),
                    [bias2] "r"(bias2),
                    [bias3] "r"(bias3),
                    [six] "r" (six)
                  : "q0", "q1", "q2", "q3", "q4",
                    "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "cc",
                    "memory");
    }
    // clang-format on
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 4; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = 0.f;
    if (flag_bias) {
      bias0 = bias[j];
    }
    if (has_beta) {
      asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_RELU6_BETA
                   : [in] "+r"(ptr_in),
                     [w0] "+r"(ptr_w0),
                     [cnt] "+r"(cnt_loop),
                     [tail] "+r"(tail_loop)
                   : [out] "r"(ptr_out),
                     [bias0] "r"(bias0),
                     [six] "r"(six),
                     [beta] "r"(beta)
                   : "q0", "q1", "q12", "q13", "q14", "q15", "cc", "memory");
    } else {
      asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_RELU6
                   : [in] "+r"(ptr_in),
                     [w0] "+r"(ptr_w0),
                     [cnt] "+r"(cnt_loop),
                     [tail] "+r"(tail_loop)
                   : [out] "r"(ptr_out), [bias0] "r"(bias0), [six] "r"(six)
                   : "q0", "q1", "q12", "q13", "q14", "q15", "cc", "memory");
    }
  }
#endif  // __aarch64__
}

void sgemv_leakey_relu(const int M,
                       const int N,
                       const float *A,
                       const float *x,
                       float *y,
                       float beta,
                       bool flag_bias,
                       const float *bias,
                       const float alpha) {
  float *data_out = y;
  const float *data_in = x;
  const float *weights_ptr = A;
  int cnt = N >> 3;
  int tail = N & 7;
  float32x4_t valpha = vdupq_n_f32(alpha);
  bool has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  float32x4_t vbeta = vdupq_n_f32(beta);
#ifdef __aarch64__
  int out_cnt = M >> 3;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    const float *ptr_w4 = ptr_w3 + N;
    const float *ptr_w5 = ptr_w4 + N;
    const float *ptr_w6 = ptr_w5 + N;
    const float *ptr_w7 = ptr_w6 + N;
    const float *bias_ptr = bias + out_idx;
    float bias_local[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (flag_bias) {
      bias_local[0] = bias_ptr[0];
      bias_local[1] = bias_ptr[1];
      bias_local[2] = bias_ptr[2];
      bias_local[3] = bias_ptr[3];
      bias_local[4] = bias_ptr[4];
      bias_local[5] = bias_ptr[5];
      bias_local[6] = bias_ptr[6];
      bias_local[7] = bias_ptr[7];
    }
    int cnt_loop = cnt;
    int tail_loop = tail;
    // clang-format off
    if (has_beta) {
      asm volatile(SGEMV_IN_8_BIAS SGEMV_KERNEL_8 SGEMV_OUT_8_LEAKEY_RELU_BETA
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [w4] "+r"(ptr_w4),
                    [w5] "+r"(ptr_w5),
                    [w6] "+r"(ptr_w6),
                    [w7] "+r"(ptr_w7),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out), [bias_ptr] "r"(bias_local), 
                    [valpha] "w" (valpha), [vbeta] "w"(vbeta)
                  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                    "v24", "v25", "cc", "memory");
    } else {
      asm volatile(SGEMV_IN_8_BIAS SGEMV_KERNEL_8 SGEMV_OUT_8_LEAKEY_RELU
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [w4] "+r"(ptr_w4),
                    [w5] "+r"(ptr_w5),
                    [w6] "+r"(ptr_w6),
                    [w7] "+r"(ptr_w7),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out), [bias_ptr] "r"(bias_local), 
                    [valpha] "w" (valpha)
                  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                    "v24", "v25", "cc", "memory");
    }
    // clang-format on
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 8; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = 0.f;
    if (flag_bias) {
      bias0 = bias[j];
    }
    if (has_beta) {
      asm volatile(
          SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_LEAKEY_RELU_BETA
          : [in] "+r"(ptr_in),
            [w0] "+r"(ptr_w0),
            [cnt] "+r"(cnt_loop),
            [tail] "+r"(tail_loop)
          : [out] "r"(ptr_out),
            [bias0] "r"(bias0),
            [alpha] "r"(alpha),
            [beta] "r"(beta)
          : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "cc", "memory");
    } else {
      asm volatile(
          SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_LEAKEY_RELU
          : [in] "+r"(ptr_in),
            [w0] "+r"(ptr_w0),
            [cnt] "+r"(cnt_loop),
            [tail] "+r"(tail_loop)
          : [out] "r"(ptr_out), [bias0] "r"(bias0), [alpha] "r"(alpha)
          : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "cc", "memory");
    }
  }
#else  // __aarch64__
  int out_cnt = M >> 2;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 4;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    float bias0 = 0.f;
    float bias1 = 0.f;
    float bias2 = 0.f;
    float bias3 = 0.f;
    if (flag_bias) {
      bias0 = bias[out_idx];
      bias1 = bias[out_idx + 1];
      bias2 = bias[out_idx + 2];
      bias3 = bias[out_idx + 3];
    }
    int cnt_loop = cnt;
    int tail_loop = tail;
    // clang-format off
    if (has_beta) {
      asm volatile(SGEMV_IN_4_BIAS SGEMV_KERNEL_4 SGEMV_OUT_4_LEAKEY_RELU_BETA
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out),
                    [bias0] "r"(bias0),
                    [bias1] "r"(bias1),
                    [bias2] "r"(bias2),
                    [bias3] "r"(bias3),
                    [alpha] "r" (alpha),
                    [vbeta] "w"(vbeta)
                  : "q0", "q1", "q2", "q3", "q4",
                    "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "cc",
                    "memory");
    } else {
      asm volatile(SGEMV_IN_4_BIAS SGEMV_KERNEL_4 SGEMV_OUT_4_LEAKEY_RELU
                  : [in] "+r"(ptr_in),
                    [w0] "+r"(ptr_w0),
                    [w1] "+r"(ptr_w1),
                    [w2] "+r"(ptr_w2),
                    [w3] "+r"(ptr_w3),
                    [cnt] "+r"(cnt_loop),
                    [tail] "+r"(tail_loop)
                  : [out] "r"(ptr_out),
                    [bias0] "r"(bias0),
                    [bias1] "r"(bias1),
                    [bias2] "r"(bias2),
                    [bias3] "r"(bias3),
                    [alpha] "r" (alpha)
                  : "q0", "q1", "q2", "q3", "q4",
                    "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "cc",
                    "memory");
    }
    // clang-format on
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 4; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = 0.f;
    if (flag_bias) {
      bias0 = bias[j];
    }
    if (has_beta) {
      asm volatile(
          SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_LEAKEY_RELU_BETA
          : [in] "+r"(ptr_in),
            [w0] "+r"(ptr_w0),
            [cnt] "+r"(cnt_loop),
            [tail] "+r"(tail_loop)
          : [out] "r"(ptr_out),
            [bias0] "r"(bias0),
            [alpha] "r"(alpha),
            [beta] "r"(beta)
          : "q0", "q1", "q3", "q4", "q12", "q13", "q14", "q15", "cc", "memory");
    } else {
      asm volatile(
          SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_LEAKEY_RELU
          : [in] "+r"(ptr_in),
            [w0] "+r"(ptr_w0),
            [cnt] "+r"(cnt_loop),
            [tail] "+r"(tail_loop)
          : [out] "r"(ptr_out), [bias0] "r"(bias0), [alpha] "r"(alpha)
          : "q0", "q1", "q3", "q4", "q12", "q13", "q14", "q15", "cc", "memory");
    }
  }
#endif  // __aarch64__
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
