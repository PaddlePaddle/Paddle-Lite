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

#include "lite/backends/arm/math/conv_impl.h"
#include "lite/backends/arm/math/packed_sgemm.h"
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
void input_trans(const float* src,
                 int src_stride,
                 float* dest,
                 int dest_stride);
void output_trans_c4(const float* src,
                     int src_stride,
                     float* dest,
                     int dest_stride) {}
void output_trans(const float* src,
                  int src_stride,
                  float* dest,
                  int dest_stride);
void output_trans_post(const float* src,
                       int src_stride,
                       float* dest,
                       int dest_stride,
                       float bias_value,
                       bool has_relu);
void weight_trans_c4(
    float* dest, const float* src, int ic, int oc, void* workspace) {}
void weight_trans(
    float* dest, const float* src, int ic, int oc, void* workspace);
void conv_compute_6x6_3x3_0(const float* input,
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
  int threads = ctx->threads();

  const int pad_h = param.paddings[0];
  const int pad_w = param.paddings[1];

  int in_n_stride = chin * hin * win;
  int out_n_stride = chout * hout * wout;
  int ic_stride = win * hin;
  int oc_stride = wout * hout;

  int tile_w = (wout + 5) / 6;
  int tile_h = (hout + 5) / 6;
  int size_tile = tile_h * tile_w;
  int m_blocks = (chout + 3) / 4;
  int max_ch = chin > chout ? chin : chout;

  int tile_block = 8;
#ifdef __aarch64__
  tile_block = 16;
#endif
  int block_count = size_tile / tile_block;

  float* tmp_work_space =
      ctx->workspace_data<float>() + ctx->llc_size() / sizeof(float);
  float* g_tmp_data = tmp_work_space;
  int tmp_data_thread_stride = tile_block * (chout + chin) * 64;
  memset(g_tmp_data, 0, threads * tmp_data_thread_stride * sizeof(float));
  float* g_trans_tmp_data = tmp_work_space + threads * tmp_data_thread_stride;
  float* g_trans_remain_tmp_data = g_trans_tmp_data + threads * 64;

  // begin compute
  for (int ni = 0; ni < num; ++ni) {
    const float* input_ptr = input + ni * in_n_stride;
    float* output_ptr = output + ni * out_n_stride;

    const float* weight_ptr = weight;
    const float* bias_ptr = bias;

#pragma omp parallel for num_threads(threads)
    for (int tbi = 0; tbi < block_count; ++tbi) {
#ifdef ARM_WITH_OMP
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

      // int tmp_u_stride = chin * tile_count;
      int tmp_u_stride = chin * tile_block;

      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index % tile_h;

        int src_x = tw_index * 6 - pad_w;
        int src_y = th_index * 6 - pad_h;
        int start_x = src_x > 0 ? 0 : -src_x;
        int end_x = src_x + 8 > win ? win - src_x : 8;
        int start_y = src_y > 0 ? 0 : -src_y;
        int end_y = src_y + 8 > hin ? hin - src_y : 8;

        float* dst_ptr = tmp_data + tbi;
        const float* src_ptr = input_ptr + src_y * win + src_x;
        if (end_x - start_x == 8 && end_y - start_y == 8) {
          // trans input
          for (int ci = 0; ci < chin; ++ci) {
            const float* src_ci = src_ptr + ci * ic_stride;
            for (int i = 0; i < 8; ++i) {
              input_trans(src_ci + i * win, 1, trans_tmp_data + i, 8);
            }
            float* dst_ci = dst_ptr + ci * tile_block;
            for (int i = 0; i < 8; ++i) {
              input_trans(trans_tmp_data + i * 8,
                          1,
                          dst_ci + i * tmp_u_stride,
                          8 * tmp_u_stride);
            }
          }
        } else {
          // trans remain input
          int x_size = end_x - start_x;
          for (int ci = 0; ci < chin; ++ci) {
            const float* src_ci = src_ptr + ci * ic_stride;
            // pad
            memset(trans_remain_tmp_data, 0, 64 * sizeof(float));
            if (x_size > 0) {
              for (int yi = start_y; yi < end_y; ++yi) {
                float* dst_yi = trans_remain_tmp_data + yi * 8 + start_x;
                const float* src_yi = src_ci + win * yi + start_x;
                memcpy(dst_yi, src_yi, x_size * sizeof(float));
              }
            }
            // trans
            for (int i = 0; i < 8; ++i) {
              input_trans(
                  trans_remain_tmp_data + 8 * i, 1, trans_tmp_data + i, 8);
            }
            float* dst_ci = dst_ptr + ci * tile_block;
            for (int i = 0; i < 8; ++i) {
              input_trans(trans_tmp_data + i * 8,
                          1,
                          dst_ci + i * tmp_u_stride,
                          8 * tmp_u_stride);
            }
          }
        }
      }
      // input trans end
      // *begin compute dot
      // *
      // input is 64*ic*tile_count
      // weight is 64 * oc * ic
      float* dst_temp_data = tmp_data + tile_count * 64 * chin;
      const float* a_ptr = weight_ptr;
      float* b_ptr = tmp_data;
      int k = (chin + 3) / 4 - 1;
      int tails = chin & 3;
      if (tile_count == tile_block) {
        for (int i = 0; i < 64; ++i) {
          for (int m = 0; m < m_blocks; ++m) {
            const float* a_ptr = weight_ptr + i * chin * chout + m * 4;
            float* c_ptr0 = dst_temp_data + (i * chout + m) * tile_count;
            float* c_ptr1 = c_ptr0 + tile_count;
            float* c_ptr2 = c_ptr1 + tile_count;
            float* c_ptr3 = c_ptr2 + tile_count;
            asm volatile(
                "vmov.i32 q8, #0\n"
                "vmov.i32 q9, #0\n"
                "vmov.i32 q10, #0\n"
                "vmov.i32 q11, #0\n"
                "vmov.i32 q12, #0\n"
                "vmov.i32 q13, #0\n"
                "vmov.i32 q14, #0\n"
                "vmov.i32 q15, #0\n"
                "11: \n" /* check loop count */
                "vld1.32   {d0-d3}, [%[a_ptr] :128]!   @ load a0~a3\n"
                "vld1.32   {d8-d11}, [%[b_ptr] :128]!  @ load b1\n"
                "cmp %[k], #0                          @ check weather k is "
                "bigger than "
                "0\n"
                "beq 0f                                @ jump to tail\n"
                "1:                                    @ main loop for k\n"
                /* Unroll 0*/
                "vld1.32  {d12-d15}, [%[b_ptr] :128]!  @ load next b1, b2\n"
                "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"
                "vld1.32	{d4-d7}, [%[a_ptr] :128]!   @ load next 2xa0~a3\n"
                "vmla.f32	q10, q4, d0[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q4, d1[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q4, d1[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q5, d0[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q5, d0[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q5, d1[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"
                "vld1.32	{d8-d11}, [%[b_ptr] :128]!  @ load next b1, b2\n"
                /* Unroll 1 */
                "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"
                "pld [%[b_ptr], #64]                    @ preload b\n"
                "vmla.f32	q10, q6, d2[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q6, d3[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q6, d3[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q7, d2[0]               @ out6 += b2 * a0\n"
                "vmla.f32	q11, q7, d2[1]              @ out7 += b2 * a1\n"
                "vmla.f32	q13, q7, d3[0]              @ out8 += b2 * a2\n"
                "vmla.f32	q15, q7, d3[1]              @ out9 += b2 * a3\n"
                "vld1.32	{d12-d15}, [%[b_ptr] :128]! @ load next b1,b2\n"
                /* Unroll 2 */
                "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"
                "vld1.32	{d0-d3}, [%[a_ptr] :128]!   @ load next a0~a3\n"
                "vmla.f32	q10, q4, d4[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q4, d5[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q4, d5[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q5, d4[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q5, d4[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q5, d5[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"
                "vld1.32	{d8-d11}, [%[b_ptr] :128]!  @ load next b1, b2\n"
                /* Unroll 3 */
                "vmla.f32	q8, q6, d6[0]               @ out0 += b1 * a0\n"
                "pld [%[a_ptr], #64]                    @ preload a\n"
                "vmla.f32	q10, q6, d6[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q6, d7[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q6, d7[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q7, d6[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q7, d6[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q7, d7[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q7, d7[1]              @ out7 += b2 * a3\n"
                "subs		%[k], %[k], #1              @ k--\n"
                "bne		1b                          @ jump to main loop\n"
                "0:                                     @ process tail\n"
                "subs		%[tails], %[tails], #1      @ tail--\n"
                "beq		3f                          @ jump to tail = 1\n"
                /* Unroll 0*/
                "vld1.32  {d12-d15}, [%[b_ptr] :128]!   @ load next b1, b2\n"
                "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"
                "vmla.f32	q10, q4, d0[1]              @ out1 += b1 * a1\n"
                "subs		%[tails], %[tails], #1      @ tail--\n"
                "vmla.f32	q12, q4, d1[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q4, d1[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q5, d0[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q5, d0[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q5, d1[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"
                "beq		4f                          @ jump to tail==2\n"
                /* Unroll 1 */
                "vld1.32	{d8-d11}, [%[b_ptr] :128]!  @ load next b1, b2\n"
                "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"
                "vld1.32	{d4-d7}, [%[a_ptr] :128]!   @ load next 2xa0~a3\n"
                "vmla.f32	q10, q6, d2[1]              @ out1 += b1 * a1\n"
                "subs		%[tails], %[tails], #1      @ tail--\n"
                "vmla.f32	q12, q6, d3[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q6, d3[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q7, d2[0]               @ out6 += b2 * a0\n"
                "vmla.f32	q11, q7, d2[1]              @ out7 += b2 * a1\n"
                "vmla.f32	q13, q7, d3[0]              @ out8 += b2 * a2\n"
                "vmla.f32	q15, q7, d3[1]              @ out9 += b2 * a3\n"
                "beq		5f                          @ jump to tail==3\n"
                /* Unroll 2 */
                "vld1.32	{d12-d15}, [%[b_ptr] :128]! @ load next b1,b2\n"
                "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"
                "vmla.f32	q10, q4, d4[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q4, d5[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q4, d5[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q5, d4[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q5, d4[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q5, d5[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"
                /* Unroll 3 */
                "vmla.f32	q8, q6, d6[0]               @ out0 += b1 * a0\n"
                "vmla.f32	q10, q6, d6[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q6, d7[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q6, d7[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q7, d6[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q7, d6[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q7, d7[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q7, d7[1]              @ out7 += b2 * a3\n"
                "b		6f\n"
                /* tails==1 final tail */
                "3:                                     @ tail=1\n"
                "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"
                "vmla.f32	q10, q4, d0[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q4, d1[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q4, d1[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q5, d0[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q5, d0[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q5, d1[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"
                /*aptr - 16 */
                "sub		%[a_ptr], %[a_ptr], #16     @ tail--\n"
                "b		6f                              @ jump to end\n"
                /* tails==2 final tail*/
                "4:                                     @ tail == 2\n"
                "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"
                "vmla.f32	q10, q6, d2[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q6, d3[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q6, d3[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q7, d2[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q7, d2[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q7, d3[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q7, d3[1]              @ out7 += b2 * a3\n"
                "b		6f                              @ jump to end\n"
                /* tails==3 final tail*/
                "5:                                     @ tail=3\n"
                "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"
                "vmla.f32	q10, q4, d4[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q4, d5[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q4, d5[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q5, d4[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q5, d4[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q5, d5[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"
                /*aptr - 16*/
                "sub		%[a_ptr], %[a_ptr], #16     @ tail--\n"
                "6:                                     @ store result\n"
                "vst1.32    {d16-d19},  [%[c_ptr0]]!    @ store r0\n"
                "vst1.32    {d20-d23},  [%[c_ptr1]]!    @ store r1\n"
                "vst1.32    {d24-d27},  [%[c_ptr2]]!    @ store r2\n"
                "vst1.32    {d28-d31},  [%[c_ptr3]]!    @ store r3\n"
                : [a_ptr] "+r"(a_ptr),
                  [b_ptr] "+r"(b_ptr),
                  [c_ptr0] "+r"(c_ptr0),
                  [c_ptr1] "+r"(c_ptr1),
                  [c_ptr2] "+r"(c_ptr2),
                  [c_ptr3] "+r"(c_ptr3),
                  [k] "+r"(k),
                  [tails] "+r"(tails)
                :
                : "q0",
                  "q1",
                  "q2",
                  "q3",
                  "q4",
                  "q5",
                  "q6",
                  "q7",
                  "q8",
                  "q9",
                  "q10",
                  "q11",
                  "q12",
                  "q13",
                  "q14",
                  "q15",
                  "cc",
                  "memory");
          }
        }
      } else {
        for (int i = 0; i < 64; ++i) {
          for (int m = 0; m < m_blocks; ++m) {
            const float* a_ptr = weight_ptr + i * chin * chout + m * 4;
            float c_out0[8];
            float c_out1[8];
            float c_out2[8];
            float c_out3[8];

            float* c_ptr0 = c_out0;
            float* c_ptr1 = c_out1;
            float* c_ptr2 = c_out2;
            float* c_ptr3 = c_out3;

            float* pout0 = dst_temp_data + (i * chout + m) * tile_count;
            float* pout1 = c_ptr0 + tile_count;
            float* pout2 = c_ptr1 + tile_count;
            float* pout3 = c_ptr2 + tile_count;
            asm volatile(
                "vmov.i32 q8, #0\n"
                "vmov.i32 q9, #0\n"
                "vmov.i32 q10, #0\n"
                "vmov.i32 q11, #0\n"
                "vmov.i32 q12, #0\n"
                "vmov.i32 q13, #0\n"
                "vmov.i32 q14, #0\n"
                "vmov.i32 q15, #0\n"
                "11: \n" /* check loop count */
                "vld1.32   {d0-d3}, [%[a_ptr] :128]!   @ load a0~a3\n"
                "vld1.32   {d8-d11}, [%[b_ptr] :128]!  @ load b1\n"
                "cmp %[k], #0                          @ check weather k is "
                "bigger than "
                "0\n"
                "beq 0f                                @ jump to tail\n"
                "1:                                    @ main loop for k\n"
                /* Unroll 0*/
                "vld1.32  {d12-d15}, [%[b_ptr] :128]!  @ load next b1, b2\n"
                "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"
                "vld1.32	{d4-d7}, [%[a_ptr] :128]!   @ load next 2xa0~a3\n"
                "vmla.f32	q10, q4, d0[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q4, d1[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q4, d1[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q5, d0[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q5, d0[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q5, d1[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"
                "vld1.32	{d8-d11}, [%[b_ptr] :128]!  @ load next b1, b2\n"
                /* Unroll 1 */
                "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"
                "pld [%[b_ptr], #64]                    @ preload b\n"
                "vmla.f32	q10, q6, d2[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q6, d3[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q6, d3[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q7, d2[0]               @ out6 += b2 * a0\n"
                "vmla.f32	q11, q7, d2[1]              @ out7 += b2 * a1\n"
                "vmla.f32	q13, q7, d3[0]              @ out8 += b2 * a2\n"
                "vmla.f32	q15, q7, d3[1]              @ out9 += b2 * a3\n"
                "vld1.32	{d12-d15}, [%[b_ptr] :128]! @ load next b1,b2\n"
                /* Unroll 2 */
                "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"
                "vld1.32	{d0-d3}, [%[a_ptr] :128]!   @ load next a0~a3\n"
                "vmla.f32	q10, q4, d4[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q4, d5[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q4, d5[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q5, d4[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q5, d4[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q5, d5[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"
                "vld1.32	{d8-d11}, [%[b_ptr] :128]!  @ load next b1, b2\n"
                /* Unroll 3 */
                "vmla.f32	q8, q6, d6[0]               @ out0 += b1 * a0\n"
                "pld [%[a_ptr], #64]                    @ preload a\n"
                "vmla.f32	q10, q6, d6[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q6, d7[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q6, d7[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q7, d6[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q7, d6[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q7, d7[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q7, d7[1]              @ out7 += b2 * a3\n"
                "subs		%[k], %[k], #1              @ k--\n"
                "bne		1b                          @ jump to main loop\n"
                "0:                                     @ process tail\n"
                "subs		%[tails], %[tails], #1      @ tail--\n"
                "beq		3f                          @ jump to tail = 1\n"
                /* Unroll 0*/
                "vld1.32  {d12-d15}, [%[b_ptr] :128]!   @ load next b1, b2\n"
                "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"
                "vmla.f32	q10, q4, d0[1]              @ out1 += b1 * a1\n"
                "subs		%[tails], %[tails], #1      @ tail--\n"
                "vmla.f32	q12, q4, d1[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q4, d1[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q5, d0[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q5, d0[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q5, d1[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"
                "beq		4f                          @ jump to tail==2\n"
                /* Unroll 1 */
                "vld1.32	{d8-d11}, [%[b_ptr] :128]!  @ load next b1, b2\n"
                "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"
                "vld1.32	{d4-d7}, [%[a_ptr] :128]!   @ load next 2xa0~a3\n"
                "vmla.f32	q10, q6, d2[1]              @ out1 += b1 * a1\n"
                "subs		%[tails], %[tails], #1      @ tail--\n"
                "vmla.f32	q12, q6, d3[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q6, d3[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q7, d2[0]               @ out6 += b2 * a0\n"
                "vmla.f32	q11, q7, d2[1]              @ out7 += b2 * a1\n"
                "vmla.f32	q13, q7, d3[0]              @ out8 += b2 * a2\n"
                "vmla.f32	q15, q7, d3[1]              @ out9 += b2 * a3\n"
                "beq		5f                          @ jump to tail==3\n"
                /* Unroll 2 */
                "vld1.32	{d12-d15}, [%[b_ptr] :128]! @ load next b1,b2\n"
                "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"
                "vmla.f32	q10, q4, d4[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q4, d5[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q4, d5[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q5, d4[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q5, d4[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q5, d5[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"
                /* Unroll 3 */
                "vmla.f32	q8, q6, d6[0]               @ out0 += b1 * a0\n"
                "vmla.f32	q10, q6, d6[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q6, d7[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q6, d7[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q7, d6[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q7, d6[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q7, d7[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q7, d7[1]              @ out7 += b2 * a3\n"
                "b		6f\n"
                /* tails==1 final tail */
                "3:                                     @ tail=1\n"
                "vmla.f32	q8, q4, d0[0]               @ out0 += b1 * a0\n"
                "vmla.f32	q10, q4, d0[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q4, d1[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q4, d1[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q5, d0[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q5, d0[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q5, d1[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q5, d1[1]              @ out7 += b2 * a3\n"
                /*aptr - 16 */
                "sub		%[a_ptr], %[a_ptr], #16     @ tail--\n"
                "b		6f                              @ jump to end\n"
                /* tails==2 final tail*/
                "4:                                     @ tail == 2\n"
                "vmla.f32	q8, q6, d2[0]               @ out0 += b1 * a0\n"
                "vmla.f32	q10, q6, d2[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q6, d3[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q6, d3[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q7, d2[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q7, d2[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q7, d3[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q7, d3[1]              @ out7 += b2 * a3\n"
                "b		6f                              @ jump to end\n"
                /* tails==3 final tail*/
                "5:                                     @ tail=3\n"
                "vmla.f32	q8, q4, d4[0]               @ out0 += b1 * a0\n"
                "vmla.f32	q10, q4, d4[1]              @ out1 += b1 * a1\n"
                "vmla.f32	q12, q4, d5[0]              @ out2 += b1 * a2\n"
                "vmla.f32	q14, q4, d5[1]              @ out3 += b1 * a3\n"
                "vmla.f32	q9, q5, d4[0]               @ out4 += b2 * a0\n"
                "vmla.f32	q11, q5, d4[1]              @ out5 += b2 * a1\n"
                "vmla.f32	q13, q5, d5[0]              @ out6 += b2 * a2\n"
                "vmla.f32	q15, q5, d5[1]              @ out7 += b2 * a3\n"
                /*aptr - 16*/
                "sub		%[a_ptr], %[a_ptr], #16     @ tail--\n"
                "6:                                     @ store result\n"
                "vst1.32    {d16-d19},  [%[c_ptr0]]!    @ store r0\n"
                "vst1.32    {d20-d23},  [%[c_ptr1]]!    @ store r1\n"
                "vst1.32    {d24-d27},  [%[c_ptr2]]!    @ store r2\n"
                "vst1.32    {d28-d31},  [%[c_ptr3]]!    @ store r3\n"
                : [a_ptr] "+r"(a_ptr),
                  [b_ptr] "+r"(b_ptr),
                  [c_ptr0] "+r"(c_ptr0),
                  [c_ptr1] "+r"(c_ptr1),
                  [c_ptr2] "+r"(c_ptr2),
                  [c_ptr3] "+r"(c_ptr3),
                  [k] "+r"(k),
                  [tails] "+r"(tails)
                :
                : "q0",
                  "q1",
                  "q2",
                  "q3",
                  "q4",
                  "q5",
                  "q6",
                  "q7",
                  "q8",
                  "q9",
                  "q10",
                  "q11",
                  "q12",
                  "q13",
                  "q14",
                  "q15",
                  "cc",
                  "memory");
            for (int i = 0; i < tile_count; ++i) {
              *pout0++ = c_out0[i];
              *pout1++ = c_out1[i];
              *pout2++ = c_out2[i];
              *pout3++ = c_out3[i];
            }
          }
        }
      }  // dot end
      // output trans
      int tmp_oc_stride = chout * tile_count;
      for (int ti = 0; ti < tile_count; ++ti) {
        int index = tile_index + ti;

        int tw_index = index % tile_w;
        int th_index = index % tile_h;

        int dst_x = tw_index * 6;
        int dst_y = th_index * 6;

        int ex = dst_x + 6 > wout ? wout - dst_x : 6;
        int ey = dst_y + 6 > hout ? hout - dst_y : 6;

        float* dst_ptr = output_ptr + dst_y * wout + dst_x;
        float* src_ptr = dst_temp_data + ti;
        if (ex == 6) {
          // trans output
          for (int ci = 0; ci < chout; ++ci) {
            float bias_value = param.bias ? bias[ci] : 0.f;
            float* dst_ci = dst_ptr + ci * oc_stride;
            float* src_ci = src_ptr + ci * tile_count;
            for (int i = 0; i < 8; ++i) {
              output_trans(
                  src_ci + i * tmp_oc_stride, 1, trans_tmp_data + i, 8);
            }
            for (int i = 0; i < 6; ++i) {
              output_trans_post(trans_tmp_data + i * 8,
                                1,
                                dst_ci + i * hout,
                                hout,
                                bias_value,
                                param.fuse_relu);
            }
          }
        } else {
          for (int ci = 0; ci < chout; ++ci) {
            float bias_value = param.bias ? bias[ci] : 0.f;
            // trans output
            float* dst_ci = dst_ptr + ci * oc_stride;
            float* src_ci = src_ptr + ci * tile_count;
            for (int i = 0; i < 8; ++i) {
              output_trans(
                  src_ci + i * tmp_oc_stride, 1, trans_tmp_data + i, 8);
            }
            for (int i = 0; i < 6; ++i) {
              output_trans_post(trans_tmp_data + i * 8,
                                1,
                                trans_remain_tmp_data + i,
                                6,
                                bias_value,
                                param.fuse_relu);
            }
            // copy to dest
            for (int i = 0; i < 6; ++i) {
              memcpy(dst_ci + i * wout,
                     trans_remain_tmp_data + i * 6,
                     6 * sizeof(float));
            }
          }
        }
      }
    }  // for block_count
  }    // for num
}  // conv_compute
/*
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
                          ARMContext* ctx);
    int threads = ctx -> threads();

    const int pad_h = param.paddings[0];
    const int pad_w = param.paddings[1];

    int ic_out = (chin + 3) / 4;
    int oc_out = (chout + 3) / 4;

    int in_n_stride = chin * hin * win;
    int out_n_stride = chout * hout * wout;

    int tile_w = (wout + 5) / 6;
    int tile_h = (hout + 5) / 6;
    int size_tile = tile_h * tile_w;
    int max_ch = chin > chout ? chin : chout;

    int tile_block = 8;
#ifdef __aarch64__
    tile_block = 16;
#endif
    int block_count = size_tile / tile_block;

    float* tmp_work_space = ctx->workspace_data<float>() + ctx->llc_size() /
sizeof(float);
    float* tmp_data = tmp_work_space;
    float* trans_tmp_data = tmp_work_space + tile_block * (ic_out + oc_out) *
256;
    float* trans_remain_tmp_data =  trans_tmp_data + 256;

    //begin compute
    for (int ni = 0; ni < num; ++ni){
        const float* input_ptr = input + ni * in_n_stride;
        float* output_ptr = output + ni * out_n_stride;

        float* weight_ptr = weight;
        float* bias_ptr = bias;

        #pragma omp parallel for num_threads(threads)
        for (int tbi = 0; tbi < block_count; ++tbi){
            int tile_index = tbi * tile_block;
            int tile_remain = size_tile - tile_index;
            int tile_count = tile_remain > tile_block ? tile_block :
tile_remain;

            int ic_stride = win * hin * 4;
            int tmp_c_stride = tile_count * 4;
            int tmp_u_stride = chin * tile_count;

            for (int ti = 0; ti < tile_count; ++ti){
                int index = tile_index + ti;

                int tw_index = index % tile_w;
                int th_index = index % tile_h;

                int src_x = tw_index * 6 - pad_w;
                int src_y = th_index * 6 - pad_h;
                int start_x = src_x > 0 ? 0 : -src_x;
                int end_x = src_x + 8 > win ? win - src_x : 8;
                int start_y = src_y > 0 ? 0 : -src_y;
                int end_y = src_y + 8 > hin ? hin - src_y : 8;

                float* dst_ptr = tmp_data + tbi * 4;
                float* src_ptr = input_ptr + (src_y * win + src_x) * 4;
                if (end_x - start_x == 8 && end_y - start_y == 8){
                    //trans input
                    for (int ci = 0; ci < ic_out; ++ci){
                        float src_ci = src_ptr + ci * ic_stride;
                        for (int i = 0; i < 8; ++i){
                            input_trans_c4(src_ci + i * win * 4, 4,
trans_tmp_data + 4 * i, 32);
                        }
                        float* dst_ci = dst_ptr + ci * tmp_c_stride;
                        for (int i = 0; i < 8; ++i){
                            input_trans_c4(trans_tmp_data + i * 32, 4, dst_ci +
i * tmp_u_stride, 8 * tmp_u_stride);
                        }

                    }
                } else {
                    //trans remain input
                    int x_size = end_x - start_x;
                    for (int ci = 0; ci < ic_out; ++ci){
                        float* src_ci = src_ptr + ci * ic_stride;
                        //pad
                        memset(trans_remain_tmp_data, 0, 256 * sizeof(float));
                        if (x_size > 0){
                            for (int yi = start_y; yi < end_y; ++yi){
                                float dst_yi = trans_remain_tmp_data + yi * 32 +
start_x * 4;
                                float src_yi = src_ci + 4 * win * yi + start_x *
4;
                                memcpy(dst_yi, src_yi, x_size * sizeof(float));
                            }
                        }
                        //trans
                        for (int i = 0; i < 8; ++i){
                            input_trans_c4(trans_remain_tmp_data + 32 * i, 4,
trans_tmp_data + 4 * i, 32);
                        }
                        float* dst_ci = dst_ptr + ci * tmp_c_stride;
                        for (int i = 0; i < 8; ++i){
                            input_trans_c4(trans_tmp_data + i * 32, 4, dst_ci +
i * tmp_u_stride, 8 * tmp_u_stride);
                        }
                    }

                }

            }
            //input trans end
            float* dst_temp_data = temp_data + tile_count * 256 * ic_out;

            if (tile_count == tile_block){
                for (int i =0; i < 64; ++i){

                }
            } else {

            }







        }
}//conv_compute
*/
void input_trans(const float* src,
                 int src_stride,
                 float* dest,
                 int dest_stride) {
  const float* src0 = src;
  const float* src1 = src0 + src_stride;
  const float* src2 = src1 + src_stride;
  const float* src3 = src2 + src_stride;
  const float* src4 = src3 + src_stride;
  const float* src5 = src4 + src_stride;
  const float* src6 = src5 + src_stride;
  const float* src7 = src6 + src_stride;

  float* dest0 = dest;
  float* dest1 = dest0 + dest_stride;
  float* dest2 = dest1 + dest_stride;
  float* dest3 = dest2 + dest_stride;
  float* dest4 = dest3 + dest_stride;
  float* dest5 = dest4 + dest_stride;
  float* dest6 = dest5 + dest_stride;
  float* dest7 = dest6 + dest_stride;

  *dest0 = *src0 - *src6 + (*src4 - *src2) * 5.25;
  *dest7 = *src7 - *src1 + (*src3 - *src5) * 5.25;

  float tmp12a = *src2 + *src6 - *src4 * 4.25;
  float tmp12b = *src1 + *src5 - *src3 * 4.25;
  *dest1 = tmp12a + tmp12b;
  *dest2 = tmp12a - tmp12b;

  float tmp34a = *src6 + *src2 * 0.25 - *src4 * 1.25;
  float tmp34b = *src1 * 0.5 - *src3 * 2.5 + *src5 * 2;
  *dest3 = tmp34a + tmp34b;
  *dest4 = tmp34a - tmp34b;

  float tmp56a = *src6 + (*src2 - *src4 * 1.25) * 4;
  float tmp56b = *src1 * 2 - *src3 * 2.5 + *src5 * 0.5;

  *dest5 = tmp56a + tmp56b;
  *dest6 = tmp56a - tmp56b;
}
void output_trans(const float* src,
                  int src_stride,
                  float* dest,
                  int dest_stride) {
  const float* src0 = src;
  const float* src1 = src0 + src_stride;
  const float* src2 = src1 + src_stride;
  const float* src3 = src2 + src_stride;
  const float* src4 = src3 + src_stride;
  const float* src5 = src4 + src_stride;
  const float* src6 = src5 + src_stride;
  const float* src7 = src6 + src_stride;

  float* dest0 = dest;
  float* dest1 = dest0 + dest_stride;
  float* dest2 = dest1 + dest_stride;
  float* dest3 = dest2 + dest_stride;
  float* dest4 = dest3 + dest_stride;
  float* dest5 = dest4 + dest_stride;

  float tmp024a = *src1 + *src2;
  float tmp135a = *src1 - *src2;
  float tmp024b = *src3 + *src4;
  float tmp135b = *src3 - *src4;
  float tmp024c = *src5 + *src6;
  float tmp135c = *src5 - *src6;

  *dest0 = *src0 + tmp024a + tmp024b + tmp024c;
  *dest2 = tmp024a + tmp024b * 4 + tmp024c * 0.25f;
  *dest4 = tmp024a + tmp024b * 16 + tmp024c * 0.0625f;

  *dest1 = tmp135a + tmp135b * 2 + tmp135c * 0.5f;
  *dest3 = tmp135a + tmp135b * 8 + tmp135c * 0.125f;
  *dest5 = *src7 + tmp135a + tmp135b * 32 + tmp135c * 0.03125f;
}
void output_trans_post(const float* src,
                       int src_stride,
                       float* dest,
                       int dest_stride,
                       float bias_value = 0,
                       bool has_relu = false) {
  const float* src0 = src;
  const float* src1 = src0 + src_stride;
  const float* src2 = src1 + src_stride;
  const float* src3 = src2 + src_stride;
  const float* src4 = src3 + src_stride;
  const float* src5 = src4 + src_stride;
  const float* src6 = src5 + src_stride;
  const float* src7 = src6 + src_stride;

  float* dest0 = dest;
  float* dest1 = dest0 + dest_stride;
  float* dest2 = dest1 + dest_stride;
  float* dest3 = dest2 + dest_stride;
  float* dest4 = dest3 + dest_stride;
  float* dest5 = dest4 + dest_stride;

  float tmp024a = *src1 + *src2;
  float tmp135a = *src1 - *src2;
  float tmp024b = *src3 + *src4;
  float tmp135b = *src3 - *src4;
  float tmp024c = *src5 + *src6;
  float tmp135c = *src5 - *src6;

  *dest0 = *src0 + tmp024a + tmp024b + tmp024c + bias_value;
  *dest2 = tmp024a + tmp024b * 4 + tmp024c * 0.25f + bias_value;
  *dest4 = tmp024a + tmp024b * 16 + tmp024c * 0.0625f + bias_value;

  *dest1 = tmp135a + tmp135b * 2 + tmp135c * 0.5f + bias_value;
  *dest3 = tmp135a + tmp135b * 8 + tmp135c * 0.125f + bias_value;
  *dest5 = *src7 + tmp135a + tmp135b * 32 + tmp135c * 0.03125f + bias_value;

  if (has_relu) {
    *dest0 = *dest0 >= 0 ? *dest0 : 0;
    *dest1 = *dest1 >= 0 ? *dest1 : 0;
    *dest2 = *dest2 >= 0 ? *dest2 : 0;
    *dest3 = *dest3 >= 0 ? *dest3 : 0;
    *dest4 = *dest4 >= 0 ? *dest4 : 0;
    *dest5 = *dest5 >= 0 ? *dest5 : 0;
  }
}

void input_trans_c4(const float* src,
                    int src_stride,
                    float* dest,
                    int dest_stride) {
  const float* src0 = src;
  const float* src1 = src0 + src_stride;
  const float* src2 = src1 + src_stride;
  const float* src3 = src2 + src_stride;
  const float* src4 = src3 + src_stride;
  const float* src5 = src4 + src_stride;
  const float* src6 = src5 + src_stride;
  const float* src7 = src6 + src_stride;

  float* dest0 = dest;
  float* dest1 = dest0 + dest_stride;
  float* dest2 = dest1 + dest_stride;
  float* dest3 = dest2 + dest_stride;
  float* dest4 = dest3 + dest_stride;
  float* dest5 = dest4 + dest_stride;
  float* dest6 = dest5 + dest_stride;
  float* dest7 = dest6 + dest_stride;

  for (int i = 0; i < 4; ++i) {
    dest0[i] = src0[i] - src6[i] + (src4[i] - src2[i]) * 5.25;
    dest7[i] = src7[i] - src1[i] + (src3[i] - src5[i]) * 5.25;

    float tmp12a = src2[i] + src6[i] - src4[i] * 4.25;
    float tmp12b = src1[i] + src5[i] - src3[i] * 4.25;
    dest1[i] = tmp12a + tmp12b;
    dest2[i] = tmp12a - tmp12b;

    float tmp34a = src6[i] + src2[i] * 0.25 - src4[i] * 1.25;
    float tmp34b = src1[i] * 0.5 - src3[i] * 2.5 + src5[i] * 2;
    dest3[i] = tmp34a + tmp34b;
    dest4[i] = tmp34a - tmp34b;

    float tmp56a = src6[i] + (src2[i] - src4[i] * 1.25) * 4;
    float tmp56b = src1[i] * 2 - src3[i] * 2.5 + src5[i] * 0.5;

    dest5[i] = tmp56a + tmp56b;
    dest6[i] = tmp56a - tmp56b;
  }
}

void weight_trans(
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

  int c_stride = ch_out * ch_in;
  int col_splits = (ch_out + 3) / 4;
  for (int i = 0; i < ch_out * ch_in * 64; ++i) {
    int dest_c = i / c_stride;
    int dest_h = i / ch_in / 4 + i % ch_in * col_splits;
    int dest_ind = dest_c * c_stride + dest_h * 4;
    dest[dest_ind] = ptr_out[i];
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
