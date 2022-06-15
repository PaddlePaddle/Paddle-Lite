/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cl_common.h>

#define SRC(i, j) src[i * src_width + j]
#define DST(i, j) dst[i * src_height + j]

__kernel void rnn_gemm_4x4(__global const CL_DTYPE* a,
                           __read_only image2d_t b,
                           __read_only image2d_t bias_ih,
                           __read_only image2d_t bias_hh,
                           __global CL_DTYPE* c,
                           const int M,
                           const int N,
                           const int K) {
  const int idy = get_global_id(0) << 2;  // id: [0, M>>2) height of out == M
  const int idx = get_global_id(1);       // id: [0, N>>2) width of out == N

  if ((idx << 2) >= N || idy >= M) return;
  CL_DTYPE16 c_v16 = (CL_DTYPE16)(0);

  CL_DTYPE8 a0, a1, a2, a3;
  CL_DTYPE4 b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, 0));
  CL_DTYPE4 b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, 1));

  CL_DTYPE4 bias_ih_0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias_ih, SAMPLER, (int2)(idx, 0));
  CL_DTYPE4 bias_hh_0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias_hh, SAMPLER, (int2)(idx, 0));

  __global const CL_DTYPE* A0 = a + idy * K;
  __global const CL_DTYPE* A1 = a + ((idy + 1) < M ? (idy + 1) : (M - 1)) * K;
  __global const CL_DTYPE* A2 = a + ((idy + 2) < M ? (idy + 2) : (M - 1)) * K;
  __global const CL_DTYPE* A3 = a + ((idy + 3) < M ? (idy + 3) : (M - 1)) * K;

  int p = 0;
  for (; p + 8 < K; p += 8) {
    a0 = vload8(0, A0 + p);
    a1 = vload8(0, A1 + p);
    a2 = vload8(0, A2 + p);
    a3 = vload8(0, A3 + p);

    if (idx == 0 && idy == 0 && p == 0) {
      printf("a0: %d %f %f %f %f %f %f %f %f\n",
             p,
             a0.s0,
             a0.s1,
             a0.s2,
             a0.s3,
             a0.s4,
             a0.s5,
             a0.s6,
             a0.s7);
      printf("b0: %d %f %f %f\n", p, b0.s0, b0.s1, b0.s2, b0.s3);
    }

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s0,
                             (CL_DTYPE4)a1.s0,
                             (CL_DTYPE4)a2.s0,
                             (CL_DTYPE4)a3.s0),
                (CL_DTYPE16)(b0.s0123, b0.s0123, b0.s0123, b0.s0123),
                c_v16);

    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 2));
    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s1,
                             (CL_DTYPE4)a1.s1,
                             (CL_DTYPE4)a2.s1,
                             (CL_DTYPE4)a3.s1),
                (CL_DTYPE16)(b1.s0123, b1.s0123, b1.s0123, b1.s0123),
                c_v16);

    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 3));
    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s2,
                             (CL_DTYPE4)a1.s2,
                             (CL_DTYPE4)a2.s2,
                             (CL_DTYPE4)a3.s2),
                (CL_DTYPE16)(b0.s0123, b0.s0123, b0.s0123, b0.s0123),
                c_v16);

    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 4));
    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s3,
                             (CL_DTYPE4)a1.s3,
                             (CL_DTYPE4)a2.s3,
                             (CL_DTYPE4)a3.s3),
                (CL_DTYPE16)(b1.s0123, b1.s0123, b1.s0123, b1.s0123),
                c_v16);
    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 5));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s4,
                             (CL_DTYPE4)a1.s4,
                             (CL_DTYPE4)a2.s4,
                             (CL_DTYPE4)a3.s4),
                (CL_DTYPE16)(b0.s0123, b0.s0123, b0.s0123, b0.s0123),
                c_v16);

    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 6));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s5,
                             (CL_DTYPE4)a1.s5,
                             (CL_DTYPE4)a2.s5,
                             (CL_DTYPE4)a3.s5),
                (CL_DTYPE16)(b1.s0123, b1.s0123, b1.s0123, b1.s0123),
                c_v16);
    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 7));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s6,
                             (CL_DTYPE4)a1.s6,
                             (CL_DTYPE4)a2.s6,
                             (CL_DTYPE4)a3.s6),
                (CL_DTYPE16)(b0.s0123, b0.s0123, b0.s0123, b0.s0123),
                c_v16);
    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 8));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s7,
                             (CL_DTYPE4)a1.s7,
                             (CL_DTYPE4)a2.s7,
                             (CL_DTYPE4)a3.s7),
                (CL_DTYPE16)(b1.s0123, b1.s0123, b1.s0123, b1.s0123),
                c_v16);
    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 9));
  }
  for (; p < K; p++) {
    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p));
    c_v16.s0123 += (CL_DTYPE4)(A0[p]) * b0;
    c_v16.s4567 += (CL_DTYPE4)(A1[p]) * b0;
    c_v16.s89ab += (CL_DTYPE4)(A2[p]) * b0;
    c_v16.scdef += (CL_DTYPE4)(A3[p]) * b0;
  }

  if (idy == 0 && idx == 0) {
    printf("rnn_gemm_4x4 c : %f %f %f %f\n",
           c_v16.s0,
           c_v16.s1,
           c_v16.s2,
           c_v16.s3);
  }
  if (idy == 0 && idx == 128) {
    printf("rnn_gemm_4x4 c 128: %f %f %f %f\n",
           c_v16.s0,
           c_v16.s1,
           c_v16.s2,
           c_v16.s3);
  }
  if (idy == 0 && idx == 256) {
    printf("rnn_gemm_4x4 c 256: %f %f %f %f\n",
           c_v16.s0,
           c_v16.s1,
           c_v16.s2,
           c_v16.s3);
  }
  if (idy == 0 && idx == 384) {
    printf("rnn_gemm_4x4 c 384: %f %f %f %f\n",
           c_v16.s0,
           c_v16.s1,
           c_v16.s2,
           c_v16.s3);
  }
  if (idy == 0 && idx == 512) {
    printf("rnn_gemm_4x4 c 512: %f %f %f %f\n",
           c_v16.s0,
           c_v16.s1,
           c_v16.s2,
           c_v16.s3);
  }

  c_v16 = c_v16 + (CL_DTYPE16)(bias_ih_0, bias_ih_0, bias_ih_0, bias_ih_0);
  c_v16 = c_v16 + (CL_DTYPE16)(bias_hh_0, bias_hh_0, bias_hh_0, bias_hh_0);

  if (((idx << 2) + 4) <= N) {
    if (idy < M) {
      vstore4(c_v16.s0123, 0, c + idy * N + (idx << 2));
    }
    if (idy + 1 < M) {
      vstore4(c_v16.s4567, 0, c + (idy + 1) * N + (idx << 2));
    }
    if (idy + 2 < M) {
      vstore4(c_v16.s89ab, 0, c + (idy + 2) * N + (idx << 2));
    }
    if (idy + 3 < M) {
      vstore4(c_v16.scdef, 0, c + (idy + 3) * N + (idx << 2));
    }
  } else {
    CL_DTYPE c_v0[4] = {c_v16.s0, c_v16.s1, c_v16.s2, c_v16.s3};
    CL_DTYPE c_v4[4] = {c_v16.s4, c_v16.s5, c_v16.s6, c_v16.s7};
    CL_DTYPE c_v8[4] = {c_v16.s8, c_v16.s9, c_v16.sa, c_v16.sb};
    CL_DTYPE c_vc[4] = {c_v16.sc, c_v16.sd, c_v16.se, c_v16.sf};
    for (int i = idx << 2; i < N; i++) {
      int c_index = i - (idx << 2);
      if (idy < M) {
        c[idy * N + i] = c_v0[c_index];
      }
      if (idy + 1 < M) {
        c[(idy + 1) * N + i] = c_v4[c_index];
      }
      if (idy + 2 < M) {
        c[(idy + 2) * N + i] = c_v8[c_index];
      }
      if (idy + 3 < M) {
        c[(idy + 3) * N + i] = c_vc[c_index];
      }
    }
  }

  if (idy == 0 && idx == 0) {
    printf("~rnn_gemm_4x4 c : %f %f %f %f\n",
           c_v16.s0,
           c_v16.s1,
           c_v16.s2,
           c_v16.s3);
    printf("bias_ih: %f %f %f %f\n",
           bias_ih_0.s0,
           bias_ih_0.s1,
           bias_ih_0.s2,
           bias_ih_0.s3);
    printf("bias_hh: %f %f %f %f\n",
           bias_hh_0.s0,
           bias_hh_0.s1,
           bias_hh_0.s2,
           bias_hh_0.s3);
  }
  if (idy == 0 && idx == 128) {
    printf("~rnn_gemm_4x4 c 128: %f %f %f %f\n",
           c_v16.s0,
           c_v16.s1,
           c_v16.s2,
           c_v16.s3);
    printf("bias_ih 128: %f %f %f %f\n",
           bias_ih_0.s0,
           bias_ih_0.s1,
           bias_ih_0.s2,
           bias_ih_0.s3);
    printf("bias_hh 128: %f %f %f %f\n",
           bias_hh_0.s0,
           bias_hh_0.s1,
           bias_hh_0.s2,
           bias_hh_0.s3);
  }
  if (idy == 0 && idx == 256) {
    printf("~rnn_gemm_4x4 c 256: %f %f %f %f\n",
           c_v16.s0,
           c_v16.s1,
           c_v16.s2,
           c_v16.s3);
    printf("bias_ih 256: %f %f %f %f\n",
           bias_ih_0.s0,
           bias_ih_0.s1,
           bias_ih_0.s2,
           bias_ih_0.s3);
    printf("bias_hh 256: %f %f %f %f\n",
           bias_hh_0.s0,
           bias_hh_0.s1,
           bias_hh_0.s2,
           bias_hh_0.s3);
  }
  if (idy == 0 && idx == 384) {
    printf("~rnn_gemm_4x4 c 384: %f %f %f %f\n",
           c_v16.s0,
           c_v16.s1,
           c_v16.s2,
           c_v16.s3);
    printf("bias_ih 384: %f %f %f %f\n",
           bias_ih_0.s0,
           bias_ih_0.s1,
           bias_ih_0.s2,
           bias_ih_0.s3);
    printf("bias_hh 384: %f %f %f %f\n",
           bias_hh_0.s0,
           bias_hh_0.s1,
           bias_hh_0.s2,
           bias_hh_0.s3);
  }
  if (idy == 0 && idx == 512) {
    printf("~rnn_gemm_4x4 c 512: %f %f %f %f\n",
           c_v16.s0,
           c_v16.s1,
           c_v16.s2,
           c_v16.s3);
    printf("bias_ih 512: %f %f %f %f\n",
           bias_ih_0.s0,
           bias_ih_0.s1,
           bias_ih_0.s2,
           bias_ih_0.s3);
    printf("bias_hh 512: %f %f %f %f\n",
           bias_hh_0.s0,
           bias_hh_0.s1,
           bias_hh_0.s2,
           bias_hh_0.s3);
  }
}

__kernel void rnn_lstm_gemm(__global const CL_DTYPE* a,
                            __read_only image2d_t b,
                            __global const CL_DTYPE* bias,  // 15 * 2048
                            __global CL_DTYPE* c,
                            const int M,
                            const int N,
                            const int K,
                            const int time_step_id,
                            const int frame_size) {
  const int idx = get_global_id(0);  // id: [0, M>>2) height of out == M

  if ((idx << 2) >= N) return;

  CL_DTYPE4 bias0 = bias ? vload4(0, bias + time_step_id * 2048 + (idx << 2))
                         : (CL_DTYPE4)0;  // 2048 fix
  CL_DTYPE4 c_v4 = (CL_DTYPE4)0;

  CL_DTYPE8 a0, a1, a2, a3;
  CL_DTYPE4 b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, 0));
  CL_DTYPE4 b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, 1));
  // CL_DTYPE4 b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(0, 0));
  // CL_DTYPE4 b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(0, 1));
  __global const CL_DTYPE* A0 =
      time_step_id == 0 ? a : a + (time_step_id - 1) * frame_size;

  // if (idx == 0){
  //   printf("bias0: %f %f %f %f\n", bias0.s0, bias0.s1, bias0.s2, bias0.s3);
  // }

  int p = 0;
  for (; p + 8 < K; p += 8) {
    a0 = vload8(0, A0 + p);
    c_v4 = mad((CL_DTYPE4)(a0.s0), b0, c_v4);

    // if (idx == 0 && p == 0){
    //   printf("--rnn_lstm_gemm time_step_id: %d\n", time_step_id);
    //   printf("rnn_lstm_gemm_kernel a %d : %f %f %f %f\n", p, a0.s0, a0.s1,
    //   a0.s2, a0.s3);
    //   printf("rnn_lstm_gemm_kernel b0: %d %d %f %f %f %f\n", p + 2, idx,
    //   b0.s0, b0.s1, b0.s2, b0.s3);
    // }
    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 2));

    c_v4 = mad((CL_DTYPE4)(a0.s1), b1, c_v4);
    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 3));

    c_v4 = mad((CL_DTYPE4)(a0.s2), b0, c_v4);
    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 4));

    c_v4 = mad((CL_DTYPE4)(a0.s3), b1, c_v4);
    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 5));

    c_v4 = mad((CL_DTYPE4)(a0.s4), b0, c_v4);
    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 6));

    c_v4 = mad((CL_DTYPE4)(a0.s5), b1, c_v4);
    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 7));

    c_v4 = mad((CL_DTYPE4)(a0.s6), b0, c_v4);
    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 8));

    c_v4 = mad((CL_DTYPE4)(a0.s7), b1, c_v4);
    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 9));
  }
  for (; p < K; p++) {
    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p));
    c_v4 += (CL_DTYPE4)(A0[p]) * b0;
  }

  // if (idx == 0){
  //   printf("rnn_lstm_gemm 0: %f %f %f %f\n", c_v4.s0, c_v4.s1, c_v4.s2,
  //   c_v4.s3);
  // }
  // if (idx == 128){
  //   printf("rnn_lstm_gemm 128 * 4: %f %f %f %f\n", c_v4.s0, c_v4.s1, c_v4.s2,
  //   c_v4.s3);
  // }
  // if (idx == 256){
  //   printf("rnn_lstm_gemm 256 * 4: %f %f %f %f\n", c_v4.s0, c_v4.s1, c_v4.s2,
  //   c_v4.s3);
  // }
  // if (idx == 384){
  //   printf("rnn_lstm_gemm 384 * 4: %f %f %f %f\n", c_v4.s0, c_v4.s1, c_v4.s2,
  //   c_v4.s3);
  // }

  c_v4 = c_v4 + bias0;

  if (((idx << 2) + 4) <= N) {
    vstore4(c_v4, 0, c + (idx << 2));
  } else {
    CL_DTYPE c_v0[4] = {c_v4.s0, c_v4.s1, c_v4.s2, c_v4.s3};
    for (int i = idx << 2; i < N; i++) {
      int c_index = i - (idx << 2);
      c[i] = c_v0[c_index];
    }
  }
  // if (idx == 0){
  //   printf("rnn_lstm_gemm 0 add : %f %f %f %f\n", c_v4.s0, c_v4.s1, c_v4.s2,
  //   c_v4.s3);
  //   printf("rnn_lstm_gemm 0 bias: %f %f %f %f\n", bias0.s0, bias0.s1,
  //   bias0.s2, bias0.s3);
  // }
  // if (idx == 128){
  //   printf("rnn_lstm_gemm 128 add : %f %f %f %f\n", c_v4.s0, c_v4.s1,
  //   c_v4.s2, c_v4.s3);
  //   printf("rnn_lstm_gemm 128 bias: %f %f %f %f\n", bias0.s0, bias0.s1,
  //   bias0.s2, bias0.s3);
  // }
  // if (idx == 256){
  //   printf("rnn_lstm_gemm 256 add : %f %f %f %f\n", c_v4.s0, c_v4.s1,
  //   c_v4.s2, c_v4.s3);
  //   printf("rnn_lstm_gemm 256 bias: %f %f %f %f\n", bias0.s0, bias0.s1,
  //   bias0.s2, bias0.s3);
  // }
  // if (idx == 384){
  //   printf("rnn_lstm_gemm 384 add : %f %f %f %f\n", c_v4.s0, c_v4.s1,
  //   c_v4.s2, c_v4.s3);
  //   printf("rnn_lstm_gemm 384 bias: %f %f %f %f\n", bias0.s0, bias0.s1,
  //   bias0.s2, bias0.s3);
  // }
}

__kernel void rnn_lstm_compute(__global CL_DTYPE* gate_value,  // 1, 2048
                               __global CL_DTYPE* state,
                               __global CL_DTYPE* state_act,
                               __global CL_DTYPE* prev_state,  // init_c
                               __global CL_DTYPE* output,      // 15 * 512
                               const int frame_size,
                               const float cell_clip,
                               const int time_step_id,
                               __global CL_DTYPE* out1,
                               __global CL_DTYPE* out2,
                               const int max_step) {
  const int idx = get_global_id(0);
  __global CL_DTYPE* value_ig = gate_value;
  __global CL_DTYPE* value_fg = value_ig + frame_size;
  __global CL_DTYPE* value_in = value_fg + frame_size;
  __global CL_DTYPE* value_og = value_in + frame_size;
  __global CL_DTYPE* output_begin = output + time_step_id * frame_size;

  if (idx == 0) {
    printf("value_ig 0: %f\n", value_ig[0]);
    printf("value_fg 0: %f\n", value_fg[0]);
    printf("value_in 0: %f\n", value_in[0]);
    printf("value_og 0: %f\n", value_og[0]);
    printf("frame_size: %d\n", frame_size);
    printf("prev_state: %f\n", prev_state[0]);
    printf("state: %f\n", state[0]);
  }

  float prev_state_tmp = convert_float(prev_state[idx]);

  float value_in_tmp = convert_float(value_in[idx]);
  value_in_tmp = (exp(value_in_tmp) - exp(-value_in_tmp)) /
                 (exp(value_in_tmp) + exp(-value_in_tmp));
  // if (idx == 0)printf("value_in_tmp: %f\n", value_in_tmp);

  // float value_ig_tmp = convert_float(value_ig[idx]) * prev_state_tmp;
  float value_ig_tmp = convert_float(value_ig[idx]) + prev_state_tmp * 0;  // 1
  // if (idx == 0)printf("value_ig_tmp: %f\n", value_ig_tmp);
  // value_ig_tmp = (exp(value_ig_tmp) - exp(-value_ig_tmp)) /
  //                 (exp(value_ig_tmp) + exp(-value_ig_tmp));
  value_ig_tmp =
      (float)(1.0f / (1.0f + pow(2.71828182f, -1.0f * (float)(value_ig_tmp))));
  // if (idx == 0)printf("value_ig_tmp act after: %f\n", value_ig_tmp);

  // float value_fg_tmp = convert_float(value_fg[idx]) * prev_state_tmp;
  float value_fg_tmp = convert_float(value_fg[idx]) + prev_state_tmp * 0;  // 2
  // if (idx == 0)printf("value_fg_tmp: %f\n", value_fg_tmp);
  // value_fg_tmp = (exp(value_fg_tmp) - exp(-value_fg_tmp)) /
  //                 (exp(value_fg_tmp) + exp(-value_fg_tmp));
  value_fg_tmp =
      (float)(1.0f / (1.0f + pow(2.71828182f, -1.0f * (float)(value_fg_tmp))));
  // if (idx == 0)printf("value_fg_tmp act after: %f\n", value_fg_tmp);

  float state_tmp = value_in_tmp * value_ig_tmp;  // 3
  // if (idx == 0)printf("state_tmp 1: %f\n", state_tmp);
  state_tmp = state_tmp + prev_state_tmp * value_fg_tmp;  // 4
  // if (idx == 0)printf("state_tmp 2: %f\n", state_tmp);
  state[idx] = state_tmp;
  if (cell_clip > 0.0) {
    if (state_tmp < -1.0 * cell_clip) {
      state_tmp = -1.0 * cell_clip;
    }
    if (state_tmp > cell_clip) {
      state_tmp = cell_clip;
    }
  }

  float value_og_tmp = convert_float(value_og[idx]) + 0 * state_tmp;  // 5
  // if (idx == 0)printf("value_og_tmp v: %f\n", value_og_tmp);
  // value_og_tmp = (exp(value_og_tmp) - exp(-value_og_tmp)) /
  //                (exp(value_og_tmp) + exp(-value_og_tmp));
  value_og_tmp =
      (float)(1.0f / (1.0f + pow(2.71828182f, -1.0f * (float)(value_og_tmp))));
  // if (idx == 0)printf("value_og_tmp a: %f\n", value_og_tmp);
  float state_act_tmp =
      (exp(state_tmp) - exp(-state_tmp)) / (exp(state_tmp) + exp(-state_tmp));

  // output_begin[idx] = value_og_tmp * state_act_tmp;                    //6
  output_begin[idx] =
      CONVERT_TYPE_TO(value_og_tmp * state_act_tmp, CL_DTYPE);  // 6
  if (idx == 0) {
    printf("value_in_tmp, value_ig_tmp: %f, %f\n", value_in_tmp, value_ig_tmp);
    printf("state_tmp, value_og[idx]: %f, %f\n", state_tmp, value_og[idx]);
    printf(
        "value_og_tmp, state_act_tmp: %f, %f\n", value_og_tmp, state_act_tmp);
    printf("rnn_lstm_compute output: %f\n", output_begin[idx]);
    printf("new state: %f\n", state[idx]);
  }
  if (idx <= 5) {
    printf("rnn_lstm_compute output~~ %d: %f\n", idx, output_begin[idx]);
    printf("time_step_id %d: max_step:%d\n", time_step_id, max_step);
  }
  if (time_step_id == (max_step - 1)) {
    out1[idx] = output_begin[idx];
    out2[idx] = state_tmp;
    if (idx <= 5) {
      printf("out1 %d: %f\n", idx, out1[idx]);
      printf("out2 %d: %f\n", idx, out2[idx]);
    }
  }
}
