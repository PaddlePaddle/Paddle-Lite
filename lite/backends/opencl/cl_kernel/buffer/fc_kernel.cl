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
__kernel void mat_transpose(__global const CL_DTYPE* src,
                            __global CL_DTYPE* dst,
                            const int src_height,
                            const int src_width) {
  const int col = get_global_id(0);  // [0, src_width)  columns of src
  const int row = get_global_id(1);  // [0, src_height) rows of src
  DST(col, row) = SRC(row, col);
}

// fc_gemm_naive: keep for check
// a: x_d
// b: filter_d
// c: output_d
__kernel void fc_gemm_naive(__global const CL_DTYPE* a,
                            __global const CL_DTYPE* b,
                            __global const CL_DTYPE* bias,
                            __global CL_DTYPE* c,
                            const int M,
                            const int N,
                            const int K) {
  const int row = get_global_id(0);  // [0, M) height of out == m
  const int col = get_global_id(1);  // [0, N) width of out == n

  if ((col >= N) || (row >= M)) {
    return;
  }

  CL_DTYPE a0, b0, c0 = (bias && col < N) ? bias[col] : 0;

  for (int p = 0; p < K; ++p) {
    a0 = *(a + row * K + p);
    b0 = *(b + p * N + col);
    c0 += a0 * b0;
  }

#ifdef RELU
  CL_DTYPE alpha;
  c[row * N + col] = activation(c0, alpha);
#else
  c[row * N + col] = c0;
#endif
}

// gemm_batch_naive: used for conv1x1, gemm of im2col_gemm
// a: filter_d
// b: x_d
// c: output_d
__kernel void gemm_batch_naive(__global const CL_DTYPE* a,
                               __global const CL_DTYPE* b,
                               __global const CL_DTYPE* bias,
                               __global CL_DTYPE* c,
                               const int M,
                               const int N,
                               const int K,
                               const int batch_size) {
  const int row = get_global_id(0);   // [0, M) height of out == m
  const int col = get_global_id(1);   // [0, N) width of out == n
  const int bidx = get_global_id(2);  // [0, batch_size)

  const __global CL_DTYPE* cur_b = b + K * N * bidx;
  __global CL_DTYPE* cur_c = c + M * N * bidx;

  if ((col >= N) || (row >= M) || (bidx >= batch_size)) {
    return;
  }

  CL_DTYPE a0, b0, c0 = (bias && col < N) ? bias[row] : 0;

  for (int p = 0; p < K; ++p) {
    a0 = *(a + row * K + p);
    b0 = *(cur_b + p * N + col);
    c0 += a0 * b0;
  }

  CL_DTYPE alpha;
  cur_c[row * N + col] = activation(c0, alpha);
}

// gemm_batch_8x4_buf_buf_N_N: used for conv1x1, gemm of im2col_gemm
// a: filter_d
// b: x_d
// c: output_d
#if 0  // TODO(ysh239): cause CL_OUT_OF_HOST_MEMORY on some devices(such as
       // snapdragon 855)
//#define PRINT_KERNEL
__kernel
void gemm_batch(__global const CL_DTYPE* Aptr,
                __global const CL_DTYPE* Bptr,
                __global const CL_DTYPE* bias,
                __global CL_DTYPE* Cptr,
                const int M, const int N, const int K, const int batch_size) {

    int row = get_global_id(0) << 3; // [0, M >> 3) height of out == m
    int col = get_global_id(1) << 2; // [0, N >> 2) width of out == n
    const int bidx = get_global_id(2); // [0, batch_size)

    // update B(input), C(output) with batch_size
    Aptr += mul24(row, K); // A += row * K
    Bptr += mad24(mul24(K, N), bidx, col); // B += K * N * bidx + col
    Cptr += mad24(mul24(M, N), bidx, mul24(row, N)); // C += M * N * bidx + row * N

    CL_DTYPE4 a8x4[8];
    CL_DTYPE4 b4x4[4] = {0.f, 0.f, 0.f, 0.f};
    CL_DTYPE4 c8x4[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    if (bias) {
        c8x4[0] = bias[row];
        c8x4[1] = bias[row + 1];
        c8x4[2] = bias[row + 2];
        c8x4[3] = bias[row + 3];
        c8x4[4] = bias[row + 4];
        c8x4[5] = bias[row + 5];
        c8x4[6] = bias[row + 6];
        c8x4[7] = bias[row + 7];
    }

    // main loop of K
    short pos = 0;
    for (; pos < K - 3; pos += 4) {
        b4x4[0] = vload4(0, Bptr + mul24(pos, N));
        b4x4[1] = vload4(0, Bptr + mul24(pos+1, N));
        b4x4[2] = vload4(0, Bptr + mul24(pos+2, N));
        b4x4[3] = vload4(0, Bptr + mul24(pos+3, N));

        // main compute of main loop K: pos + 3 < K
#pragma unroll(8)
        for (int i = 0; i < 8 && i < M; ++i) { // M direction
            a8x4[i] = vload4(0, Aptr + mad24(i, K, pos));

            c8x4[i] += a8x4[i].x * b4x4[0];
            c8x4[i] += a8x4[i].y * b4x4[1];
            c8x4[i] += a8x4[i].z * b4x4[2];
            c8x4[i] += a8x4[i].w * b4x4[3];
        }
    }

    // compute left K
    if (pos < K) {
        b4x4[0] = 0.0f;
        b4x4[1] = 0.0f;
        b4x4[2] = 0.0f;
        // b4x4[3] = 0.0f; // impossible used
        switch (K - pos) {
            case 3:
                b4x4[2] = vload4(0, Bptr + mul24(pos+2, N));

            case 2:
                b4x4[1] = vload4(0, Bptr + mul24(pos+1, N));

            case 1:
                b4x4[0] = vload4(0, Bptr + mul24(pos, N));
        }

#pragma unroll(8)
        for (int i = 0; i < 8; i++) {
            a8x4[i] = vload4(0, Aptr + mad24(i, K, pos));

            c8x4[i] += a8x4[i].x * b4x4[0] +
                       a8x4[i].y * b4x4[1] +
                       a8x4[i].z * b4x4[2];
        }
    }

#ifdef RELU
#pragma unroll(8)
    for (int i = 0; i < 8; ++i) {
        c8x4[i] = fmax(c8x4[i], (CL_DTYPE4)0.f);
    }
#endif

    // store c
    if (row + 7 < M && col + 3 < N) {
#pragma unroll(8)
        for (int i = 0; i < 8; i++) { // M direction
            vstore4(c8x4[i], 0, Cptr + mad24(i, N, col));
        }
    } else {
        for (int i = 0; i < 8 && i + row < M; ++i) { // M direction
            if (col + 3 < N) {
                vstore4(c8x4[i], 0, Cptr + mad24(i, N, col));
            } else {
                switch (N - col) {
                    case 3:
                        *(Cptr + mad24(i, N, col + 2))  = c8x4[i].s2;
                    case 2:
                        *(Cptr + mad24(i, N, col + 1))  = c8x4[i].s1;
                    case 1:
                        *(Cptr + mad24(i, N, col))  = c8x4[i].s0;
               }
            }
        }
    }
}
#endif

// fc_gemv_naive: keep for check
// used for fc with M = 1
// a: param.input  {M, K}
// b: param.w      {K, N}
// c: param.output {M, N}
__kernel void fc_gemv_naive(__global const CL_DTYPE* a,
                            __global const CL_DTYPE* b,
                            __global const CL_DTYPE* bias,
                            __global CL_DTYPE* c,
                            const int M,
                            const int N,
                            const int K) {
  const int col = get_global_id(0);  // gws[0]: [0, N) width of B == N

  if (col >= N) {
    return;
  }
  CL_DTYPE c0 = bias ? bias[col] : 0;
  for (int p = 0; p < K; ++p) {
    CL_DTYPE a0 = *(a + p);
    CL_DTYPE b0 = *(b + p * N + col);
    c0 += a0 * b0;
  }

#ifdef RELU
  CL_DTYPE alpha;
  c[col] = activation(c0, alpha);
#else
  c[col] = c0;
#endif
}

// fc_gemv_1x4: for fc with M = 1
// a: param.input  {M, K}
// b: param.w      {K, N}
// c: param.output {M, N}
__kernel void fc_gemv_1x4(__global const float* a,
                          __global const float* b,
                          __global const float* bias,
                          __global float* c,
                          const int M,
                          const int N,
                          const int K,
                          __global const float* alpha) {
  const int col = get_global_id(0)
                  << 2;  // gws[0]: [0, N >> 2) height of B == N

  if (col + 3 < N) {
    half4 c0 = 0.0f;
    if (bias) {
      c0.x = bias[col];
      c0.y = bias[col + 1];
      c0.z = bias[col + 2];
      c0.w = bias[col + 3];
    }

    // main loop of K
    int p = 0;
    for (; p < K - 3; p += 4) {
      half4 a0 = convert_half4(vload4(0, a + p));

      half4 b0 = convert_half4(vload4(0, b + p * N + col));
      half4 b1 = convert_half4(vload4(0, b + (p + 1) * N + col));
      half4 b2 = convert_half4(vload4(0, b + (p + 2) * N + col));
      half4 b3 = convert_half4(vload4(0, b + (p + 3) * N + col));

      c0 += a0.x * b0;
      c0 += a0.y * b1;
      c0 += a0.z * b2;
      c0 += a0.w * b3;
    }

    // compute left K
    half4 b2 = 0.0f, b1 = 0.0f, b0 = 0.0f, a0 = 0.0f;
    switch (K - p) {
      case 3: {
        b2 = convert_half4(vload4(0, b + (p + 2) * N + col));
        a0.z = a[p + 2];
      }
      case 2: {
        b1 = convert_half4(vload4(0, b + (p + 1) * N + col));
        a0.y = a[p + 1];
      }
      case 1: {
        b0 = convert_half4(vload4(0, b + (p)*N + col));
        a0.x = a[p];
      }
    }
    c0 += a0.x * b0;
    c0 += a0.y * b1;
    c0 += a0.z * b2;

    half4 alpha0 = 0.0f;
#ifdef PRELU_MORE
    alpha0.x = alpha[col];
    alpha0.y = alpha[col + 1];
    alpha0.z = alpha[col + 2];
    alpha0.w = alpha[col + 3];
#else
    alpha0.x = alpha[0];
    alpha0.y = alpha[0];
    alpha0.z = alpha[0];
    alpha0.w = alpha[0];
#endif
    if (col % 4 == 0) {
      float4 act_res = convert_float4(activation_type4(c0, alpha0));
      vstore4(act_res, 0, c + col);
    } else {
      switch (col % 4) {
        case 3:
          c[col + 2] = activation(c0.z, alpha0.z);
        case 2:
          c[col + 1] = activation(c0.y, alpha0.y);
        case 1:
          c[col] = activation(c0.x, alpha0.x);
      }
    }
  } else {
    const int left_col = N - col;
    for (int col_offset = 0; col_offset < left_col; ++col_offset) {
      half c0 = bias ? bias[col] : 0;
      for (int p = 0; p < K; ++p) {
        half b0 = *(b + p * N + col + col_offset);
        half a0 = *(a + p);
        c0 += a0 * b0;
      }
      half alpha0 = 0.0f;
#ifdef PRELU_MORE
      alpha0 = alpha[col];
#else
      alpha0 = alpha[0];
#endif
      c[col + col_offset] = activation(c0, alpha0);
    }
  }
}

// fc_gemm_4x4: for fc with M = 1
// a: param.input  {M, K}
// b: param.w      {K, N}
// c: param.output {M, N}
__kernel void fc_gemm_4x4(__global const float* a,
                          __global const float* b,
                          __global const float* bias,
                          __global float* c,
                          const int M,
                          const int N,
                          const int K,
                          __global const float* alpha) {
  const int row = get_global_id(0) << 2;  // id: [0, M>>2) height of out == M
  const int col = get_global_id(1) << 2;  // id: [0, N>>2) width of out == N

  if (row + 3 < M && col + 3 < N) {
    CL_COMPUTE_DTYPE bias0 = bias ? bias[col] : 0,
                     bias1 = bias ? bias[col + 1] : 0,
                     bias2 = bias ? bias[col + 2] : 0,
                     bias3 = bias ? bias[col + 3] : 0;

    CL_COMPUTE_DTYPE c00 = bias0, c01 = bias1, c02 = bias2, c03 = bias3,
                     c10 = bias0, c11 = bias1, c12 = bias2, c13 = bias3,
                     c20 = bias0, c21 = bias1, c22 = bias2, c23 = bias3,
                     c30 = bias0, c31 = bias1, c32 = bias2, c33 = bias3;

    for (int p = 0; p < K; ++p) {
      CL_COMPUTE_DTYPE
      a00 = *(a + row * K + p), a10 = *(a + (row + 1) * K + p),
      a20 = *(a + (row + 2) * K + p), a30 = *(a + (row + 3) * K + p),

      b00 = *(b + p * N + col), b01 = *(b + p * N + (col + 1)),
      b02 = *(b + p * N + (col + 2)), b03 = *(b + p * N + (col + 3));

      c00 += a00 * b00;
      c01 += a00 * b01;
      c02 += a00 * b02;
      c03 += a00 * b03;
      c10 += a10 * b00;
      c11 += a10 * b01;
      c12 += a10 * b02;
      c13 += a10 * b03;
      c20 += a20 * b00;
      c21 += a20 * b01;
      c22 += a20 * b02;
      c23 += a20 * b03;
      c30 += a30 * b00;
      c31 += a30 * b01;
      c32 += a30 * b02;
      c33 += a30 * b03;
    }
    half alpha0 = 0.0f;
    half alpha1 = 0.0f;
    half alpha2 = 0.0f;
    half alpha3 = 0.0f;
#ifdef PRELU_MORE
    alpha0 = alpha[col];
    alpha1 = alpha[col + 1];
    alpha2 = alpha[col + 2];
    alpha3 = alpha[col + 3];
#else
    alpha0 = alpha[0];
    alpha1 = alpha[0];
    alpha2 = alpha[0];
    alpha3 = alpha[0];
#endif
    c[row * N + col] = activation(c00, alpha0);
    c[row * N + (col + 1)] = activation(c01, alpha1);
    c[row * N + (col + 2)] = activation(c02, alpha2);
    c[row * N + (col + 3)] = activation(c03, alpha3);
    c[(row + 1) * N + col] = activation(c10, alpha0);
    c[(row + 1) * N + (col + 1)] = activation(c11, alpha1);
    c[(row + 1) * N + (col + 2)] = activation(c12, alpha2);
    c[(row + 1) * N + (col + 3)] = activation(c13, alpha3);
    c[(row + 2) * N + col] = activation(c20, alpha0);
    c[(row + 2) * N + (col + 1)] = activation(c21, alpha1);
    c[(row + 2) * N + (col + 2)] = activation(c22, alpha2);
    c[(row + 2) * N + (col + 3)] = activation(c23, alpha3);
    c[(row + 3) * N + col] = activation(c30, alpha0);
    c[(row + 3) * N + (col + 1)] = activation(c31, alpha1);
    c[(row + 3) * N + (col + 2)] = activation(c32, alpha2);
    c[(row + 3) * N + (col + 3)] = activation(c33, alpha3);
  } else {
    for (int cidx = col; cidx < N; ++cidx) {
      for (int ridx = row; ridx < M; ++ridx) {
        CL_COMPUTE_DTYPE a0 = 0;
        CL_COMPUTE_DTYPE b0 = 0;
        CL_COMPUTE_DTYPE c0 = bias ? bias[cidx] : 0;
        for (int p = 0; p < K; ++p) {
          a0 = *(a + ridx * K + p);
          b0 = *(b + p * N + cidx), c0 += a0 * b0;
        }
        half alpha0 = 0.0f;
#ifdef PRELU_MORE
        alpha0 = alpha[cidx];
#else
        alpha0 = alpha[0];
#endif
        c[ridx * N + cidx] = activation(c0, alpha0);
      }
    }
  }
}
