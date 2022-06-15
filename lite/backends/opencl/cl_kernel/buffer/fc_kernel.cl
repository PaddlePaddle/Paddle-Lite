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

  //   if (col + 3 < N) {
  //     half4 c0 = 0.0f;
  //     if (bias) {
  //       c0.x = bias[col];
  //       c0.y = bias[col + 1];
  //       c0.z = bias[col + 2];
  //       c0.w = bias[col + 3];
  //     }

  //     // main loop of K
  //     int p = 0;
  //     for (; p < K - 3; p += 4) {
  //       half4 a0 = convert_half4(vload4(0, a + p));

  //       half4 b0 = convert_half4(vload4(0, b + p * N + col));
  //       half4 b1 = convert_half4(vload4(0, b + (p + 1) * N + col));
  //       half4 b2 = convert_half4(vload4(0, b + (p + 2) * N + col));
  //       half4 b3 = convert_half4(vload4(0, b + (p + 3) * N + col));

  //       c0 += a0.x * b0;
  //       c0 += a0.y * b1;
  //       c0 += a0.z * b2;
  //       c0 += a0.w * b3;
  //     }

  //     // compute left K
  //     half4 b2 = 0.0f, b1 = 0.0f, b0 = 0.0f, a0 = 0.0f;
  //     switch (K - p) {
  //       case 3: {
  //         b2 = convert_half4(vload4(0, b + (p + 2) * N + col));
  //         a0.z = a[p + 2];
  //       }
  //       case 2: {
  //         b1 = convert_half4(vload4(0, b + (p + 1) * N + col));
  //         a0.y = a[p + 1];
  //       }
  //       case 1: {
  //         b0 = convert_half4(vload4(0, b + (p)*N + col));
  //         a0.x = a[p];
  //       }
  //     }
  //     c0 += a0.x * b0;
  //     c0 += a0.y * b1;
  //     c0 += a0.z * b2;

  //     half4 alpha0 = 0.0f;
  // #ifdef PRELU_MORE
  //     alpha0.x = alpha[col];
  //     alpha0.y = alpha[col + 1];
  //     alpha0.z = alpha[col + 2];
  //     alpha0.w = alpha[col + 3];
  // #else
  //     alpha0.x = alpha[0];
  //     alpha0.y = alpha[0];
  //     alpha0.z = alpha[0];
  //     alpha0.w = alpha[0];
  // #endif
  //     if (col % 4 == 0) {
  //       float4 act_res = convert_float4(activation_type4(c0, alpha0));
  //       vstore4(act_res, 0, c + col);
  //     } else {
  //       switch (col % 4) {
  //         case 3:
  //           c[col + 2] = activation(c0.z, alpha0.z);
  //         case 2:
  //           c[col + 1] = activation(c0.y, alpha0.y);
  //         case 1:
  //           c[col] = activation(c0.x, alpha0.x);
  //       }
  //     }
  //   } else {
  //     const int left_col = N - col;
  //     for (int col_offset = 0; col_offset < left_col; ++col_offset) {
  //       half c0 = bias ? bias[col] : 0;
  //       for (int p = 0; p < K; ++p) {
  //         half b0 = *(b + p * N + col + col_offset);
  //         half a0 = *(a + p);
  //         c0 += a0 * b0;
  //       }
  //       half alpha0 = 0.0f;
  // #ifdef PRELU_MORE
  //       alpha0 = alpha[col];
  // #else
  //       alpha0 = alpha[0];
  // #endif
  //       c[col + col_offset] = activation(c0, alpha0);
  //     }
  //   }
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

__kernel void adreno_gemm_4x8(__global const CL_DTYPE* a,
                              __read_only image2d_t b,
                              __global const CL_DTYPE* bias,
                              __global CL_DTYPE* c,
                              const int M,
                              const int N,
                              const int K,
                              __global const CL_DTYPE* alpha) {
  const int idy = get_global_id(0) << 2;  // id: [0, M>>2) height of out == M
  const int idx = get_global_id(1);       // id: [0, N>>2) width of out == N
  // int2 tid = (int2)(get_local_id(0), get_local_id(1));
  // if (idx  == 2){
  //   printf("m: %d, n: %d, k: %d ~~~~~~~~\n", M, N, K);
  // }
  if ((idx << 2) >= N || idy >= M) return;

  // if (idx == 1 && idy == 1){
  //   printf("=======idx1\n");
  // }
  // CL_COMPUTE_DTYPE bias0 = bias ? bias[col] : 0,
  //                  bias1 = bias ? bias[col + 1] : 0,
  //                  bias2 = bias ? bias[col + 2] : 0,
  //                  bias3 = bias ? bias[col + 3] : 0;

  // CL_COMPUTE_DTYPE c00 = bias0, c01 = bias1, c02 = bias2, c03 = bias3,
  //                  c10 = bias0, c11 = bias1, c12 = bias2, c13 = bias3,
  //                  c20 = bias0, c21 = bias1, c22 = bias2, c23 = bias3,
  //                  c30 = bias0, c31 = bias1, c32 = bias2, c33 = bias3;

  CL_DTYPE4 bias0 = bias ? vload4(idx, bias) : (CL_DTYPE4)0;

  CL_DTYPE4 c0 = bias0, c1 = bias0, c2 = bias0, c3 = bias0;
  CL_DTYPE16 c_v16 =
      (CL_DTYPE16)(bias0.s0123, bias0.s0123, bias0.s0123, bias0.s0123);
  // CL_DTYPE4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  // CL_DTYPE4 a0, a1, a2, a3;
  CL_DTYPE8 a0, a1, a2, a3;
  // CL_DTYPE4 b0, b1, b2, b3;
  CL_DTYPE4 b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, 0));
  CL_DTYPE4 b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, 1));
  __global const CL_DTYPE* A0 = a + idy * K;
  __global const CL_DTYPE* A1 = A0 + K;
  __global const CL_DTYPE* A2 = A1 + K;
  __global const CL_DTYPE* A3 = A2 + K;
  for (int p = 0; p < K; p += 8) {
    // a0 = vload4(0, A0 + p);
    // a1 = vload4(0, A1 + p);
    // a2 = vload4(0, A2 + p);
    // a3 = vload4(0, A3 + p);

    a0 = vload8(0, A0 + p);
    a1 = vload8(0, A1 + p);
    a2 = vload8(0, A2 + p);
    a3 = vload8(0, A3 + p);

    // b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p));
    // b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 1));
    // b2 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 2));
    // b3 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 3));

    // if (idx == 0 && idy == 0){
    //   half4 k0 = a[0];
    //   half4 k1 = a[1];
    //   half4 k2 = a[2];
    //   half4 k3 = a[3];
    //   half4 k4 = a[4];
    //   printf("%d k0: %f %f %f %f\n", p, k0.s0, k0.s1, k0.s2, k0.s3);
    //   printf("%d k1: %f %f %f %f\n", p, k1.s0, k1.s1, k1.s2, k1.s3);
    //   printf("%d k2: %f %f %f %f\n", p, k2.s0, k2.s1, k2.s2, k2.s3);
    //   printf("%d k3: %f %f %f %f\n", p, k3.s0, k3.s1, k3.s2, k3.s3);
    //   printf("%d k3: %f %f %f %f\n", p, k4.s0, k4.s1, k4.s2, k4.s3);
    //   printf("(idy + 1) * K + p: %d\n", (idy + 1) * K + p);
    //   printf("%d a0: %f %f %f %f\n", p, a0.s0, a0.s1, a0.s2, a0.s3);
    //   printf("%d a1: %f %f %f %f\n", p, a1.s0, a1.s1, a1.s2, a1.s3);
    //   printf("%d a2: %f %f %f %f\n", p, a2.s0, a2.s1, a2.s2, a2.s3);
    //   printf("%d a3: %f %f %f %f\n", p, a3.s0, a3.s1, a3.s2, a3.s3);
    //   printf("%d b0: %f %f %f %f\n", p, b0.s0, b0.s1, b0.s2, b0.s3);
    //   printf("%d b1: %f %f %f %f\n", p, b1.s0, b1.s1, b1.s2, b1.s3);
    //   printf("%d b2: %f %f %f %f\n", p, b2.s0, b2.s1, b2.s2, b2.s3);
    //   printf("%d b3: %f %f %f %f\n", p, b3.s0, b3.s1, b3.s2, b3.s3);
    // }

    // c0 += a0.s0 * b0;
    // c1 += a1.s0 * b0;
    // c2 += a2.s0 * b0;
    // c3 += a3.s0 * b0;

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

    // c0 += a0.s1 * b1;
    // c1 += a1.s1 * b1;
    // c2 += a2.s1 * b1;
    // c3 += a3.s1 * b1;

    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 3));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s2,
                             (CL_DTYPE4)a1.s2,
                             (CL_DTYPE4)a2.s2,
                             (CL_DTYPE4)a3.s2),
                (CL_DTYPE16)(b0.s0123, b0.s0123, b0.s0123, b0.s0123),
                c_v16);

    // c0 += a0.s2 * b0;
    // c1 += a1.s2 * b0;
    // c2 += a2.s2 * b0;
    // c3 += a3.s2 * b0;

    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 4));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s3,
                             (CL_DTYPE4)a1.s3,
                             (CL_DTYPE4)a2.s3,
                             (CL_DTYPE4)a3.s3),
                (CL_DTYPE16)(b1.s0123, b1.s0123, b1.s0123, b1.s0123),
                c_v16);

    // c0 += a0.s3 * b1;
    // c1 += a1.s3 * b1;
    // c2 += a2.s3 * b1;
    // c3 += a3.s3 * b1;

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

    // if (idy == 0 && idx == 2){
    //   printf("~~~~~~~~%f %f %f %f\n", c0.x, c0.y, c0.z, c0.w);
    //   printf("a0~~~~~~~~%f %f %f %f\n", a0.x, a0.y, a0.z, a0.w);
    //   printf("b0~~~~~~~~%f %f %f %f\n", b0.x, b0.y, b0.z, b0.w);

    // }
    // if (idx == 0 && idy == 0){
    //   printf("%d c0: %f %f %f %f\n", p, c0.s0, c0.s1, c0.s2, c0.s3);
    //   printf("%d c1: %f %f %f %f\n", p, c1.s0, c1.s1, c1.s2, c1.s3);
    //   printf("%d c2: %f %f %f %f\n", p, c2.s0, c2.s1, c2.s2, c2.s3);
    //   printf("%d c3: %f %f %f %f\n", p, c3.s0, c3.s1, c3.s2, c3.s3);
    // }
  }
  // for (p; p < K; p++){
  //   b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p));
  //   c0 += a[idy * K + p] * b0;
  //   c1 += a[(idy + 1) * K + p] * b0;
  //   c2 += a[(idy + 2) * K + p] * b0;
  //   c3 += a[(idy + 3) * K + p] * b0;
  // }
  vstore4(c_v16.s0123, 0, c + idy * N + (idx << 2));
  vstore4(c_v16.s4567, 0, c + (idy + 1) * N + (idx << 2));
  vstore4(c_v16.s89ab, 0, c + (idy + 2) * N + (idx << 2));
  vstore4(c_v16.scdef, 0, c + (idy + 3) * N + (idx << 2));

  // if (idx == 0 && idy == 0){
  //   printf("c0: %f %f %f %f\n", c[0], c[0], c[0], c[0]);
  //   printf("c1: %f %f %f %f\n", c[1], c[1], c[1], c[1]);
  //   printf("c2: %f %f %f %f\n", c[2], c[2], c[2], c[2]);
  //   printf("c3: %f %f %f %f\n", c[3], c[3], c[3], c[3]);
  // }

  // if (idy == 0 && idx == 2){
  //   printf("~~~~~~~~%f %f %f %f\n", c0.x, c0.y, c0.z, c0.w);
  // }
  // printf("~~~%d %d %d %d %d %d %d\n", idx << 2, (idy + 1) * N, ((idy + 1) * N
  // + idx << 2), (idy * N + idx << 2) >> 2, ((idy + 1) * N + idx << 2) >> 2,
  // ((idy + 2) * N + idx << 2) >> 2, ((idy + 3) * N + idx << 2) >> 2);

  //     for (int cidx = col; cidx < N; ++cidx) {
  //       for (int ridx = row; ridx < M; ++ridx) {
  //         CL_COMPUTE_DTYPE a0 = 0;
  //         CL_COMPUTE_DTYPE b0 = 0;
  //         CL_COMPUTE_DTYPE c0 = bias ? bias[cidx] : 0;
  //         for (int p = 0; p < K; ++p) {
  //           a0 = *(a + ridx * K + p);
  //           b0 = *(b + p * N + cidx), c0 += a0 * b0;
  //         }
  //         half alpha0 = 0.0f;
  // #ifdef PRELU_MORE
  //         alpha0 = alpha[cidx];
  // #else
  //         alpha0 = alpha[0];
  // #endif
  //         c[ridx * N + cidx] = activation(c0, alpha0);
  //       }
  //     }
}

__kernel void adreno_gemv_1x4(__global const CL_DTYPE* a,
                              __read_only image2d_t b,
                              __global const CL_DTYPE* bias,
                              __global CL_DTYPE* c,
                              const int M,
                              const int N,
                              const int K,
                              __global const CL_DTYPE* alpha) {
  const int idx = get_global_id(0);  // id: [0, M>>2) height of out == M

  if ((idx << 2) >= N) return;

  CL_DTYPE4 bias0 = bias ? vload4(idx, bias) : (CL_DTYPE4)0;
  CL_DTYPE4 c_v4 = bias0;
  // CL_DTYPE16 c_v16 = (CL_DTYPE16)(0);

  // if(idx == 1) printf("--------0\n");

  CL_DTYPE8 a0, a1, a2, a3;
  CL_DTYPE4 b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, 0));
  CL_DTYPE4 b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, 1));
  __global const CL_DTYPE* A0 = a;

  int p = 0;
  for (; p + 8 < K; p += 8) {
    a0 = vload8(0, A0 + p);

    c_v4 = mad((CL_DTYPE4)(a0.s0), b0, c_v4);
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

  // if (idx == 2){
  //   printf("~~~~%f %f %f %f %f %f %f %f\n", c_v16.s0, c_v16.s1, c_v16.s2,
  //   c_v16.s3, c_v16.s4, c_v16.s5, c_v16.s6, c_v16.s7);
  //   printf("~~~~%f %f %f %f %f %f %f %f\n", c_v16.s8, c_v16.s9, c_v16.sa,
  //   c_v16.sb, c_v16.sc, c_v16.sd, c_v16.se, c_v16.sf);
  // }

  if (((idx << 2) + 4) <= N) {
    // if (idx == 0) printf("--------0\n");
    // if (idx == 1) printf("--------1\n");
    // if (idx == 2) printf("--------2\n");
    vstore4(c_v4, 0, c + (idx << 2));
  } else {
    // if(idx == 1) printf("--------1\n");
    CL_DTYPE c_v0[4] = {c_v4.s0, c_v4.s1, c_v4.s2, c_v4.s3};
    for (int i = idx << 2; i < N; i++) {
      int c_index = i - (idx << 2);
      c[i] = c_v0[c_index];
    }
  }
}

__kernel void adreno_gemv_trans_1x4(__global const CL_DTYPE* a,
                                    __read_only image2d_t b,
                                    __global const CL_DTYPE* bias,
                                    __global CL_DTYPE* c,
                                    const int M,
                                    const int N,
                                    const int K,
                                    __global const CL_DTYPE* alpha) {
  const int idx = get_global_id(0);  // id: [0, M>>2) height of out == M

  if ((idx << 2) >= N) return;

  CL_DTYPE4 bias0 = bias ? vload4(idx, bias) : (CL_DTYPE4)0;
  CL_DTYPE4 c0 = bias0;
  // CL_DTYPE16 c_v16 = (CL_DTYPE16)(0);
  // if (idx == 1){ printf("--243-\n");}
  CL_DTYPE4 a0, a1, a2, a3;
  int idx_4 = idx << 2;
  // CL_DTYPE4 b0, b1, b2, b3;
  __global const CL_DTYPE* A0 = a;
  CL_DTYPE4 b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(0, idx_4));
  CL_DTYPE4 b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(0, idx_4 + 1));
  CL_DTYPE4 b2 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(0, idx_4 + 2));
  CL_DTYPE4 b3 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(0, idx_4 + 3));
  int p = 0;
  for (; p + 4 <= K; p += 4) {
    a0 = vload4(0, A0 + p);

    // b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(p >> 2, (idx <<
    // 2)));
    // b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(p >> 2, (idx << 2) +
    // 1));
    // b2 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(p >> 2, (idx << 2) +
    // 2));
    // b3 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(p >> 2, (idx << 2) +
    // 3));

    c0.s0 += dot(a0, b0);
    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)((p >> 2) + 1, idx_4));
    c0.s1 += dot(a0, b1);
    b1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, b, SAMPLER, (int2)((p >> 2) + 1, idx_4 + 1));
    c0.s2 += dot(a0, b2);
    b2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, b, SAMPLER, (int2)((p >> 2) + 1, idx_4 + 2));
    c0.s3 += dot(a0, b3);
    b3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, b, SAMPLER, (int2)((p >> 2) + 1, idx_4 + 3));

    // if (idx == 0){
    //   printf("a: %f %f %f %f\n", a0.s0, a0.s1, a0.s2, a0.s3);
    //   printf("b: %f %f %f %f\n", b0.s0, b0.s1, b0.s2, b0.s3);
    //   printf("c: %f %f %f %f\n", c0.s0, c0.s1, c0.s2, c0.s3);
    // }
  }
  // for (; p < K; p++) {
  //   b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p));
  //   c_v4 += (CL_DTYPE4)(A0[p]) * b0;
  // }

  if (((idx << 2) + 4) <= N) {
    vstore4(c0, 0, c + (idx << 2));
  } else {
    // if(idx == 1) printf("--------1\n");
    // CL_DTYPE c_v0[4] = {c_v4.s0, c_v4.s1, c_v4.s2, c_v4.s3};
    // for (int i = idx << 2; i < N; i++){
    //   int c_index = i - (idx << 2);
    //   c[i] = c_v0[c_index];
    // }
  }
}

__kernel void adreno_gemm_4x4(__global const CL_DTYPE* a,
                              __read_only image2d_t b,
                              __global const CL_DTYPE* bias,
                              __global CL_DTYPE* c,
                              const int M,
                              const int N,
                              const int K,
                              __global const CL_DTYPE* alpha) {
  const int idy = get_global_id(0) << 2;  // id: [0, M>>2) height of out == M
  const int idx = get_global_id(1);       // id: [0, N>>2) width of out == N

  if ((idx << 2) >= N || idy >= M) return;

  CL_DTYPE4 bias0 = bias ? vload4(idx, bias) : (CL_DTYPE4)0;
  CL_DTYPE16 c_v16 =
      (CL_DTYPE16)(bias0.s0123, bias0.s0123, bias0.s0123, bias0.s0123);
  // CL_DTYPE16 c_v16 = (CL_DTYPE16)(0);

  CL_DTYPE8 a0, a1, a2, a3;
  CL_DTYPE4 b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, 0));
  CL_DTYPE4 b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, 1));
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
    // if (idx == 0 && idy == 0){
    //   printf("a: %d %f\n", p, A0[p]);
    //   printf("b: %f %f %f %f\n", b0.s0, b0.s1, b0.s2, b0.s3);
    //   printf("c: %f %f %f %f\n", c_v16.s0, c_v16.s1, c_v16.s2, c_v16.s3);
    // }
  }

  // if (idx == 2){
  //   printf("~~~~%f %f %f %f %f %f %f %f\n", c_v16.s0, c_v16.s1, c_v16.s2,
  //   c_v16.s3, c_v16.s4, c_v16.s5, c_v16.s6, c_v16.s7);
  //   printf("~~~~%f %f %f %f %f %f %f %f\n", c_v16.s8, c_v16.s9, c_v16.sa,
  //   c_v16.sb, c_v16.sc, c_v16.sd, c_v16.se, c_v16.sf);
  // }

  if (((idx << 2) + 4) <= N) {
    // if (idx == 0) printf("--------0\n");
    // if (idx == 1) printf("--------1\n");
    // if (idx == 2) printf("--------2\n");
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
        // if (idx == 0){ printf("0===%f\n", c_v0[c_index]); }
        // if (idx == 1){ printf("1===%f\n", c_v0[c_index]); }
        // if (idx == 2){ printf("2===%f\n", c_v0[c_index]); }
        c[idy * N + i] = c_v0[c_index];
      }
      if (idy + 1 < M) {
        // if (idx == 2){ printf("1===%f\n", c_v4[c_index]); }
        c[(idy + 1) * N + i] = c_v4[c_index];
      }
      if (idy + 2 < M) {
        // if (idx == 2){ printf("2===%f\n", c_v8[c_index]); }
        c[(idy + 2) * N + i] = c_v8[c_index];
      }
      if (idy + 3 < M) {
        // if (idx == 2){ printf("3===%f\n", c_vc[c_index]); }
        c[(idy + 3) * N + i] = c_vc[c_index];
      }
    }
  }
}

__kernel void adreno_gemm_trans_4x4(__global const CL_DTYPE* a,
                                    __read_only image2d_t b,
                                    __global const CL_DTYPE* bias,
                                    __global CL_DTYPE* c,
                                    const int M,
                                    const int N,
                                    const int K,
                                    __global const CL_DTYPE* alpha) {
  const int idy = get_global_id(0) << 2;  // id: [0, M>>2) height of out == M
  const int idx = get_global_id(1);       // id: [0, N>>2) width of out == N

  if ((idx << 2) >= N || idy >= M) return;

  CL_DTYPE4 bias0 = bias ? vload4(idx, bias) : (CL_DTYPE4)0;

  CL_DTYPE4 a0, a1, a2, a3;
  CL_DTYPE4 c0, c1, c2, c3;
  CL_DTYPE4 b0, b1, b2, b3;
  c0 = bias0;
  c1 = bias0;
  c2 = bias0;
  c3 = bias0;
  __global const CL_DTYPE* A0 = a + idy * K;
  __global const CL_DTYPE* A1 = a + ((idy + 1) < M ? (idy + 1) : (M - 1)) * K;
  __global const CL_DTYPE* A2 = a + ((idy + 2) < M ? (idy + 2) : (M - 1)) * K;
  __global const CL_DTYPE* A3 = a + ((idy + 3) < M ? (idy + 3) : (M - 1)) * K;

  int p = 0;
  for (; p + 4 <= K; p += 4) {
    a0 = vload4(0, A0 + p);
    a1 = vload4(0, A1 + p);
    a2 = vload4(0, A2 + p);
    a3 = vload4(0, A3 + p);

    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(p >> 2, (idx << 2)));
    b1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, b, SAMPLER, (int2)(p >> 2, (idx << 2) + 1));
    b2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, b, SAMPLER, (int2)(p >> 2, (idx << 2) + 2));
    b3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, b, SAMPLER, (int2)(p >> 2, (idx << 2) + 3));

    c0.s0 += dot(a0, b0);
    c0.s1 += dot(a0, b1);
    c0.s2 += dot(a0, b2);
    c0.s3 += dot(a0, b3);

    c1.s0 += dot(a1, b0);
    c1.s1 += dot(a1, b1);
    c1.s2 += dot(a1, b2);
    c1.s3 += dot(a1, b3);

    c2.s0 += dot(a2, b0);
    c2.s1 += dot(a2, b1);
    c2.s2 += dot(a2, b2);
    c2.s3 += dot(a2, b3);

    c3.s0 += dot(a3, b0);
    c3.s1 += dot(a3, b1);
    c3.s2 += dot(a3, b2);
    c3.s3 += dot(a3, b3);
  }

  if (K % 4 != 0) {
    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(p >> 2, (idx << 2)));
    b1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, b, SAMPLER, (int2)(p >> 2, (idx << 2) + 1));
    b2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, b, SAMPLER, (int2)(p >> 2, (idx << 2) + 2));
    b3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, b, SAMPLER, (int2)(p >> 2, (idx << 2) + 3));

    CL_DTYPE b_v0[4] = {b0.s0, b0.s1, b0.s2, b0.s3};
    CL_DTYPE b_v4[4] = {b1.s0, b1.s1, b1.s2, b1.s3};
    CL_DTYPE b_v8[4] = {b2.s0, b2.s1, b2.s2, b2.s3};
    CL_DTYPE b_vc[4] = {b3.s0, b3.s1, b3.s2, b3.s3};
    for (; p < K; p++) {
      int b_id = p % 4;
      CL_DTYPE4 b_tmp =
          (CL_DTYPE4)(b_v0[b_id], b_v4[b_id], b_v8[b_id], b_vc[b_id]);
      c0 += (CL_DTYPE4)(A0[p]) * b_tmp;
      c1 += (CL_DTYPE4)(A1[p]) * b_tmp;
      c2 += (CL_DTYPE4)(A2[p]) * b_tmp;
      c3 += (CL_DTYPE4)(A3[p]) * b_tmp;
    }
  }

  if (((idx << 2) + 4) <= N) {
    if (idy < M) {
      vstore4(c0, 0, c + idy * N + (idx << 2));
    }
    if (idy + 1 < M) {
      vstore4(c1, 0, c + (idy + 1) * N + (idx << 2));
    }
    if (idy + 2 < M) {
      vstore4(c2, 0, c + (idy + 2) * N + (idx << 2));
    }
    if (idy + 3 < M) {
      vstore4(c3, 0, c + (idy + 3) * N + (idx << 2));
    }
  } else {
    CL_DTYPE c_v0[4] = {c0.s0, c0.s1, c0.s2, c0.s3};
    CL_DTYPE c_v4[4] = {c1.s0, c1.s1, c1.s2, c1.s3};
    CL_DTYPE c_v8[4] = {c2.s0, c2.s1, c2.s2, c2.s3};
    CL_DTYPE c_vc[4] = {c3.s0, c3.s1, c3.s2, c3.s3};
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
}

__kernel void mali_gemm_trans_4x4(__global const CL_DTYPE* a,
                                  __global const CL_DTYPE* b,
                                  __global const CL_DTYPE* bias,
                                  __global CL_DTYPE* c,
                                  const int M,
                                  const int N,
                                  const int K,
                                  __global const CL_DTYPE* alpha) {
  const int idy = get_global_id(0) << 2;  // id: [0, M>>2) height of out == M
  const int idx = get_global_id(1) << 2;  // id: [0, N>>2) width of out == N

  // if (idx == 0 && idy == 0){
  //   printf("mali_gemm_trans_4x4\n");
  // }

  if (idx >= N || idy >= M) return;

  CL_DTYPE4 bias0 = bias ? vload4(0, bias + idx) : (CL_DTYPE4)0;

  CL_DTYPE4 a0, a1, a2, a3;
  CL_DTYPE4 c0, c1, c2, c3;
  CL_DTYPE4 b0, b1, b2, b3;
  c0 = bias0;
  c1 = bias0;
  c2 = bias0;
  c3 = bias0;
  __global const CL_DTYPE* A0 = a + idy * K;
  __global const CL_DTYPE* A1 = a + ((idy + 1) < M ? (idy + 1) : (M - 1)) * K;
  __global const CL_DTYPE* A2 = a + ((idy + 2) < M ? (idy + 2) : (M - 1)) * K;
  __global const CL_DTYPE* A3 = a + ((idy + 3) < M ? (idy + 3) : (M - 1)) * K;

  __global const CL_DTYPE* B0 = b + idx * K;
  __global const CL_DTYPE* B1 = b + ((idx + 1) < N ? (idx + 1) : (N - 1)) * K;
  __global const CL_DTYPE* B2 = b + ((idx + 2) < N ? (idx + 2) : (N - 1)) * K;
  __global const CL_DTYPE* B3 = b + ((idx + 3) < N ? (idx + 3) : (N - 1)) * K;

  // if (idx == 4){printf("~~~~~~~`4\n");}

  int p = 0;
  for (; p + 4 <= K; p += 4) {
    a0 = vload4(0, A0 + p);
    a1 = vload4(0, A1 + p);
    a2 = vload4(0, A2 + p);
    a3 = vload4(0, A3 + p);

    b0 = convert_half4(vload4(0, B0 + p));
    b1 = convert_half4(vload4(0, B1 + p));
    b2 = convert_half4(vload4(0, B2 + p));
    b3 = convert_half4(vload4(0, B3 + p));

    c0.s0 += dot(a0, b0);
    c0.s1 += dot(a0, b1);
    c0.s2 += dot(a0, b2);
    c0.s3 += dot(a0, b3);

    c1.s0 += dot(a1, b0);
    c1.s1 += dot(a1, b1);
    c1.s2 += dot(a1, b2);
    c1.s3 += dot(a1, b3);

    c2.s0 += dot(a2, b0);
    c2.s1 += dot(a2, b1);
    c2.s2 += dot(a2, b2);
    c2.s3 += dot(a2, b3);

    c3.s0 += dot(a3, b0);
    c3.s1 += dot(a3, b1);
    c3.s2 += dot(a3, b2);
    c3.s3 += dot(a3, b3);

    // if (idx == 4 && idy == 0){
    //   printf("a: %f %f %f %f\n", a0.s0, a0.s1, a0.s2, a0.s3);
    //   printf("b: %f %f %f %f\n", b0.s0, b0.s1, b0.s2, b0.s3);
    //   printf("c: %f %f %f %f\n", c0.s0, c0.s1, c0.s2, c0.s3);
    // }
  }

  for (; p < K; p++) {
    CL_DTYPE4 b_tmp = (CL_DTYPE4)(B0[p], B1[p], B2[p], B3[p]);
    c0 += (CL_DTYPE4)(A0[p]) * b_tmp;
    c1 += (CL_DTYPE4)(A1[p]) * b_tmp;
    c2 += (CL_DTYPE4)(A2[p]) * b_tmp;
    c3 += (CL_DTYPE4)(A3[p]) * b_tmp;
  }

  if ((idx + 4) <= N) {
    if (idy < M) {
      vstore4(c0, 0, c + idy * N + idx);
    }
    if (idy + 1 < M) {
      vstore4(c1, 0, c + (idy + 1) * N + idx);
    }
    if (idy + 2 < M) {
      vstore4(c2, 0, c + (idy + 2) * N + idx);
    }
    if (idy + 3 < M) {
      vstore4(c3, 0, c + (idy + 3) * N + idx);
    }
  } else {
    CL_DTYPE c_v0[4] = {c0.s0, c0.s1, c0.s2, c0.s3};
    CL_DTYPE c_v4[4] = {c1.s0, c1.s1, c1.s2, c1.s3};
    CL_DTYPE c_v8[4] = {c2.s0, c2.s1, c2.s2, c2.s3};
    CL_DTYPE c_vc[4] = {c3.s0, c3.s1, c3.s2, c3.s3};
    for (int i = idx; i < N; i++) {
      int c_index = i - idx;
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
}

/*
__kernel void adreno_gemm_4x4(__global const CL_DTYPE* a,
                          __read_only image2d_t b,
                          __global const CL_DTYPE* bias,
                          __global CL_DTYPE* c,
                          const int M,
                          const int N,
                          const int K,
                          __global const CL_DTYPE* alpha) {
  const int idy = get_global_id(0) << 2;  // id: [0, M>>2) height of out == M
  const int idx = get_global_id(1);  // id: [0, N>>2) width of out == N

  if ((idx << 2) >= N || idy >= M) return;

  CL_DTYPE4 bias0 = bias ? vload4(idx, bias) : (CL_DTYPE4)0;
  // CL_DTYPE16 c_v16 = (CL_DTYPE16)(bias0.s0123, bias0.s0123, bias0.s0123,
bias0.s0123);
  CL_DTYPE16 c_v16 = (CL_DTYPE16)(0);

  CL_DTYPE8 a0, a1, a2, a3;
  CL_DTYPE4 b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, 0));
  CL_DTYPE4 b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, 1));
  const CL_DTYPE* A0 = a + idy * K;
  const CL_DTYPE* A1 = A0 + K;
  const CL_DTYPE* A2 = A1 + K;
  const CL_DTYPE* A3 = A2 + K;
  a0 = vload8(0, A0);
  // if (idx == 0 && idy == 0){
  //   printf("00======%f %f %f %f\n", a0.s0, a0.s1, a0.s2, a0.s3);
  // }
  for (int p = 0; p < K; p += 8) {
    a0 = vload8(0, A0 + p);
    a1 = vload8(0, A1 + p);
    a2 = vload8(0, A2 + p);
    a3 = vload8(0, A3 + p);

    // if (idx == 0 && idy == 0){
    //   printf("%d a1 ======%f %f %f %f\n", p, a0.s0, a0.s1, a0.s2, a0.s3,
a0.s4, a0.s5, a0.s6, a0.s7);
    //   printf("%d a1 ======%f %f %f %f\n", p, a1.s0, a1.s1, a1.s2, a1.s3,
a1.s4, a1.s5, a1.s6, a1.s7);
    //   printf("%d a1 ======%f %f %f %f\n", p, a2.s0, a2.s1, a2.s2, a2.s3,
a2.s4, a2.s5, a2.s6, a2.s7);
    //   printf("%d a1 ======%f %f %f %f\n", p, a3.s0, a3.s1, a3.s2, a3.s3,
a3.s4, a3.s5, a3.s6, a3.s7);
    //   printf("%d b1 ======%f %f %f %f\n", p, b0.s0, b0.s1, b0.s2, b0.s3);
    //   printf("%d c1 ======%f %f %f %f\n", p, c_v16.s0, c_v16.s1, c_v16.s2,
c_v16.s3);
    // }

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s0, (CL_DTYPE4)a1.s0,
(CL_DTYPE4)a2.s0, (CL_DTYPE4)a3.s0),
            (CL_DTYPE16)(b0.s0123, b0.s0123, b0.s0123, b0.s0123), c_v16);
    // if (idx == 0 && idy == 0){
    //   printf("%d b0 ======%f %f %f %f\n", p, b0.s0, b0.s1, b0.s2, b0.s3);
    //   printf("%d c0 ======%f %f %f %f\n", p, c_v16.s0, c_v16.s1, c_v16.s2,
c_v16.s3);
    // }

    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 2));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s1, (CL_DTYPE4)a1.s1,
(CL_DTYPE4)a2.s1, (CL_DTYPE4)a3.s1),
            (CL_DTYPE16)(b1.s0123, b1.s0123, b1.s0123, b1.s0123), c_v16);
    // if (idx == 0 && idy == 0){
    //   printf("%d b1 ======%f %f %f %f\n", p, b1.s0, b1.s1, b1.s2, b1.s3);
    //   printf("%d c1 ======%f %f %f %f\n", p, c_v16.s0, c_v16.s1, c_v16.s2,
c_v16.s3);
    // }
    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 3));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s2, (CL_DTYPE4)a1.s2,
(CL_DTYPE4)a2.s2, (CL_DTYPE4)a3.s2),
            (CL_DTYPE16)(b0.s0123, b0.s0123, b0.s0123, b0.s0123), c_v16);
    // if (idx == 0 && idy == 0){
    //   printf("%d b2 ======%f %f %f %f\n", p, b0.s0, b0.s1, b0.s2, b0.s3);
    //   printf("%d c2 ======%f %f %f %f\n", p, c_v16.s0, c_v16.s1, c_v16.s2,
c_v16.s3);
    // }

    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 4));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s3, (CL_DTYPE4)a1.s3,
(CL_DTYPE4)a2.s3, (CL_DTYPE4)a3.s3),
            (CL_DTYPE16)(b1.s0123, b1.s0123, b1.s0123, b1.s0123), c_v16);
    // if (idx == 0 && idy == 0){
    //   printf("%d b3 ======%f %f %f %f\n", p, b1.s0, b1.s1, b1.s2, b1.s3);
    //   printf("%d c3 ======%f %f %f %f\n", p, c_v16.s0, c_v16.s1, c_v16.s2,
c_v16.s3);
    // }
    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 5));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s4, (CL_DTYPE4)a1.s4,
(CL_DTYPE4)a2.s4, (CL_DTYPE4)a3.s4),
            (CL_DTYPE16)(b0.s0123, b0.s0123, b0.s0123, b0.s0123), c_v16);

    // if (idx == 0 && idy == 0){
    //   printf("%d b4 ======%f %f %f %f\n", p, b0.s0, b0.s1, b0.s2, b0.s3);
    //   printf("%d c4 ======%f %f %f %f\n", p, c_v16.s0, c_v16.s1, c_v16.s2,
c_v16.s3);
    // }

    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 6));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s5, (CL_DTYPE4)a1.s5,
(CL_DTYPE4)a2.s5, (CL_DTYPE4)a3.s5),
            (CL_DTYPE16)(b1.s0123, b1.s0123, b1.s0123, b1.s0123), c_v16);
    // if (idx == 0 && idy == 0){
    //   printf("%d b5 ======%f %f %f %f\n", p, b1.s0, b1.s1, b1.s2, b1.s3);
    //   printf("%d c5 ======%f %f %f %f\n", p, c_v16.s0, c_v16.s1, c_v16.s2,
c_v16.s3);
    // }
    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 7));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s6, (CL_DTYPE4)a1.s6,
(CL_DTYPE4)a2.s6, (CL_DTYPE4)a3.s6),
            (CL_DTYPE16)(b0.s0123, b0.s0123, b0.s0123, b0.s0123), c_v16);
    // if (idx == 0 && idy == 0){
    //   printf("%d b6 ======%f %f %f %f\n", p, b0.s0, b0.s1, b0.s2, b0.s3);
    //   printf("%d c6 ======%f %f %f %f\n", p, c_v16.s0, c_v16.s1, c_v16.s2,
c_v16.s3);
    // }
    b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 8));

    c_v16 = mad((CL_DTYPE16)((CL_DTYPE4)a0.s7, (CL_DTYPE4)a1.s7,
(CL_DTYPE4)a2.s7, (CL_DTYPE4)a3.s7),
            (CL_DTYPE16)(b1.s0123, b1.s0123, b1.s0123, b1.s0123), c_v16);
    // if (idx == 0 && idy == 0){
    //   printf("%d b7 ======%f %f %f %f\n", p, b1.s0, b1.s1, b1.s2, b1.s3);
    //   printf("%d c7 ======%f %f %f %f\n", p, c_v16.s0, c_v16.s1, c_v16.s2,
c_v16.s3);
    // }
    b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 9));
  }
  // if (idx == 0 && idy == 0){
  //   printf("c7 ======%f %f %f %f\n", c_v16.s0, c_v16.s1, c_v16.s2, c_v16.s3);
  // }
  vstore4(c_v16.s0123, 0, c + idy * N + (idx << 2));
  vstore4(c_v16.s4567, 0, c + (idy + 1) * N + (idx << 2));
  vstore4(c_v16.s89ab, 0, c + (idy + 2) * N + (idx << 2));
  vstore4(c_v16.scdef, 0, c + (idy + 3) * N + (idx << 2));
  // if (idx == 0 && idy == 0){
  //   printf("======%f %f %f %f\n", c_v16.s0, c_v16.s1, c_v16.s2, c_v16.s3);
  // }
}
*/

// __kernel void adreno_gemm_4x4(__global const CL_DTYPE* a,
//                           __read_only image2d_t b,
//                           __global const CL_DTYPE* bias,
//                           __global CL_DTYPE* c,
//                           const int M,
//                           const int N,
//                           const int K,
//                           __global const CL_DTYPE* alpha) {
//   const int idy = get_global_id(0) << 2;  // id: [0, M>>2) height of out == M
//   const int idx = get_global_id(1);  // id: [0, N>>2) width of out == N
//   if (idy <= M && (idx << 2) <= N) {
//     CL_DTYPE4 bias0 = bias ? vload4(idx, bias) : (CL_DTYPE4)0;

//     CL_DTYPE4 c0 = bias0, c1 = bias0, c2 = bias0, c3 = bias0;
//     CL_DTYPE4 a0, a1, a2, a3;
//     CL_DTYPE4 b0, b1, b2, b3;
//     for (int p = 0; p < K; p+=4) {
//       a0 = vload4((idy * K + p) >> 2, a);
//       a1 = vload4(((idy + 1) * K + p) >> 2, a);
//       a2 = vload4(((idy + 2) * K + p) >> 2, a);
//       a3 = vload4(((idy + 3) * K + p) >> 2, a);

//       b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p));
//       b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 1));
//       b2 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 2));
//       b3 = READ_IMG_TYPE(CL_DTYPE_CHAR, b, SAMPLER, (int2)(idx, p + 3));

//       c0 += a0.s0 * b0;
//       c0 += a0.s1 * b1;
//       c0 += a0.s2 * b2;
//       c0 += a0.s3 * b3;

//       c1 += a1.s0 * b0;
//       c1 += a1.s1 * b1;
//       c1 += a1.s2 * b2;
//       c1 += a1.s3 * b3;

//       c2 += a2.s0 * b0;
//       c2 += a2.s1 * b1;
//       c2 += a2.s2 * b2;
//       c2 += a2.s3 * b3;

//       c3 += a3.s0 * b0;
//       c3 += a3.s1 * b1;
//       c3 += a3.s2 * b2;
//       c3 += a3.s3 * b3;
//     }
//     vstore4(c0, (idy * N + (idx << 2)) >> 2, c);
//     vstore4(c1, ((idy + 1) * N + (idx << 2)) >> 2, c);
//     vstore4(c2, ((idy + 2) * N + (idx << 2)) >> 2, c);
//     vstore4(c3, ((idy + 3) * N + (idx << 2)) >> 2, c);
//   } else {
//   }
// }
