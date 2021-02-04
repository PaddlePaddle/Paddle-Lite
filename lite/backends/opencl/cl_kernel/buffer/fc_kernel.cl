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
// fc_gemv_1x4: for fc with M = 1
// a: param.input  {M, K}
// b: param.w      {K, N}
// c: param.output {M, N}
__kernel
void fc_gemv_1x4(__global const float* a,
                 __global const float* b,
                 __global const float* bias,
                 __global float* c,
                 const int M, const int N, const int K,
                 __global const float* alpha) {
    const int col = get_global_id(0) << 2; // gws[0]: [0, N >> 2) height of B == N

    if (col + 3 < N) {
        half4 c0 = 0.0f;
        if (bias) {
            c0.x = bias[col];
            c0.y = bias[col+1];
            c0.z = bias[col+2];
            c0.w = bias[col+3];
        }

        // main loop of K
        int p = 0;
        for (; p < K - 3; p += 4) {
            half4 a0 = convert_half4(vload4(0, a + p));

            half4 b0 = convert_half4(vload4(0, b + p * N + col));
            half4 b1 = convert_half4(vload4(0, b + (p+1) * N + col));
            half4 b2 = convert_half4(vload4(0, b + (p+2) * N + col));
            half4 b3 = convert_half4(vload4(0, b + (p+3) * N + col));

            c0 += a0.x * b0;
            c0 += a0.y * b1;
            c0 += a0.z * b2;
            c0 += a0.w * b3;
        }

        // compute left K
        half4 b2 = 0.0f,
                          b1 = 0.0f,
                          b0 = 0.0f,
                          a0 = 0.0f;
        switch (K - p) {
            case 3: {
                b2 = convert_half4(vload4(0, b + (p+2) * N + col));
                a0.z = a[p + 2];
            }
            case 2: {
                b1 = convert_half4(vload4(0, b + (p+1) * N + col));
                a0.y = a[p + 1];
            }
            case 1: {
                b0 = convert_half4(vload4(0, b + (p) * N + col));
                a0.x = a[p];
            }
        }
        c0 += a0.x * b0;
        c0 += a0.y * b1;
        c0 += a0.z * b2;

        half4 alpha0 = 0.0f;
#ifdef PRELU
#ifdef PRELU_MORE
        alpha0.x = alpha[col];
        alpha0.y = alpha[col+1];
        alpha0.z = alpha[col+2];
        alpha0.w = alpha[col+3];
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
#else
       if (col % 4 == 0) {
            float4 act_res = convert_float4(activation_type4(c0));
            vstore4(act_res, 0, c + col);
        } else {
            switch (col % 4) {
                case 3:
                    c[col + 2] = activation(c0.z);
                case 2:
                    c[col + 1] = activation(c0.y);
                case 1:
                    c[col] = activation(c0.x);
            }
        }
#endif
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
#ifdef PRELU
#ifdef PRELU_MORE
           alpha0 = alpha[col];
#else
           alpha0 = alpha[0];
#endif
           c[col + col_offset] = activation(c0, alpha0);
#else
           c[col + col_offset] = activation(c0);
#endif

       }
    }
}


// fc_gemm_4x4: for fc with M = 1
// a: param.input  {M, K}
// b: param.w      {K, N}
// c: param.output {M, N}
__kernel
void fc_gemm_4x4(__global const float* a,
                 __global const float* b,
                 __global const float* bias,
                 __global float* c,
                 const int M, const int N, const int K,
                 __global const float* alpha) {
    const int row = get_global_id(0) << 2; // id: [0, M>>2) height of out == M
    const int col = get_global_id(1) << 2; // id: [0, N>>2) width of out == N

    if (row+3 < M && col+3 < N) {
        CL_COMPUTE_DTYPE bias0 = bias ? bias[col]   : 0,
                         bias1 = bias ? bias[col+1] : 0,
                         bias2 = bias ? bias[col+2] : 0,
                         bias3 = bias ? bias[col+3] : 0;

        CL_COMPUTE_DTYPE c00 = bias0, c01 = bias1, c02 = bias2, c03 = bias3,
                         c10 = bias0, c11 = bias1, c12 = bias2, c13 = bias3,
                         c20 = bias0, c21 = bias1, c22 = bias2, c23 = bias3,
                         c30 = bias0, c31 = bias1, c32 = bias2, c33 = bias3;

       for (int p = 0; p < K; ++p) {
            CL_COMPUTE_DTYPE
                a00 = *(a + row       * K + p),
                a10 = *(a + (row + 1) * K + p),
                a20 = *(a + (row + 2) * K + p),
                a30 = *(a + (row + 3) * K + p),

                b00 = *(b + p * N + col),
                b01 = *(b + p * N + (col + 1)),
                b02 = *(b + p * N + (col + 2)),
                b03 = *(b + p * N + (col + 3));

            c00 += a00 * b00; c01 += a00 * b01; c02 += a00 * b02; c03 += a00 * b03;
            c10 += a10 * b00; c11 += a10 * b01; c12 += a10 * b02; c13 += a10 * b03;
            c20 += a20 * b00; c21 += a20 * b01; c22 += a20 * b02; c23 += a20 * b03;
            c30 += a30 * b00; c31 += a30 * b01; c32 += a30 * b02; c33 += a30 * b03;
        }
        half alpha0 = 0.0f;
        half alpha1 = 0.0f;
        half alpha2 = 0.0f;
        half alpha3 = 0.0f;
#ifdef PRELU
#ifdef PRELU_MORE
        alpha0 = alpha[col];
        alpha1 = alpha[col+1];
        alpha2 = alpha[col+2];
        alpha3 = alpha[col+3];
#else
        alpha0 = alpha[0];
        alpha1 = alpha[0];
        alpha2 = alpha[0];
        alpha3 = alpha[0];
#endif
        c[row*N+col] = activation(c00, alpha0);     c[row*N+(col+1)] = activation(c01, alpha1);     c[row*N+(col+2)] = activation(c02, alpha2);     c[row*N+(col+3)] = activation(c03, alpha3);
        c[(row+1)*N+col] = activation(c10, alpha0); c[(row+1)*N+(col+1)] = activation(c11, alpha1); c[(row+1)*N+(col+2)] = activation(c12, alpha2); c[(row+1)*N+(col+3)] = activation(c13, alpha3);
        c[(row+2)*N+col] = activation(c20, alpha0); c[(row+2)*N+(col+1)] = activation(c21, alpha1); c[(row+2)*N+(col+2)] = activation(c22, alpha2); c[(row+2)*N+(col+3)] = activation(c23, alpha3);
        c[(row+3)*N+col] = activation(c30, alpha0); c[(row+3)*N+(col+1)] = activation(c31, alpha1); c[(row+3)*N+(col+2)] = activation(c32, alpha2); c[(row+3)*N+(col+3)] = activation(c33, alpha3);
#else
        c[row*N+col] = activation(c00);     c[row*N+(col+1)] = activation(c01);     c[row*N+(col+2)] = activation(c02);     c[row*N+(col+3)] = activation(c03);
        c[(row+1)*N+col] = activation(c10); c[(row+1)*N+(col+1)] = activation(c11); c[(row+1)*N+(col+2)] = activation(c12); c[(row+1)*N+(col+3)] = activation(c13);
        c[(row+2)*N+col] = activation(c20); c[(row+2)*N+(col+1)] = activation(c21); c[(row+2)*N+(col+2)] = activation(c22); c[(row+2)*N+(col+3)] = activation(c23);
        c[(row+3)*N+col] = activation(c30); c[(row+3)*N+(col+1)] = activation(c31); c[(row+3)*N+(col+2)] = activation(c32); c[(row+3)*N+(col+3)] = activation(c33);
#endif
    } else {
        for (int cidx = col; cidx < N; ++cidx) {
            for (int ridx = row; ridx < M; ++ridx) {
                CL_COMPUTE_DTYPE a0 = 0;
                CL_COMPUTE_DTYPE b0 = 0;
                CL_COMPUTE_DTYPE c0 = bias ? bias[cidx] : 0;
                for (int p = 0; p < K; ++p) {
                    a0 = *(a + ridx * K + p);
                    b0 = *(b + p * N + cidx),
                    c0 += a0 * b0;
                }
                half alpha0 = 0.0f;
#ifdef PRELU
#ifdef PRELU_MORE
                alpha0 = alpha[cidx];
#else
                alpha0 = alpha[0];
#endif
                c[ridx * N + cidx] = activation(c0, alpha0);
#else
                c[ridx * N + cidx] = activation(c0);
#endif
            }
        }
    }
}
