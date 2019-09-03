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

#if 0
// naive gemm: keep for check
__kernel
void mat_mul(__global const CL_DTYPE* x,
             __global const CL_DTYPE* y,
             __global CL_DTYPE* out,
             const int M, const int N, const int K) {
  const int row = get_global_id(0); // [0, M) columns of out == m
  const int col = get_global_id(1); // [0, N) rows of out == n

  if ((col >= N) || (row >= M)) {
    return;
  }

  CL_DTYPE x0, y0,
        out0 = 0;

  for (int p = 0; p < K; ++p) {
    x0 = *(x + row * K + p);
    y0 = *(y + p * N + col);
    out0 += x0 * y0;
  }

  out[row * N + col] = out0;
}
#endif // naive gemm

__kernel
void mat_mul(__global const CL_DTYPE* a,
             __global const CL_DTYPE* b,
             __global CL_DTYPE* c,
             const int M, const int N, const int K) {
    const int row = get_global_id(0) << 2; // id: [0, M>>2) height of out == M
    const int col = get_global_id(1) << 2; // id: [0, N>>2) width of out == N

    if (row+3 < M && col+3 < N) {
        CL_DTYPE c00 = 0, c01 = 0, c02 = 0, c03 = 0,
                 c10 = 0, c11 = 0, c12 = 0, c13 = 0,
                 c20 = 0, c21 = 0, c22 = 0, c23 = 0,
                 c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        for (int p = 0; p < K; p++) {

            CL_DTYPE a00 = *(a + row       * K + p), 
                     a10 = *(a + (row + 1) * K + p), 
                     a20 = *(a + (row + 2) * K + p), 
                     a30 = *(a + (row + 3) * K + p),  

                     b00 = *(b + p * N + col),
                     b01 = *(b + p * N + (col+1)),
                     b02 = *(b + p * N + (col+2)),
                     b03 = *(b + p * N + (col+3));
        
            c00 += a00 * b00; c01 += a00 * b01; c02 += a00 * b02; c03 += a00 * b03;
            c10 += a10 * b00; c11 += a10 * b01; c12 += a10 * b02; c13 += a10 * b03;
            c20 += a20 * b00; c21 += a20 * b01; c22 += a20 * b02; c23 += a20 * b03;
            c30 += a30 * b00; c31 += a30 * b01; c32 += a30 * b02; c33 += a30 * b03;
        }
        c[row*N+col] = c00;     c[row*N+(col+1)] = c01;     c[row*N+(col+2)] = c02;     c[row*N+(col+3)] = c03;
        c[(row+1)*N+col] = c10; c[(row+1)*N+(col+1)] = c11; c[(row+1)*N+(col+2)] = c12; c[(row+1)*N+(col+3)] = c13;
        c[(row+2)*N+col] = c20; c[(row+2)*N+(col+1)] = c21; c[(row+2)*N+(col+2)] = c22; c[(row+2)*N+(col+3)] = c23;
        c[(row+3)*N+col] = c30; c[(row+3)*N+(col+1)] = c31; c[(row+3)*N+(col+2)] = c32; c[(row+3)*N+(col+3)] = c33;
    } else {
        for(int cidx = col; cidx < N; ++cidx) {
            for (int ridx = row; ridx < M; ++ridx) {
                CL_DTYPE a0, b0, c0 = 0;
                for (int p = 0; p < K; ++p) {
                    a0 = *(a + ridx * K + p);
                    b0 = *(b + p * N + cidx),
                    c0 += a0 * b0;
                }
                c[ridx * N + cidx] = c0;
            }
        }
    }
} 

