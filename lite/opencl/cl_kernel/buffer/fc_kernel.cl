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
__kernel
void mat_transpose(__global const CL_DTYPE* src,
                   __global CL_DTYPE* dst,
                   const int src_height, const int src_width) {
  const int col = get_global_id(0); // [0, src_width)  columns of src
  const int row = get_global_id(1); // [0, src_height) rows of src
  DST(col, row) = SRC(row, col);
}

#if 0
// naive gemm: keep for check
__kernel
void fc(__global const CL_DTYPE* a,
        __global const CL_DTYPE* b,
        __global const CL_DTYPE* bias,
        __global CL_DTYPE* c,
        const int M, const int N, const int K) {
  const int row = get_global_id(0); // [0, M) height of out == m
  const int col = get_global_id(1); // [0, N) width of out == n

  if ((col >= N) || (row >= M)) {
    return;
  }

  CL_DTYPE a0, b0,
      c0 = (bias && col < N) ? bias[col] : 0;

  for (int p = 0; p < K; ++p) {
    a0 = *(a + row * K + p);
    b0 = *(b + p * N + col);
    c0 += a0 * b0;
  }

#if defined(RELU)
  c[row * N + col] = max(c0, 0);
#else
  c[row * N + col] = c0;
#endif

}
#endif // naive gemm


__kernel
void fc(__global const CL_DTYPE* a,
        __global const CL_DTYPE* b,
        __global const CL_DTYPE* bias,
        __global CL_DTYPE* c,
        const int M, const int N, const int K) {
    const int row = get_global_id(0) << 2; // id: [0, M>>2) height of out == M
    const int col = get_global_id(1) << 2; // id: [0, N>>2) width of out == N

    if (row+3 < M && col+3 < N) {
        CL_DTYPE bias0 = bias ? bias[col]   : 0,
                 bias1 = bias ? bias[col+1] : 0,
                 bias2 = bias ? bias[col+2] : 0,
                 bias3 = bias ? bias[col+3] : 0;

        CL_DTYPE c00 = bias0, c01 = bias1, c02 = bias2, c03 = bias3,
                 c10 = bias0, c11 = bias1, c12 = bias2, c13 = bias3,
                 c20 = bias0, c21 = bias1, c22 = bias2, c23 = bias3,
                 c30 = bias0, c31 = bias1, c32 = bias2, c33 = bias3;

       for (int p = 0; p < K; ++p) {
            CL_DTYPE
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
#if defined(RELU)
        c[row*N+col] = max(c00, 0);     c[row*N+(col+1)] = max(c01, 0);     c[row*N+(col+2)] = max(c02, 0);     c[row*N+(col+3)] = max(c03, 0);
        c[(row+1)*N+col] = max(c10, 0); c[(row+1)*N+(col+1)] = max(c11, 0); c[(row+1)*N+(col+2)] = max(c12, 0); c[(row+1)*N+(col+3)] = max(c13, 0);
        c[(row+2)*N+col] = max(c20, 0); c[(row+2)*N+(col+1)] = max(c21, 0); c[(row+2)*N+(col+2)] = max(c22, 0); c[(row+2)*N+(col+3)] = max(c23, 0);
        c[(row+3)*N+col] = max(c30, 0); c[(row+3)*N+(col+1)] = max(c31, 0); c[(row+3)*N+(col+2)] = max(c32, 0); c[(row+3)*N+(col+3)] = max(c33, 0);
#else
        c[row*N+col] = c00;     c[row*N+(col+1)] = c01;     c[row*N+(col+2)] = c02;     c[row*N+(col+3)] = c03;
        c[(row+1)*N+col] = c10; c[(row+1)*N+(col+1)] = c11; c[(row+1)*N+(col+2)] = c12; c[(row+1)*N+(col+3)] = c13;
        c[(row+2)*N+col] = c20; c[(row+2)*N+(col+1)] = c21; c[(row+2)*N+(col+2)] = c22; c[(row+2)*N+(col+3)] = c23;
        c[(row+3)*N+col] = c30; c[(row+3)*N+(col+1)] = c31; c[(row+3)*N+(col+2)] = c32; c[(row+3)*N+(col+3)] = c33;
#endif
    } else {
        for (int cidx = col; cidx < N; ++cidx) {
            for (int ridx = row; ridx < M; ++ridx) {
                CL_DTYPE a0, b0, c0 = bias ? bias[cidx] : 0;
                for (int p = 0; p < K; ++p) {
                    a0 = *(a + ridx * K + p);
                    b0 = *(b + p * N + cidx),
                    c0 += a0 * b0;
                }
#if defined(RELU)
                c[ridx * N + cidx] = max(c0, 0);
#else
                c[ridx * N + cidx] = c0;
#endif
            }
        }
    }
}
