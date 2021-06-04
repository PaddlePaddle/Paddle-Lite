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

#undef A
#undef B
#undef C

#define C(i, j, ldc) c[mad24(i, ldc, j)]
#define A(i, j, lda) a[mad24(i, lda, j)]
#define B(i, j, ldb) b[mad24(i, ldb, j)]

__kernel void mat_mul_naive(__global const CL_DTYPE* a,
                            __global const CL_DTYPE* b,
                            __global CL_DTYPE* c,
                            const int M,
                            const int N,
                            const int K,
                            const int lda,
                            const int ldb,
                            const int ldc,
                            const float alpha) {
  const int m = get_global_id(0);
  const int n = get_global_id(1);

  CL_DTYPE res = 0.f;
  for (short k = 0; k < K; ++k) {
    res += A(m, k, lda) * B(k, n, ldb);
  }
  C(m, n, ldc) = res * alpha;
}

// gemv_1x4: M = 1
// a: param.input  {M, K}
// b: param.w      {K, N}
// c: param.output {M, N}
__kernel void gemv_1x4(__global const CL_DTYPE* a,
                       __global const CL_DTYPE* b,
                       __global CL_DTYPE* c,
                       const int M,
                       const int N,
                       const int K,
                       const int lda,
                       const int ldb,
                       const int ldc,
                       const CL_DTYPE alpha) {
  const int col = get_global_id(0)
                  << 2;  // gws[0]: [0, N >> 2) height of B == N

  if (col + 3 < N) {
    half4 a1x4 = 0.0f, b0_1x4 = 0.0f, b1_1x4 = 0.0f, b2_1x4 = 0.0f,
          b3_1x4 = 0.0f, c1x4 = 0.0f;

    // main loop of K
    short p = 0;
    __global CL_DTYPE* base_b = (__global CL_DTYPE*)&B(p, col, ldb);
    for (; p < K - 3; p += 4) {
      a1x4 = convert_half4(vload4(0, &A(0, p, lda)));

      b0_1x4 = convert_half4(vload4(0, base_b));
      b1_1x4 = convert_half4(vload4(0, base_b + ldb));
      b2_1x4 = convert_half4(vload4(0, base_b + ldb * 2));
      b3_1x4 = convert_half4(vload4(0, base_b + ldb * 3));

      c1x4 = mad(a1x4.x, b0_1x4, c1x4);
      c1x4 = mad(a1x4.y, b1_1x4, c1x4);
      c1x4 = mad(a1x4.z, b2_1x4, c1x4);
      c1x4 = mad(a1x4.w, b3_1x4, c1x4);

      base_b += 4 * ldb;
    }

    // compute left K with b3_1x4 unused
    b2_1x4 = 0.0f;
    b1_1x4 = 0.0f;
    b0_1x4 = 0.0f;
    a1x4 = 0.0f;

    switch (K - p) {
      case 3: {
        b2_1x4 = convert_half4(vload4(0, &B(p + 2, col, ldb)));
        a1x4.z = a[p + 2];
      }
      case 2: {
        b1_1x4 = convert_half4(vload4(0, &B(p + 1, col, ldb)));
        a1x4.y = a[p + 1];
      }
      case 1: {
        b0_1x4 = convert_half4(vload4(0, &B(p, col, ldb)));
        a1x4.x = a[p];
      }
    }
    c1x4 = mad(a1x4.x, b0_1x4, c1x4);
    c1x4 = mad(a1x4.y, b1_1x4, c1x4);
    c1x4 = mad(a1x4.z, b2_1x4, c1x4);
    c1x4 = c1x4 * (half4)alpha;

    if (col % 4 == 0) {
      float4 c_res = convert_float4(c1x4);
      vstore4(c_res, 0, c + col);
    } else {
      switch (col % 4) {
        case 3:
          c[col + 2] = c1x4.z;
        case 2:
          c[col + 1] = c1x4.y;
        case 1:
          c[col] = c1x4.x;
      }
    }
  } else {
    for (short col_idx = col; col_idx < N; ++col_idx) {
      half c0 = 0, a0 = 0, b0 = 0;
      for (short p = 0; p < K; ++p) {
        b0 = B(p, col_idx, ldb);
        a0 = A(0, p, lda);
        c0 = mad(a0, b0, c0);
      }
      C(0, col_idx, ldc) = c0 * alpha;
    }
  }
}
