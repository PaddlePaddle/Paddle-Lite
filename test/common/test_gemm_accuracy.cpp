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

#include <cstdlib>
#include <ctime>
#include <iostream>
#include "../test_helper.h"
#include "common/log.h"
#include "memory/t_malloc.h"
#include "operators/math/gemm.h"

#define a(i, j) a[(i)*lda + (j)]
#define b(i, j) b[(i)*ldb + (j)]
#define c(i, j) c[(i)*ldc + (j)]
#define c1(i, j) c1[(i)*ldc + (j)]

void print_matrix(int m, int n, int ldc, float *c) {
  for (int i = 0; i < m; ++i) {
    std::cout << c(i, 0);
    for (int j = 1; j < n; ++j) {
      std::cout << " | " << c(i, j);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int do_sgemm(int m, int n, int k, bool relu, int t1, int t2, int pr) {
  int lda = k;
  int ldb = n;
  int ldc = n;

  float *a =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * m * k));
  float *b =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * k * n));
  float *c =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * m * n));
  float *c1 =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * m * n));
  float *scale =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * m));
  float *bias =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * m));

  srand(unsigned(time(0)));
  for (int i = 0; i < m * k; ++i) {
    a[i] = t1 + rand() % t2;
  }
  for (int i = 0; i < k * n; ++i) {
    b[i] = t1 + rand() % t2;
  }
  for (int i = 0; i < m; ++i) {
    scale[i] = t1 + rand() % t2;
  }
  for (int i = 0; i < m; ++i) {
    bias[i] = t1 + rand() % t2;
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float r = 0;
      for (int p = 0; p < k; p++) {
        r += a(i, p) * b(p, j);
      }
      r *= scale[i];
      r += bias[i];
      if (relu && (r < 0)) {
        r = 0;
      }
      c1(i, j) = r;
    }
  }

  paddle_mobile::operators::math::Gemm gemm;
  gemm.SgemmWithBn(m, n, k, 1, a, lda, b, ldb, 0.3, c, ldc, relu, scale, bias,
                   nullptr);
  int eq = 0;
  int neq = 0;
  for (int i = 0; i < m * n; ++i) {
    if (static_cast<int>(c[i]) == static_cast<int>(c1[i])) {
      ++eq;
    } else {
      ++neq;
    }
  }

  if (pr > 0) {
    std::cout << "A:" << std::endl;
    print_matrix(m, k, lda, a);
    std::cout << "B:" << std::endl;
    print_matrix(k, n, ldb, b);
    std::cout << "C:" << std::endl;
    print_matrix(m, n, ldc, c);
    std::cout << "C1:" << std::endl;
    print_matrix(m, n, ldc, c1);
  }

  std::cout << "mnk=" << m << " " << n << " " << k << " relu=" << relu
            << "   eq=" << eq << " neq=" << neq << std::endl;

  PADDLE_MOBILE_ENFORCE(neq == 0, "The execution of do_sgemm is failed!");

  paddle_mobile::memory::Free(a);
  paddle_mobile::memory::Free(b);
  paddle_mobile::memory::Free(c);
  paddle_mobile::memory::Free(c1);
  paddle_mobile::memory::Free(scale);
  paddle_mobile::memory::Free(bias);

  return 0;
}

int main() {
  do_sgemm(9, 9, 9, true, 10, 10, 10);
  do_sgemm(10, 6, 12, false, 10, 10, 0);
  do_sgemm(512, 256, 384, false, 10, 10, 0);
  do_sgemm(1366, 768, 256, false, 10, 10, 0);
  do_sgemm(1255, 755, 333, false, 10, 10, 0);
  do_sgemm(555, 777, 999, false, 10, 10, 0);

  do_sgemm(10, 6, 12, true, -4, 10, 0);
  do_sgemm(512, 256, 384, true, -4, 10, 0);
  do_sgemm(1366, 768, 256, true, -4, 10, 0);
  do_sgemm(1255, 755, 333, true, -4, 10, 0);
  do_sgemm(555, 777, 999, true, -4, 10, 0);
  return 0;
}
