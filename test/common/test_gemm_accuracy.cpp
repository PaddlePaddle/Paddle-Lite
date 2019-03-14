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
#include "operators/math/gemm/cblas.h"

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

int do_sgemm(int m, int n, int k, int pr) {
  const float alpha = 1.f;
  const float beta = 0.f;
  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  float *a =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * m * k));
  float *b =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * k * n));
  float *c =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * m * n));
  float *c1 =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * m * n));

  std::mt19937 rng(111);
  std::uniform_real_distribution<double> uniform_dist(0, 1);
  const float lower = -10.f;
  const float upper = 10.f;

  for (int i = 0; i < m * k; ++i) {
    a[i] = static_cast<float>(uniform_dist(rng) * (upper - lower) + lower);
  }
  for (int i = 0; i < k * n; ++i) {
    b[i] = static_cast<float>(uniform_dist(rng) * (upper - lower) + lower);
  }
  memcpy(c, c1, sizeof(float) * m * n);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float r = 0;
      for (int p = 0; p < k; p++) {
        r += a(i, p) * b(p, j);
      }
      c1(i, j) = alpha * r;
    }
  }

  std::cout << "run cblas_sgemm..." << std::endl;
  paddle_mobile::operators::math::cblas_sgemm(false, false, m, n, k, alpha, a,
                                              lda, b, ldb, 0.f, c, ldc);

  std::cout << "compare results..." << std::endl;
  for (int i = 0; i < m * n; ++i) {
    if (abs(c[i] - c1[i]) >= 1e-2) {
      std::cout << "c[" << i << "] != c1[" << i << "]: " << c[i] << " vs "
                << c1[i] << std::endl;
      exit(1);
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

  paddle_mobile::memory::Free(a);
  paddle_mobile::memory::Free(b);
  paddle_mobile::memory::Free(c);
  paddle_mobile::memory::Free(c1);

  return 0;
}

int main(int argc, char *argv[]) {
  do_sgemm(1, 1, 1, 1);

  do_sgemm(9, 9, 1, 1);
  do_sgemm(999, 99, 1, 0);
  do_sgemm(999, 1, 1, 0);
  do_sgemm(1, 9, 9, 1);
  do_sgemm(1, 99, 999, 0);
  do_sgemm(1, 1, 999, 0);

  do_sgemm(9, 9, 9, 1);
  do_sgemm(10, 6, 12, 1);
  do_sgemm(512, 256, 384, 0);
  do_sgemm(1366, 768, 256, 0);
  do_sgemm(1255, 755, 333, 0);
  do_sgemm(555, 777, 999, 0);

  do_sgemm(10, 6, 12, 1);
  do_sgemm(512, 256, 384, 0);
  do_sgemm(1366, 768, 256, 0);
  do_sgemm(1255, 755, 333, 0);
  do_sgemm(555, 777, 999, 0);

  return 0;
}
