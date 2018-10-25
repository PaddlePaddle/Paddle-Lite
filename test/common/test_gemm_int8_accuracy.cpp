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
#include <random>
#include "../test_helper.h"
#include "common/log.h"
#include "memory/t_malloc.h"
#include "operators/math/gemm.h"
#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#define a(i, j) a[(i)*lda + (j)]
#define b(i, j) b[(i)*ldb + (j)]
#define c(i, j) c[(i)*ldc + (j)]
#define c1(i, j) c1[(i)*ldc + (j)]

using std::default_random_engine;
using std::uniform_int_distribution;

void print_matirx(int m, int n, int ldc, int32_t *c) {
  for (int i = 0; i < m; ++i) {
    std::cout << c(i, 0);
    for (int j = 1; j < n; ++j) {
      std::cout << " | " << c(i, j);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void print_matirx(int m, int n, int ldc, int8_t *c) {
  for (int i = 0; i < m; ++i) {
    std::cout << static_cast<int32_t>(c(i, 0));
    for (int j = 1; j < n; ++j) {
      std::cout << " | " << static_cast<int32_t>(c(i, j));
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int do_sgemm(int m, int n, int k, bool relu, int pr) {
  int lda = k;
  int ldb = n;
  int ldc = n;
  default_random_engine e;
  uniform_int_distribution<int8_t> pixel(-127, 127);
  int8_t *a = static_cast<int8_t *>(
      paddle_mobile::memory::Alloc(sizeof(int8_t) * m * k));
  int8_t *b = static_cast<int8_t *>(
      paddle_mobile::memory::Alloc(sizeof(int8_t) * k * n));
  int32_t *c = static_cast<int32_t *>(
      paddle_mobile::memory::Alloc(sizeof(int32_t) * m * n));
  int32_t *c1 = static_cast<int32_t *>(
      paddle_mobile::memory::Alloc(sizeof(int32_t) * m * n));

  for (int i = 0; i < m * k; ++i) {
    a[i] = pixel(e);
  }
  for (int i = 0; i < k * n; ++i) {
    b[i] = pixel(e);
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      int32_t r = 0;
      for (int p = 0; p < k; p++) {
        r += static_cast<int32_t>(a(i, p)) * static_cast<int32_t>(b(p, j));
      }
      c1(i, j) = r;
    }
  }

  paddle_mobile::operators::math::Gemm gemm;
#ifdef _OPENMP
  gemm.Sgemm_omp(m, n, k, static_cast<int8_t>(1), a, lda, b, ldb,
                 static_cast<int8_t>(0), c, ldc, relu, nullptr);
#else
  gemm.Sgemm(m, n, k, static_cast<int8_t>(1), a, lda, b, ldb,
             static_cast<int8_t>(0), c, ldc, relu, nullptr);
#endif
  int eq = 0;
  int neq = 0;
  for (int i = 0; i < m * n; ++i) {
    if (c[i] == c1[i]) {
      ++eq;
    } else {
      ++neq;
    }
  }

  if (pr > 0) {
    std::cout << "A:" << std::endl;
    print_matirx(m, k, lda, a);
    std::cout << "B:" << std::endl;
    print_matirx(k, n, ldb, b);
    std::cout << "C:" << std::endl;
    print_matirx(m, n, ldc, c);
    std::cout << "C1:" << std::endl;
    print_matirx(m, n, ldc, c1);
  }

  std::cout << "mnk=" << m << " " << n << " " << k << " relu=" << relu
            << "   eq=" << eq << " neq=" << neq << std::endl;

  paddle_mobile::memory::Free(a);
  paddle_mobile::memory::Free(b);
  paddle_mobile::memory::Free(c);
  paddle_mobile::memory::Free(c1);

  return 0;
}

int main() {
#ifdef _OPENMP
  omp_set_num_threads(8);
#endif
  do_sgemm(9, 9, 9, false, 1);
  do_sgemm(10, 6, 12, false, 0);
  do_sgemm(512, 256, 384, false, 0);
  do_sgemm(1366, 768, 256, false, 0);
  do_sgemm(1255, 755, 333, false, 0);
  do_sgemm(599, 1133, 393, false, 0);
  do_sgemm(777, 555, 999, false, 0);
  do_sgemm(333, 797, 939, false, 0);
  do_sgemm(1024, 1024, 1024, false, 0);

  return 0;
}
