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
#include <limits>
#include <random>
#include <type_traits>
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

template <typename T>
void print_matrix(int m, int n, int ldc, T *c) {
  for (int i = 0; i < m; ++i) {
    if (std::is_same<T, int8_t>::value) {
      std::cout.setf(std::ios::left);
      std::cout.width(4);
      std::cout << static_cast<int32_t>(c(i, 0));
    } else {
      std::cout.setf(std::ios::left);
      std::cout.width(6);
      std::cout << c(i, 0);
    }
    for (int j = 1; j < n; ++j) {
      if (std::is_same<T, int8_t>::value) {
        std::cout << " | ";
        std::cout.setf(std::ios::left);
        std::cout.width(4);
        std::cout << static_cast<int32_t>(c(i, j));
      } else {
        std::cout << " | ";
        std::cout.setf(std::ios::left);
        std::cout.width(6);
        std::cout << c(i, j);
      }
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

int32_t qadd_int32(int32_t l, int32_t r) {
  int64_t res = static_cast<int64_t>(l) + static_cast<int64_t>(r);
  if (res > std::numeric_limits<int32_t>::max())
    return std::numeric_limits<int32_t>::max();
  else if (res < std::numeric_limits<int32_t>::min())
    return std::numeric_limits<int32_t>::min();
  else
    return static_cast<int32_t>(res);
}

// round to zero
float round2zero(float v) {
  float res;
  if (v > 0)
    res = std::floor(v);
  else if (v < 0)
    res = std::ceil(v);
  return res;
}

int8_t qscale_int32(int32_t v, float scale) {
  float res = static_cast<float>(v) * scale;
  res = round2zero(res);
  if (res > 127)
    return static_cast<int8_t>(127);
  else if (res < -127)
    return static_cast<int8_t>(-127);
  else
    return static_cast<int8_t>(res);
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

  return 0;
}

int do_sgemm_with_bias(int m, int n, int k, bool relu, int pr,
                       bool addOnRow = false) {
  int lda = k;
  int ldb = n;
  int ldc = n;
  float scale = 1;
  default_random_engine e;
  uniform_int_distribution<int8_t> pixel(-127, 127);
  int8_t *a = static_cast<int8_t *>(
      paddle_mobile::memory::Alloc(sizeof(int8_t) * m * k));
  int8_t *b = static_cast<int8_t *>(
      paddle_mobile::memory::Alloc(sizeof(int8_t) * k * n));
  int8_t *c = static_cast<int8_t *>(
      paddle_mobile::memory::Alloc(sizeof(int8_t) * m * n));
  int8_t *c1 = static_cast<int8_t *>(
      paddle_mobile::memory::Alloc(sizeof(int8_t) * m * n));

  int32_t *bias = nullptr;
  if (addOnRow) {
    bias = static_cast<int32_t *>(
        paddle_mobile::memory::Alloc(sizeof(int32_t) * n));
  } else {
    bias = static_cast<int32_t *>(
        paddle_mobile::memory::Alloc(sizeof(int32_t) * m));
  }

  for (int i = 0; i < m * k; ++i) {
    a[i] = pixel(e);
  }
  for (int i = 0; i < k * n; ++i) {
    b[i] = pixel(e);
  }

  if (addOnRow) {
    for (int i = 0; i < n; ++i) {
      bias[i] = static_cast<int32_t>(pixel(e));
    }
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        int32_t bias_v = bias[j];
        int32_t r = 0;
        for (int p = 0; p < k; p++) {
          r += static_cast<int32_t>(a(i, p)) * static_cast<int32_t>(b(p, j));
        }
        r = qadd_int32(r, bias_v);
        if (relu) r = std::max(0, r);
        c1(i, j) = qscale_int32(r, scale);
      }
    }
  } else {
    for (int i = 0; i < m; ++i) {
      bias[i] = static_cast<int32_t>(pixel(e));
    }
    for (int i = 0; i < m; ++i) {
      int32_t bias_v = bias[i];
      for (int j = 0; j < n; ++j) {
        int32_t r = 0;
        for (int p = 0; p < k; p++) {
          r += static_cast<int32_t>(a(i, p)) * static_cast<int32_t>(b(p, j));
        }
        r = qadd_int32(r, bias_v);
        if (relu) r = std::max(0, r);
        c1(i, j) = qscale_int32(r, scale);
      }
    }
  }

  paddle_mobile::operators::math::Gemm gemm;
#ifdef _OPENMP
  gemm.Sgemm_omp(m, n, k, scale, a, lda, b, ldb, static_cast<float>(0), c, ldc,
                 relu, bias, addOnRow);
#else
  gemm.Sgemm(m, n, k, scale, a, lda, b, ldb, static_cast<float>(0), c, ldc,
             relu, bias, addOnRow);
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
    print_matrix(m, k, lda, a);
    std::cout << "B:" << std::endl;
    print_matrix(k, n, ldb, b);
    std::cout << "Bias:" << std::endl;
    if (addOnRow) {
      print_matrix(1, n, n, bias);
    } else {
      print_matrix(m, 1, 1, bias);
    }
    std::cout << "C:" << std::endl;
    print_matrix(m, n, ldc, c);
    std::cout << "C1:" << std::endl;
    print_matrix(m, n, ldc, c1);
  }

  std::cout << "mnk=" << m << " " << n << " " << k << " relu=" << relu
            << "   eq=" << eq << " neq=" << neq << std::endl;

  PADDLE_MOBILE_ENFORCE(neq == 0,
                        "The execution of do_sgemm_with_bias is failed!");

  paddle_mobile::memory::Free(a);
  paddle_mobile::memory::Free(b);
  paddle_mobile::memory::Free(c);
  paddle_mobile::memory::Free(c1);
  paddle_mobile::memory::Free(bias);

  return 0;
}

int main() {
#ifdef _OPENMP
  omp_set_num_threads(4);
#endif
  std::cout << "\n\n******************************************************\n\n"
            << std::endl;
  std::cout << "Test gemm without bias:" << std::endl;
  do_sgemm(9, 9, 9, false, 1);
  do_sgemm(10, 6, 12, false, 0);
  do_sgemm(512, 256, 384, false, 0);
  do_sgemm(1366, 768, 256, false, 0);
  do_sgemm(1255, 755, 333, false, 0);
  do_sgemm(599, 1133, 393, false, 0);
  do_sgemm(777, 555, 999, false, 0);
  do_sgemm(333, 797, 939, false, 0);
  do_sgemm(1024, 1024, 1024, false, 0);

  std::cout << "\n\n******************************************************\n\n"
            << std::endl;
  std::cout << "Test gemm with bias(bias is added on column):" << std::endl;
  do_sgemm_with_bias(9, 9, 9, false, 1);
  do_sgemm_with_bias(10, 6, 12, false, 0);
  do_sgemm_with_bias(512, 256, 384, false, 0);
  do_sgemm_with_bias(1366, 768, 256, false, 0);
  do_sgemm_with_bias(1255, 755, 333, false, 0);
  do_sgemm_with_bias(599, 1133, 393, false, 0);
  do_sgemm_with_bias(777, 555, 999, false, 0);
  do_sgemm_with_bias(333, 797, 939, false, 0);
  do_sgemm_with_bias(1024, 1024, 1024, false, 0);

  std::cout << "\n\n******************************************************\n\n"
            << std::endl;
  std::cout << "Test gemm with bias(bias is added on row):" << std::endl;
  do_sgemm_with_bias(9, 9, 9, false, 1, true);
  do_sgemm_with_bias(10, 6, 12, false, 0, true);
  do_sgemm_with_bias(512, 256, 384, false, 0, true);
  do_sgemm_with_bias(1366, 768, 256, false, 0, true);
  do_sgemm_with_bias(1255, 755, 333, false, 0, true);
  do_sgemm_with_bias(599, 1133, 393, false, 0, true);
  do_sgemm_with_bias(777, 555, 999, false, 0, true);
  do_sgemm_with_bias(333, 797, 939, false, 0, true);
  do_sgemm_with_bias(1024, 1024, 1024, false, 0, true);

  std::cout << "\n\n******************************************************\n\n"
            << std::endl;
  std::cout << "Test gemm with relu and bias:" << std::endl;
  do_sgemm_with_bias(9, 9, 9, true, 1);
  do_sgemm_with_bias(10, 6, 12, true, 0);
  do_sgemm_with_bias(512, 256, 384, true, 0);
  do_sgemm_with_bias(1366, 768, 256, true, 0);
  do_sgemm_with_bias(1255, 755, 333, true, 0);
  do_sgemm_with_bias(599, 1133, 393, true, 0);
  do_sgemm_with_bias(777, 555, 999, true, 0);
  do_sgemm_with_bias(333, 797, 939, true, 0);
  do_sgemm_with_bias(1024, 1024, 1024, true, 0);

  return 0;
}
