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

#include <iostream>
#include "common/log.h"
#include "operators/math/gemm.h"

#define a(i, j) a[(i)*lda + (j)]
#define b(i, j) b[(i)*ldb + (j)]
#define c1(i, j) c1[(i)*ldc + (j)]

#define m 62
#define n 63
#define k 74

int main() {
  int lda = k;
  int ldb = n;
  int ldc = n;

  float a[62 * 74];
  float b[74 * 63];
  float c[62 * 63] = {0};
  float c1[62 * 63] = {0};
  for (int i = 0; i < m * k; ++i) {
    a[i] = 2;
  }
  for (int i = 0; i < k * n; ++i) {
    b[i] = 2;
  }
  for (int i = 0; i < m * n; ++i) {
    c[i] = 2;
    c1[i] = 2;
  }

  paddle_mobile::operators::math::sgemm(m, n, k, 0.9, a, lda, b, ldb, 0.3, c,
                                        ldc);
  for (int i = 0; i < m * n; ++i) {
    std::cout << c[i] << " | ";
    if (i % n == (n - 1)) {
      std::cout << std::endl;
    }
  }
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      c1(i, j) *= 0.3;
      for (int p = 0; p < k; ++p) {
        c1(i, j) += 0.9 * a(i, p) * b(p, j);
      }
    }
  }
  std::cout << "正确结果对比:" << std::endl;
  for (int i = 0; i < m * n; ++i) {
    std::cout << c1[i] << " | ";
    if (i % n == (n - 1)) {
      std::cout << std::endl;
    }
  }
  return 0;
}
