/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include <iostream>
#include "common/log.h"
#include "operators/math/Gemm.h"

#define a(i, j) a[(j)*lda + (i)]
#define b(i, j) b[(j)*ldb + (i)]
#define c1(i, j) c1[(j)*ldc + (i)]

int main() {
  int m = 45;
  int n = 46;
  int k = 125;
  int lda = m;
  int ldb = k;
  int ldc = m;

  float a[45 * 125];
  float b[125 * 46];
  float c[45 * 46] = {0};
  float c1[45 * 46] = {0};
  for (int i = 0; i < m * k; ++i) {
    a[i] = 2;
  }
  for (int i = 0; i < k * n; ++i) {
    b[i] = 2;
  }

  paddle_mobile::operators::math::sgemm(m, n, k, 1, a, lda, b, ldb, 0, c, ldc);
  for (int i = 0; i < m * n; ++i) {
    std::cout << c[i] << " | ";
    if (i % n == (n - 1)) {
      std::cout << std::endl;
    }
  }
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      for (int p = 0; p < k; ++p) {
        c1(i, j) += a(i, p) * b(p, j);
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
