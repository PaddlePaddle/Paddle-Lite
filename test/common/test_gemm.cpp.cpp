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
