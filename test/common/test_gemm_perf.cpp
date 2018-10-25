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
#include "../test_helper.h"
#include "../test_include.h"
#include "operators/math/gemm.h"
#include "operators/math/math_function.h"

#define a(i, j) a[(i)*lda + (j)]
#define b(i, j) b[(i)*ldb + (j)]
#define c1(i, j) c1[(i)*ldc + (j)]

#define m 1024
#define n 1024
#define k 1024

int main() {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(8);
  Tensor aa, bb, cc;
  auto aaptr = aa.mutable_data<float>({m, k});
  auto bbptr = bb.mutable_data<float>({k, n});
  auto ccptr = cc.mutable_data<float>({m, n});

  for (int i = 0; i < m * k; ++i) {
    aaptr[i] = 2;
  }
  for (int i = 0; i < k * n; ++i) {
    bbptr[i] = 2;
  }
  for (int i = 0; i < m * n; ++i) {
    ccptr[i] = 2;
  }

  Tensor aa_int8, bb_int8, cc_int8;
  auto aaptr_int8 = aa_int8.mutable_data<int8_t>({m, k});
  auto bbptr_int8 = bb_int8.mutable_data<int8_t>({k, n});
  auto ccptr_int8 = cc_int8.mutable_data<int32_t>({m, n});

  for (int i = 0; i < m * k; ++i) {
    aaptr_int8[i] = static_cast<int8_t>(2);
  }
  for (int i = 0; i < k * n; ++i) {
    bbptr_int8[i] = static_cast<int8_t>(2);
  }
  for (int i = 0; i < m * n; ++i) {
    ccptr_int8[i] = static_cast<int32_t>(2);
  }

  // float
  // warm-up 10 times
  for (int j = 0; j < 10; ++j) {
    paddle_mobile::operators::math::matmul<float>(
        aa, false, bb, false, static_cast<float>(1), &cc, static_cast<float>(0),
        false, nullptr);
  }

  auto time1 = time();
  for (int j = 0; j < 10; ++j) {
    paddle_mobile::operators::math::matmul<float>(
        aa, false, bb, false, static_cast<float>(1), &cc, static_cast<float>(0),
        false, nullptr);
  }
  auto time2 = time();
  std::cout << "float gemm  cost :" << time_diff(time1, time2) / 10 << "ms\n";

  // int8_t
  // warm-up 10 times
  for (int j = 0; j < 10; ++j) {
    paddle_mobile::operators::math::matmul<int8_t>(
        aa_int8, false, bb_int8, false, static_cast<int8_t>(1), &cc_int8,
        static_cast<int8_t>(0), false, nullptr);
  }

  auto time3 = time();
  for (int j = 0; j < 10; ++j) {
    paddle_mobile::operators::math::matmul<int8_t>(
        aa_int8, false, bb_int8, false, static_cast<int8_t>(1), &cc_int8,
        static_cast<int8_t>(0), false, nullptr);
  }
  auto time4 = time();
  std::cout << "int8_t gemm  cost :" << time_diff(time3, time4) / 10 << "ms\n";

  return 0;
}
