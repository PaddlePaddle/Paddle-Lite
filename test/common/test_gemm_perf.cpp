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
#include "operators/math/gemm.h"
#include "operators/math/math_function.h"

#define a(i, j) a[(i)*lda + (j)]
#define b(i, j) b[(i)*ldb + (j)]
#define c1(i, j) c1[(i)*ldc + (j)]

#define m 1024
#define n 1024
#define k 1024

int main() {
  Tensor aa, bb, cc, scale, bias;
  auto aaptr = aa.mutable_data<float>({m, k});
  auto bbptr = bb.mutable_data<float>({k, n});
  auto ccptr = cc.mutable_data<float>({m, n});
  auto scaleptr = scale.mutable_data<float>({m});
  auto biasptr = bias.mutable_data<float>({m});

  for (int i = 0; i < m * k; ++i) {
    aaptr[i] = 2;
  }
  for (int i = 0; i < k * n; ++i) {
    bbptr[i] = 2;
  }
  for (int i = 0; i < m * n; ++i) {
    ccptr[i] = 2;
  }
  for (int i = 0; i < m; ++i) {
    scaleptr[i] = 1;
    biasptr[i] = 0;
  }

  auto time1 = time();
  for (int j = 0; j < 10; ++j) {
    paddle_mobile::operators::math::matmul<float>(
        aa, false, bb, false, static_cast<float>(1), &cc, static_cast<float>(0),
        false, biasptr);

    //    paddle_mobile::operators::math::matmulWithBn<float>(
    //        aa, false, bb, false, static_cast<float>(1), &cc,
    //        static_cast<float>(0), true, &scale, &bias, 0);
  }
  auto time2 = time();
  std::cout << "gemm  cost :" << time_diff(time1, time2) / 10 << "ms\n";

  return 0;
}
