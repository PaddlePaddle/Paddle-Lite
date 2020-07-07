// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/x86/matmul_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(matmul_x86, retrive_op) {
  auto matmul = KernelRegistry::Global().Create("matmul");
  ASSERT_FALSE(matmul.empty());
  ASSERT_TRUE(matmul.front());
}

TEST(matmul_x86, init) {
  lite::kernels::x86::MatMulCompute<float> matmul;
  ASSERT_EQ(matmul.precision(), PRECISION(kFloat));
  ASSERT_EQ(matmul.target(), TARGET(kX86));
}

TEST(matmul_x86, run_test) {
  lite::Tensor x, y, out;
  constexpr int batch_size = 1;
  std::vector<int64_t> x_shape{batch_size, 3, 2};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> y_shape{2, 4};
  y.Resize(lite::DDim(y_shape));
  std::vector<int64_t> out_shape{batch_size, 3, 4};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto y_data = y.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < y.dims().production(); i++) {
    y_data[i] = static_cast<float>(i);
  }
  // MatMulCompute matmul;
  MatMulCompute<float> matmul;
  operators::MatMulParam param;

  param.X = &x;
  param.Y = &y;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  matmul.SetContext(std::move(ctx));
  matmul.SetParam(param);
  matmul.Run();

  std::vector<float> ref_result = {4, 5, 6, 7, 12, 17, 22, 27, 20, 29, 38, 47};

  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_result[i], 1e-3);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(matmul, kX86, kFloat, kNCHW, def);
