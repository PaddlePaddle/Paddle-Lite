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
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/x86/activation_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(leaky_relu_x86, retrive_op) {
  auto leaky_relu = KernelRegistry::Global().Create("leaky_relu");
  ASSERT_FALSE(leaky_relu.empty());
  ASSERT_TRUE(leaky_relu.front());
}

TEST(leaky_relu_x86, init) {
  LeakyReluCompute<float> leaky_relu;
  ASSERT_EQ(leaky_relu.precision(), PRECISION(kFloat));
  ASSERT_EQ(leaky_relu.target(), TARGET(kX86));
}

TEST(leaky_relu_x86, run_test) {
  lite::Tensor x, out;
  constexpr int batch_size = 1;
  std::vector<int64_t> x_shape{batch_size, 3, 2, 2};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape{batch_size, 3, 2, 2};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = static_cast<float>(i) / 12.0 - 0.5;
  }
  LeakyReluCompute<float> leaky_relu;
  operators::ActivationParam param;

  param.X = &x;
  param.Out = &out;
  param.Leaky_relu_alpha = 0.05;

  leaky_relu.SetParam(param);
  leaky_relu.Run();

  std::vector<float> ref_data({-0.025f,
                               -0.02083333f,
                               -0.01666667f,
                               -0.0125f,
                               -0.00833333f,
                               -0.00416667f,
                               0.f,
                               0.08333334f,
                               0.16666667f,
                               0.25f,
                               0.33333334f,
                               0.41666666f});
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_data[i], 1e-05);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(leaky_relu, kX86, kFloat, kNCHW, def);
