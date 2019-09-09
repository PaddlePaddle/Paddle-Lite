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

#include "lite/kernels/x86/activation_compute.h"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(relu_x86, retrive_op) {
  auto relu =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>("relu");
  ASSERT_FALSE(relu.empty());
  ASSERT_TRUE(relu.front());
}

TEST(relu_x86, init) {
  ReluCompute<float> relu;
  ASSERT_EQ(relu.precision(), PRECISION(kFloat));
  ASSERT_EQ(relu.target(), TARGET(kX86));
}

TEST(relu_x86, run_test) {
  lite::Tensor x, out;
  constexpr int batch_size = 1;
  std::vector<int64_t> x_shape{batch_size, 3, 2, 2};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape{batch_size, 3, 2, 2};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    int sign = i % 2 == 0 ? 1 : -1;
    x_data[i] = static_cast<float>(i * sign);
  }
  // ReluCompute relu;
  ReluCompute<float> relu;
  operators::ActivationParam param;

  param.X = &x;
  param.Out = &out;

  relu.SetParam(param);
  relu.Run();

  LOG(INFO) << "output: ";
  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
    int sign = i % 2 == 0 ? 1 : 0;
    ASSERT_EQ(out_data[i], i * sign);
  }
}

TEST(relu_grad_x86, retrive_op) {
  auto relu_grad =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>("relu_grad");
  ASSERT_FALSE(relu_grad.empty());
  ASSERT_TRUE(relu_grad.front());
}

TEST(relu_grad_x86, init) {
  ReluCompute<float> relu_grad;
  ASSERT_EQ(relu_grad.precision(), PRECISION(kFloat));
  ASSERT_EQ(relu_grad.target(), TARGET(kX86));
}

TEST(relu_grad_x86, run_test) {
  lite::Tensor x, out, x_grad, out_grad;
  constexpr int batch_size = 1;
  std::vector<int64_t> x_shape{batch_size, 3, 2, 2};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape{batch_size, 3, 2, 2};
  out.Resize(lite::DDim(out_shape));
  std::vector<int64_t> x_grad_shape{batch_size, 3, 2, 2};
  x_grad.Resize(lite::DDim(x_grad_shape));
  std::vector<int64_t> out_grad_shape{batch_size, 3, 2, 2};
  out_grad.Resize(lite::DDim(out_grad_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();
  auto x_grad_data = x_grad.mutable_data<float>();
  auto out_grad_data = out_grad.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    int sign = i % 2 == 0 ? 1 : -1;
    x_data[i] = static_cast<float>(i * sign);
  }
  for (int64_t i = 0; i < out.dims().production(); i++) {
    int sign = i % 2 == 0 ? 1 : 0;
    out_data[i] = static_cast<float>(i * sign);
    out_grad_data[i] = 1;
  }
  // ReluGradCompute relu_grad;
  ReluGradCompute<float> relu_grad;
  operators::ActivationGradParam param;

  param.X = &x;
  param.Out = &out;
  param.Out_grad = &out_grad;
  param.X_grad = &x_grad;

  relu_grad.SetParam(param);
  relu_grad.Run();

  LOG(INFO) << "output: ";
  for (int i = 0; i < x_grad.dims().production(); i++) {
    LOG(INFO) << x_grad_data[i];
    int sign = out_data[i] > 0 ? 1 : 0;
    ASSERT_EQ(x_grad_data[i], sign);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(relu, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu_grad, kX86, kFloat, kNCHW, def);
