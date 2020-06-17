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
#include "lite/kernels/x86/activation_compute.cc"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(tanh_x86, retrive_op) {
  auto tanh = KernelRegistry::Global().Create("tanh");
  ASSERT_FALSE(tanh.empty());
  ASSERT_TRUE(tanh.front());
}

TEST(tanh_x86, init) {
  TanhCompute<float> tanh;
  ASSERT_EQ(tanh.precision(), PRECISION(kFloat));
  ASSERT_EQ(tanh.target(), TARGET(kX86));
}

TEST(tanh_x86, run_test) {
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
    x_data[i] = static_cast<float>(i * sign) * 0.08f;
  }
  // TanhCompute tanh;
  TanhCompute<float> tanh;
  operators::ActivationParam param;

  param.X = &x;
  param.Out = &out;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  tanh.SetContext(std::move(ctx));
  tanh.SetParam(param);
  tanh.Run();

  LOG(INFO) << "output: ";
  std::vector<float> ref_data{0.f,
                              -0.079829f,
                              0.158648f,
                              -0.235495f,
                              0.309506f,
                              -0.379949f,
                              0.446243f,
                              -0.507977f,
                              0.564899f,
                              -0.616909f,
                              0.664036f,
                              -0.706419f};
  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
    EXPECT_NEAR(out_data[i], ref_data[i], 1e-5);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(tanh, kX86, kFloat, kNCHW, def);
