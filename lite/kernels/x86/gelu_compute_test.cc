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

TEST(gelu_x86, retrive_op) {
  auto gelu = KernelRegistry::Global().Create("gelu");
  ASSERT_FALSE(gelu.empty());
  ASSERT_TRUE(gelu.front());
}

TEST(gelu_x86, init) {
  GeluCompute<float> gelu;
  ASSERT_EQ(gelu.precision(), PRECISION(kFloat));
  ASSERT_EQ(gelu.target(), TARGET(kX86));
}

TEST(gelu_x86, run_test) {
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
    x_data[i] = static_cast<float>(i * sign) * 0.8f;
  }
  // GeluCompute gelu;
  GeluCompute<float> gelu;
  operators::ActivationParam param;

  param.X = &x;
  param.Out = &out;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  gelu.SetContext(std::move(ctx));
  gelu.SetParam(param);
  gelu.Run();

  LOG(INFO) << "output: ";
  std::vector<float> ref_data{0.f,
                              -0.169484f,
                              1.512321f,
                              -0.019674f,
                              3.197801f,
                              -0.000126719f,
                              4.8f,
                              -0.f,
                              6.4000001f,
                              -0.f,
                              8.f,
                              -0.f};
  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
    EXPECT_NEAR(out_data[i], ref_data[i], 1e-5);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(gelu, kX86, kFloat, kNCHW, def);
