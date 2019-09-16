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

#include "lite/kernels/x86/squeeze_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

// squeeze
TEST(squeeze_x86, retrive_op) {
  auto squeeze =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>(
          "squeeze");
  ASSERT_FALSE(squeeze.empty());
  ASSERT_TRUE(squeeze.front());
}

TEST(squeeze_x86, init) {
  lite::kernels::x86::SqueezeCompute<float> squeeze;
  ASSERT_EQ(squeeze.precision(), PRECISION(kFloat));
  ASSERT_EQ(squeeze.target(), TARGET(kX86));
}

TEST(squeeze_x86, run_test) {
  lite::Tensor x;
  lite::Tensor out;
  std::vector<int64_t> x_shape({1, 3, 1, 5});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({3, 5});
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }

  // SqueezeCompute squeeze;
  SqueezeCompute<float> squeeze;
  operators::SqueezeParam param;

  param.X = &x;
  param.Out = &out;
  std::vector<std::vector<float>> ref_res({{3, 5}, {3, 5}});
  std::vector<std::vector<int>> axes({{0, -2}, {}});
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  for (int i = 0; i < 2; ++i) {
    param.axes = axes[i];
    squeeze.SetContext(std::move(ctx));
    squeeze.SetParam(param);
    squeeze.Run();

    for (int j = 0; j < out.dims().production(); ++j) {
      EXPECT_NEAR(out_data[j], x_data[j], 1e-5);
    }
  }
}

// squeeze2
TEST(squeeze2_x86, retrive_op) {
  auto squeeze2 =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>(
          "squeeze2");
  ASSERT_FALSE(squeeze2.empty());
  ASSERT_TRUE(squeeze2.front());
}

TEST(squeeze2_x86, init) {
  lite::kernels::x86::Squeeze2Compute<float> squeeze2;
  ASSERT_EQ(squeeze2.precision(), PRECISION(kFloat));
  ASSERT_EQ(squeeze2.target(), TARGET(kX86));
}

TEST(squeeze2_x86, run_test) {
  lite::Tensor x;
  lite::Tensor xshape;
  lite::Tensor out;
  std::vector<int64_t> x_shape({1, 3, 1, 5});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({3, 5});
  out.Resize(lite::DDim(out_shape));
  std::vector<int64_t> xshape_shape({1, 3, 1, 5});
  xshape.Resize(lite::DDim(xshape_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();
  auto xshape_data = xshape.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
    xshape_data[i] = static_cast<float>(i);
  }

  // Squeeze2Compute squeeze2;
  Squeeze2Compute<float> squeeze2;
  operators::SqueezeParam param;

  param.X = &x;
  param.Out = &out;
  param.XShape = &xshape;
  std::vector<std::vector<float>> ref_res({{3, 5}, {3, 5}});
  std::vector<std::vector<int>> axes({{0, -2}, {}});
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  for (int i = 0; i < 2; ++i) {
    param.axes = axes[i];
    squeeze2.SetContext(std::move(ctx));
    squeeze2.SetParam(param);
    squeeze2.Run();

    for (int j = 0; j < out.dims().production(); ++j) {
      EXPECT_NEAR(out_data[j], x_data[j], 1e-5);
    }
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(squeeze, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(squeeze2, kX86, kFloat, kNCHW, def);
