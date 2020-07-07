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
#include "lite/kernels/x86/reshape_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

// reshape
TEST(reshape_x86, retrive_op) {
  auto reshape = KernelRegistry::Global().Create("reshape");
  ASSERT_FALSE(reshape.empty());
  ASSERT_TRUE(reshape.front());
}

TEST(reshape_x86, init) {
  lite::kernels::x86::ReshapeCompute<float> reshape;
  ASSERT_EQ(reshape.precision(), PRECISION(kFloat));
  ASSERT_EQ(reshape.target(), TARGET(kX86));
}

TEST(reshape_x86, run_test) {
  lite::Tensor x, actual_shape;
  lite::Tensor out;
  std::vector<int64_t> x_shape({1, 2, 4, 1});
  x.Resize(lite::DDim(x_shape));
  actual_shape.Resize(lite::DDim(std::vector<int64_t>({4})));
  std::vector<int64_t> out_shape({1, 8, 1, 1});
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto actual_data = actual_shape.mutable_data<int>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }
  actual_data[0] = 1;
  actual_data[1] = 8;
  actual_data[2] = 1;
  actual_data[1] = 1;

  std::vector<int> shape({1, 8, 1, 1});

  // ReshapeCompute reshape;
  ReshapeCompute<float> reshape;
  operators::ReshapeParam param;

  param.x = &x;
  param.output = &out;
  param.shape_vct = shape;
  param.shape_tensor = &actual_shape;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  for (int i = 0; i < 2; ++i) {
    if (1 == i) param.shape_tensor = nullptr;
    reshape.SetContext(std::move(ctx));
    reshape.SetParam(param);
    reshape.Run();

    for (int j = 0; j < out.dims().production(); ++j) {
      EXPECT_NEAR(out_data[j], x_data[j], 1e-5);
    }
  }
}

// reshape2
TEST(reshape2_x86, retrive_op) {
  auto reshape2 = KernelRegistry::Global().Create("reshape2");
  ASSERT_FALSE(reshape2.empty());
  ASSERT_TRUE(reshape2.front());
}

TEST(reshape2_x86, init) {
  lite::kernels::x86::Reshape2Compute<float> reshape2;
  ASSERT_EQ(reshape2.precision(), PRECISION(kFloat));
  ASSERT_EQ(reshape2.target(), TARGET(kX86));
}

TEST(reshape2_x86, run_test) {
  lite::Tensor x, actual_shape;
  lite::Tensor out, xshape;
  std::vector<int64_t> x_shape({1, 2, 4});
  x.Resize(lite::DDim(x_shape));
  actual_shape.Resize(lite::DDim(std::vector<int64_t>({3})));
  std::vector<int64_t> out_shape({1, 4, 2});
  out.Resize(lite::DDim(out_shape));
  std::vector<int64_t> xshape_shape({1, 4, 2});
  xshape.Resize(lite::DDim(xshape_shape));

  auto x_data = x.mutable_data<float>();
  auto actual_data = actual_shape.mutable_data<int>();
  auto out_data = out.mutable_data<float>();
  auto xshape_data = xshape.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
    xshape_data[i] = static_cast<float>(i);
  }
  actual_data[0] = 1;
  actual_data[1] = 4;
  actual_data[2] = 2;

  std::vector<int> shape({1, 4, 2});

  // Reshape2Compute reshape2;
  Reshape2Compute<float> reshape2;
  operators::ReshapeParam param;

  param.x = &x;
  param.output = &out;
  param.xshape = &xshape;
  param.shape_vct = shape;
  param.shape_tensor = &actual_shape;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  for (int i = 0; i < 2; ++i) {
    if (1 == i) param.shape_tensor = nullptr;
    reshape2.SetContext(std::move(ctx));
    reshape2.SetParam(param);
    reshape2.Run();

    for (int j = 0; j < out.dims().production(); ++j) {
      EXPECT_NEAR(out_data[j], x_data[j], 1e-5);
    }
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(reshape, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(reshape2, kX86, kFloat, kNCHW, def);
