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

#include <memory>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/x86/transpose_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

// transpose
TEST(transpose_x86, retrive_op) {
  auto transpose = KernelRegistry::Global().Create("transpose");
  ASSERT_FALSE(transpose.empty());
  ASSERT_TRUE(transpose.front());
}

TEST(transpose_x86, init) {
  lite::kernels::x86::TransposeCompute<float> transpose;
  ASSERT_EQ(transpose.precision(), PRECISION(kFloat));
  ASSERT_EQ(transpose.target(), TARGET(kX86));
}

TEST(transpose_x86, run_test) {
  lite::Tensor x;
  lite::Tensor out;
  std::vector<int64_t> x_shape({3, 4, 5});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({3, 5, 4});
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }

  // TransposeCompute transpose;
  TransposeCompute<float> transpose;
  operators::TransposeParam param;

  param.x = &x;
  param.output = &out;
  std::vector<int> axis({0, 2, 1});
  param.axis = axis;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  transpose.SetContext(std::move(ctx));
  transpose.SetParam(param);
  transpose.Run();

  for (int j = 0; j < out.dims().production(); ++j) {
    // EXPECT_NEAR(out_data[j], x_data[j], 1e-5);
    LOG(INFO) << out_data[j];
  }
}

// transpose2
TEST(transpose2_x86, retrive_op) {
  auto transpose2 = KernelRegistry::Global().Create("transpose2");
  ASSERT_FALSE(transpose2.empty());
  ASSERT_TRUE(transpose2.front());
}

TEST(transpose2_x86, init) {
  lite::kernels::x86::Transpose2Compute<float> transpose2;
  ASSERT_EQ(transpose2.precision(), PRECISION(kFloat));
  ASSERT_EQ(transpose2.target(), TARGET(kX86));
}

TEST(transpose2_x86, run_test) {
  lite::Tensor x;
  lite::Tensor xshape;
  lite::Tensor out;
  std::vector<int64_t> x_shape({3, 4, 5});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({3, 5, 4});
  out.Resize(lite::DDim(out_shape));
  std::vector<int64_t> xshape_shape({3, 4, 5});
  xshape.Resize(lite::DDim(xshape_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();
  auto xshape_data = xshape.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
    xshape_data[i] = static_cast<float>(i);
  }

  // Transpose2Compute transpose2;
  Transpose2Compute<float> transpose2;
  operators::TransposeParam param;

  param.x = &x;
  param.output = &out;
  param.xshape = &xshape;
  std::vector<int> axis({0, 2, 1});
  param.axis = axis;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  transpose2.SetContext(std::move(ctx));
  transpose2.SetParam(param);
  transpose2.Run();

  for (int j = 0; j < out.dims().production(); ++j) {
    LOG(INFO) << out_data[j];
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(transpose, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(transpose2, kX86, kFloat, kNCHW, def);
