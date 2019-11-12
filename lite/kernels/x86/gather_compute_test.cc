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

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(gather_x86, retrive_op) {
  auto gather =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>(
          "gather");
  ASSERT_FALSE(gather.empty());
  ASSERT_TRUE(gather.front());
}

TEST(gather_x86, init) {
  GatherCompute<float> gather;
  ASSERT_EQ(gather.precision(), PRECISION(kFloat));
  ASSERT_EQ(gather.target(), TARGET(kX86));
}

void test_case_int32() {
  lite::Tensor x, index, out;
  std::vector<int64_t> x_shape{10};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> index_shape{3};
  index.Resize(lite::DDim(index_shape));
  std::vector<int64_t> out_shape{3};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto index_data = index.mutable_data<int32_t>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }
  std::vector<int32_t> index_value{1, 3, 5};
  for (int64_t i = 0; i < index.dims().production(); ++i) {
    index_data = index_value[i];
  }

  GatherCompute<float> gather;
  operators::ActivationParam param;

  param.X = &x;
  param.Index = &index;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  gather.SetContext(std::move(ctx));
  gather.SetParam(param);
  gather.Run();

  std::vector<float> ref_data{1, 3, 5};
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_data[i], 1e-5);
  }
}

void test_case_int64() {
  lite::Tensor x, index, out;
  std::vector<int64_t> x_shape{10};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> index_shape{3};
  index.Resize(lite::DDim(index_shape));
  std::vector<int64_t> out_shape{3};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto index_data = index.mutable_data<int64_t>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }
  std::vector<int64_t> index_value{1, 3, 5};
  for (int64_t i = 0; i < index.dims().production(); ++i) {
    index_data = index_value[i];
  }

  GatherCompute<float> gather;
  operators::ActivationParam param;

  param.X = &x;
  param.Index = &index;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  gather.SetContext(std::move(ctx));
  gather.SetParam(param);
  gather.Run();

  std::vector<float> ref_data{1., 3., 5.};
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_data[i], 1e-5);
  }
}

void test_case_2dims_int32() {
  lite::Tensor x, index, out;
  std::vector<int64_t> x_shape{10, 20};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> index_shape{3};
  index.Resize(lite::DDim(index_shape));
  std::vector<int64_t> out_shape{3};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto index_data = index.mutable_data<int32_t>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }
  std::vector<int32_t> index_value{1, 3, 5};
  for (int64_t i = 0; i < index.dims().production(); ++i) {
    index_data = index_value[i];
  }

  GatherCompute<float> gather;
  operators::ActivationParam param;

  param.X = &x;
  param.Index = &index;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  gather.SetContext(std::move(ctx));
  gather.SetParam(param);
  gather.Run();

  std::vector<float> ref_data(60);
  for (int i = 0; i < 20; ++i) {
    ref_data[i] = 20 + i;
  }
  for (int i = 20; i < 40; ++i) {
    ref_data[i] = 60 + i;
  }
  for (int i = 40; i < 60; ++i) {
    ref_data[i] = 100 + i;
  }
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_data[i], 1e-5);
  }
}

void test_case_2dims_int64() {
  lite::Tensor x, index, out;
  std::vector<int64_t> x_shape{10, 20};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> index_shape{3};
  index.Resize(lite::DDim(index_shape));
  std::vector<int64_t> out_shape{3};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto index_data = index.mutable_data<int64_t>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }
  std::vector<int64_t> index_value{1, 3, 5};
  for (int64_t i = 0; i < index.dims().production(); ++i) {
    index_data = index_value[i];
  }

  GatherCompute<float> gather;
  operators::ActivationParam param;

  param.X = &x;
  param.Index = &index;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  gather.SetContext(std::move(ctx));
  gather.SetParam(param);
  gather.Run();

  std::vector<float> ref_data(60);
  for (int i = 0; i < 20; ++i) {
    ref_data[i] = 20 + i;
  }
  for (int i = 20; i < 40; ++i) {
    ref_data[i] = 60 + i;
  }
  for (int i = 40; i < 60; ++i) {
    ref_data[i] = 100 + i;
  }
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_data[i], 1e-5);
  }
}

TEST(gather_x86, run_test_int32) { test_case_int32(); }

TEST(gather_x86, run_test_int64) { test_case_int64(); }

TEST(gather_x86, run_test_dims_size_two_int32) { test_case_2dims_int32(); }

TEST(gather_x86, run_test_dims_size_two_int64) { test_case_2dims_int64(); }

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(gather, kX86, kFloat, kNCHW, def);
