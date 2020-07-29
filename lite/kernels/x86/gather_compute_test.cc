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
#include "lite/kernels/x86/gather_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(gather_x86, retrive_op) {
  auto gather = KernelRegistry::Global().Create("gather");
  ASSERT_FALSE(gather.empty());
  int cnt = 0;
  for (auto item = gather.begin(); item != gather.end(); ++item) {
    cnt++;
    ASSERT_TRUE(*item);
  }
  ASSERT_EQ(cnt, 2);
}

TEST(gather_x86, int32_init) {
  GatherCompute<float, int32_t> gather;
  ASSERT_EQ(gather.precision(), PRECISION(kFloat));
  ASSERT_EQ(gather.target(), TARGET(kX86));
}

TEST(gather_x86, int64_init) {
  GatherCompute<float, int64_t> gather;
  ASSERT_EQ(gather.precision(), PRECISION(kFloat));
  ASSERT_EQ(gather.target(), TARGET(kX86));
}

template <typename T>
void test_case_1dims() {
  lite::Tensor x, index, out;
  std::vector<int64_t> x_shape{10};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> index_shape{3};
  index.Resize(lite::DDim(index_shape));
  std::vector<int64_t> out_shape{3};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto index_data = index.mutable_data<T>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }
  std::vector<float> index_value{1, 3, 5};
  for (int i = 0; i < index.dims().production(); ++i) {
    index_data[i] = static_cast<T>(index_value[i]);
  }

  GatherCompute<float, T> gather;
  operators::GatherParam param;

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

template <typename T>
void test_case_2dims() {
  lite::Tensor x, index, out;
  std::vector<int64_t> x_shape{10, 20};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> index_shape{3};
  index.Resize(lite::DDim(index_shape));
  std::vector<int64_t> out_shape{3, 20};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto index_data = index.mutable_data<T>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }
  std::vector<float> index_value{1, 3, 5};
  for (int i = 0; i < index.dims().production(); ++i) {
    index_data[i] = static_cast<T>(index_value[i]);
  }

  GatherCompute<float, T> gather;
  operators::GatherParam param;

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
    ref_data[i] = static_cast<float>(20 + i);
  }
  for (int i = 20; i < 40; ++i) {
    ref_data[i] = static_cast<float>(40 + i);
  }
  for (int i = 40; i < 60; ++i) {
    ref_data[i] = static_cast<float>(60 + i);
  }
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_data[i], 1e-5);
  }
}

TEST(gather_x86, run_test_1dims) {
  test_case_1dims<int32_t>();
  test_case_1dims<int64_t>();
}

TEST(gather_x86, run_test_2dims) {
  test_case_2dims<int32_t>();
  test_case_2dims<int64_t>();
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(gather, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(gather, kX86, kFloat, kNCHW, int64_in);
