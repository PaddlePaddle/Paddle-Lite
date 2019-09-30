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

#include "lite/kernels/x86/slice_compute.h"
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(slice_x86, retrive_op) {
  auto slice =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>("slice");
  ASSERT_FALSE(slice.empty());
  ASSERT_TRUE(slice.front());
}

TEST(slice_x86, init) {
  lite::kernels::x86::SliceCompute<float> slice;
  ASSERT_EQ(slice.precision(), PRECISION(kFloat));
  ASSERT_EQ(slice.target(), TARGET(kX86));
}

void test_case1(lite::Tensor x, lite::Tensor out) {
  std::vector<int64_t> x_shape({3});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({3});
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }

  std::vector<int> starts({-3});
  std::vector<int> ends({3});
  std::vector<int> axes({0});

  // SliceCompute slice;
  SliceCompute<float> slice;
  operators::SliceParam param;

  param.X = &x;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
  }
}

void test_case2(lite::Tensor x, lite::Tensor out) {
  std::vector<int64_t> x_shape({3, 4});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({3, 4});
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }

  std::vector<int> starts({-3, 0});
  std::vector<int> ends({3, 100});
  std::vector<int> axes({0, 1});

  // SliceCompute slice;
  SliceCompute<float> slice;
  operators::SliceParam param;

  param.X = &x;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
  }
}

void test_case3(lite::Tensor x, lite::Tensor out) {
  std::vector<int64_t> x_shape({3, 4, 5});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({3, 4, 2});
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }

  std::vector<int> starts({-3, 0, 2});
  std::vector<int> ends({3, 100, -1});
  std::vector<int> axes({0, 1, 2});

  // SliceCompute slice;
  SliceCompute<float> slice;
  operators::SliceParam param;

  param.X = &x;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
  }
}
void test_case4(lite::Tensor x, lite::Tensor out) {
  std::vector<int64_t> x_shape({3, 4, 5, 6});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({3, 4, 2, 6});
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }

  std::vector<int> starts({-3, 0, 2});
  std::vector<int> ends({3, 100, -1});
  std::vector<int> axes({0, 1, 2});

  // SliceCompute slice;
  SliceCompute<float> slice;
  operators::SliceParam param;

  param.X = &x;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
  }
}

void test_case5(lite::Tensor x, lite::Tensor out) {
  std::vector<int64_t> x_shape({3, 4, 5, 6, 3});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({3, 4, 2, 6, 3});
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }

  std::vector<int> starts({-3, 0, 2});
  std::vector<int> ends({3, 100, -1});
  std::vector<int> axes({0, 1, 2});

  // SliceCompute slice;
  SliceCompute<float> slice;
  operators::SliceParam param;

  param.X = &x;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
  }
}
void test_case6(lite::Tensor x, lite::Tensor out) {
  std::vector<int64_t> x_shape({3, 4, 5, 6, 5, 2});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({3, 4, 2, 6, 5, 2});
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }

  std::vector<int> starts({-3, 0, 2});
  std::vector<int> ends({3, 100, -1});
  std::vector<int> axes({0, 1, 2});

  // SliceCompute slice;
  SliceCompute<float> slice;
  operators::SliceParam param;

  param.X = &x;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
  }
}

TEST(slice_x86, run_test) {
  lite::Tensor x;
  lite::Tensor out;

  test_case1(x, out);
  test_case2(x, out);
  test_case3(x, out);
  test_case4(x, out);
  test_case5(x, out);
  test_case6(x, out);
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(slice, kX86, kFloat, kNCHW, def);
