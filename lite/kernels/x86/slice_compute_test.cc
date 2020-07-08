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
#include "lite/kernels/x86/slice_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

static void slice_ref(const float* input,
                      std::vector<int64_t> in_dims,
                      std::vector<int> axes,
                      std::vector<int> starts,
                      std::vector<int> ends,
                      float* out) {
  auto out_dims = in_dims;
  std::vector<int> real_starts(in_dims.size(), 0);
  std::vector<int> real_ends(in_dims.size(), 0);
  std::vector<int> real_step(in_dims.size(), 0);
  for (size_t i = 0; i < in_dims.size(); i++) {
    real_ends[i] = in_dims[i];
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int dim_value = in_dims[axes[i]];
    if (dim_value > 0) {
      int start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
      int end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim_value);
      out_dims[axes[i]] = end - start;
      real_starts[axes[i]] = start;
      real_ends[axes[i]] = end;
    }
  }
  const int LEN = in_dims.size();
  std::vector<int> dst_step(LEN);
  for (size_t i = 0; i < in_dims.size(); ++i) {
    dst_step[i] = 1;
  }
  std::vector<int> src_step(LEN);
  for (size_t i = 0; i < in_dims.size(); ++i) {
    src_step[i] = 1;
  }
  int out_num = out_dims[in_dims.size() - 1];
  for (int i = in_dims.size() - 2; i >= 0; i--) {
    dst_step[i] = out_dims[i + 1] * dst_step[i + 1];
    src_step[i] = in_dims[i + 1] * src_step[i + 1];
    out_num *= out_dims[i];
  }

  for (int dst_id = 0; dst_id < out_num; dst_id++) {
    int src_id = 0;
    int index_id = dst_id;
    for (size_t j = 0; j < out_dims.size(); j++) {
      int cur_id = index_id / dst_step[j];
      index_id = index_id % dst_step[j];
      src_id += (cur_id + real_starts[j]) * src_step[j];
    }
    out[dst_id] = input[src_id];
  }
}

TEST(slice_x86, retrive_op) {
  auto slice = KernelRegistry::Global().Create("slice");
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
  param.axes = axes;
  param.starts = starts;
  param.ends = ends;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  std::vector<float> out_ref(out.numel(), 0);
  slice_ref(x_data, x_shape, axes, starts, ends, out_ref.data());

  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_ref[i], out_data[i], 1e-4);
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
  param.axes = axes;
  param.starts = starts;
  param.ends = ends;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  std::vector<float> out_ref(out.numel(), 0);
  slice_ref(x_data, x_shape, axes, starts, ends, out_ref.data());

  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_ref[i], out_data[i], 1e-4);
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
  param.axes = axes;
  param.starts = starts;
  param.ends = ends;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  std::vector<float> out_ref(out.numel(), 0);
  slice_ref(x_data, x_shape, axes, starts, ends, out_ref.data());

  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_ref[i], out_data[i], 1e-4);
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
  param.axes = axes;
  param.starts = starts;
  param.ends = ends;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  std::vector<float> out_ref(out.numel(), 0);
  slice_ref(x_data, x_shape, axes, starts, ends, out_ref.data());

  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_ref[i], out_data[i], 1e-4);
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
  param.axes = axes;
  param.starts = starts;
  param.ends = ends;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  std::vector<float> out_ref(out.numel(), 0);
  slice_ref(x_data, x_shape, axes, starts, ends, out_ref.data());

  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_ref[i], out_data[i], 1e-4);
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
  param.axes = axes;
  param.starts = starts;
  param.ends = ends;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  std::vector<float> out_ref(out.numel(), 0);
  slice_ref(x_data, x_shape, axes, starts, ends, out_ref.data());

  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_ref[i], out_data[i], 1e-4);
  }
}

void test_tensor_case1(lite::Tensor x, lite::Tensor out) {
  std::vector<int64_t> x_shape({10});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({5});
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }

  std::vector<int> starts({3});
  std::vector<int> ends({8});
  std::vector<int> axes({0});

  // SliceCompute slice;
  SliceCompute<float> slice;
  operators::SliceParam param;

  param.X = &x;
  param.Out = &out;
  param.axes = axes;
  lite::Tensor starts_tensor, ends_tensor;
  starts_tensor.Resize(DDim({1}));
  ends_tensor.Resize(DDim({1}));
  starts_tensor.mutable_data<int>()[0] = starts[0];
  ends_tensor.mutable_data<int>()[0] = ends[0];
  param.StartsTensor = &starts_tensor;
  param.EndsTensor = &ends_tensor;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  std::vector<float> out_ref(out.numel(), 0);
  slice_ref(x_data, x_shape, axes, starts, ends, out_ref.data());

  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_ref[i], out_data[i], 1e-4);
  }
}

void test_tensor_case3(lite::Tensor x, lite::Tensor out) {
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
  param.axes = axes;
  lite::Tensor starts_tensor, ends_tensor;
  starts_tensor.Resize(DDim({3}));
  ends_tensor.Resize(DDim({3}));
  for (size_t i = 0; i < starts.size(); ++i) {
    starts_tensor.mutable_data<int>()[i] = starts[i];
    ends_tensor.mutable_data<int>()[i] = ends[i];
  }
  param.StartsTensor = &starts_tensor;
  param.EndsTensor = &ends_tensor;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  std::vector<float> out_ref(out.numel(), 0);
  slice_ref(x_data, x_shape, axes, starts, ends, out_ref.data());

  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_ref[i], out_data[i], 1e-4);
  }
}

void test_tensor_list_case1(lite::Tensor x, lite::Tensor out) {
  std::vector<int64_t> x_shape({10});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({5});
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }

  std::vector<int> starts({3});
  std::vector<int> ends({8});
  std::vector<int> axes({0});

  // SliceCompute slice;
  SliceCompute<float> slice;
  operators::SliceParam param;

  param.X = &x;
  param.Out = &out;
  param.axes = axes;
  param.StartsTensorList.clear();
  param.EndsTensorList.clear();
  lite::Tensor starts_tensor, ends_tensor;
  for (int i = 0; i < 1; ++i) {
    starts_tensor.Resize(DDim({1}));
    ends_tensor.Resize(DDim({1}));
    starts_tensor.mutable_data<int>()[0] = starts[0];
    ends_tensor.mutable_data<int>()[0] = ends[0];
    param.StartsTensorList.push_back(&starts_tensor);
    param.EndsTensorList.push_back(&ends_tensor);
  }

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  std::vector<float> out_ref(out.numel(), 0);
  slice_ref(x_data, x_shape, axes, starts, ends, out_ref.data());

  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_ref[i], out_data[i], 1e-4);
  }
}

void test_tensor_list_case3(lite::Tensor x, lite::Tensor out) {
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
  param.axes = axes;
  param.StartsTensorList.clear();
  param.EndsTensorList.clear();
  lite::Tensor starts_tensor0, ends_tensor0;
  lite::Tensor starts_tensor1, ends_tensor1;
  lite::Tensor starts_tensor2, ends_tensor2;
  starts_tensor0.Resize(DDim({1}));
  starts_tensor1.Resize(DDim({1}));
  starts_tensor2.Resize(DDim({1}));
  ends_tensor0.Resize(DDim({1}));
  ends_tensor1.Resize(DDim({1}));
  ends_tensor2.Resize(DDim({1}));
  starts_tensor0.mutable_data<int>()[0] = starts[0];
  starts_tensor1.mutable_data<int>()[0] = starts[1];
  starts_tensor2.mutable_data<int>()[0] = starts[2];
  ends_tensor0.mutable_data<int>()[0] = ends[0];
  ends_tensor1.mutable_data<int>()[0] = ends[1];
  ends_tensor2.mutable_data<int>()[0] = ends[2];
  param.StartsTensorList.emplace_back(&starts_tensor0);
  param.StartsTensorList.emplace_back(&starts_tensor1);
  param.StartsTensorList.emplace_back(&starts_tensor2);
  param.EndsTensorList.emplace_back(&ends_tensor0);
  param.EndsTensorList.emplace_back(&ends_tensor1);
  param.EndsTensorList.emplace_back(&ends_tensor2);

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  slice.SetContext(std::move(ctx));
  slice.SetParam(param);
  slice.Run();

  std::vector<float> out_ref(out.numel(), 0);
  slice_ref(x_data, x_shape, axes, starts, ends, out_ref.data());

  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_ref[i], out_data[i], 1e-4);
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

TEST(slice_x86, test_tensor) {
  lite::Tensor x;
  lite::Tensor out;

  test_tensor_case1(x, out);
  test_tensor_case3(x, out);
}

TEST(slice_x86, test_tensor_list) {
  lite::Tensor x;
  lite::Tensor out;

  test_tensor_list_case1(x, out);
  test_tensor_list_case3(x, out);
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(slice, kX86, kFloat, kNCHW, def);
