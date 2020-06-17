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

#include "lite/kernels/cuda/elementwise_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include "lite/api/test_helper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

using Tensor = lite::Tensor;

static void ElementwiseAddRef(float* x, float* y, float* out, int num) {
  for (int i = 0; i < num; ++i) {
    out[i] = x[i] + y[i];
  }
}

static void ElementwiseBroadcastRef(
    float* x, float* y, float* out, int pre, int n, int post) {
  for (int i = 0; i < pre * n * post; ++i) {
    int idx = (i / post) % n;
    out[i] = x[i] + y[idx];
  }
}

TEST(elementwise_add, normal) {
  ElementwiseAddCompute elementwise_add_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::ElementwiseParam param;
  Tensor x, y, out;
  Tensor x_cpu, y_cpu, out_cpu;
  Tensor x_ref, y_ref, out_ref;

  const int n = 1;
  const int c = 3;
  const int h = 2000;
  const int w = 2000;

  x.Resize({n, c, h, w});
  y.Resize({n, c, h, w});
  out.Resize({n, c, h, w});
  x_cpu.Resize({n, c, h, w});
  y_cpu.Resize({n, c, h, w});
  out_cpu.Resize({n, c, h, w});
  x_ref.Resize({n, c, h, w});
  y_ref.Resize({n, c, h, w});
  out_ref.Resize({n, c, h, w});

  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));

  auto* x_cpu_data = x_cpu.mutable_data<float>();
  auto* y_cpu_data = y_cpu.mutable_data<float>();
  auto* out_cpu_data = out_cpu.mutable_data<float>();

  auto* x_ref_data = x_ref.mutable_data<float>();
  auto* y_ref_data = y_ref.mutable_data<float>();
  auto* out_ref_data = out_ref.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = i + 5.0;
    x_ref_data[i] = i + 5.0;
  }
  for (int i = 0; i < y_cpu.numel(); ++i) {
    y_cpu_data[i] = i - 5.0;
    y_ref_data[i] = i - 5.0;
  }

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  y.Assign<float, lite::DDim, TARGET(kCUDA)>(y_cpu_data, y_cpu.dims());

  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  elementwise_add_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  elementwise_add_kernel.SetContext(std::move(ctx));
  elementwise_add_kernel.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);
  ElementwiseAddRef(x_ref_data, y_ref_data, out_ref_data, out.numel());
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_cpu_data[i], out_ref_data[i], 1e-5);
  }
}

TEST(elementwise_add, bias) {
  ElementwiseAddCompute elementwise_add_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::ElementwiseParam param;
  Tensor x, y, out;
  Tensor x_cpu, y_cpu, out_cpu;
  Tensor x_ref, y_ref, out_ref;

  const int n = 1;
  const int c = 3;
  const int h = 2000;
  const int w = 2000;

  x.Resize({n, c, h, w});
  y.Resize({c, 1, 1});
  out.Resize({n, c, h, w});
  x_cpu.Resize({n, c, h, w});
  y_cpu.Resize({c, 1, 1});
  out_cpu.Resize({n, c, h, w});
  x_ref.Resize({n, c, h, w});
  y_ref.Resize({c, 1, 1});
  out_ref.Resize({n, c, h, w});

  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));

  auto* x_cpu_data = x_cpu.mutable_data<float>();
  auto* y_cpu_data = y_cpu.mutable_data<float>();
  auto* out_cpu_data = out_cpu.mutable_data<float>();

  auto* x_ref_data = x_ref.mutable_data<float>();
  auto* y_ref_data = y_ref.mutable_data<float>();
  auto* out_ref_data = out_ref.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = i + 5.0;
    x_ref_data[i] = i + 5.0;
  }
  for (int i = 0; i < y_cpu.numel(); ++i) {
    y_cpu_data[i] = i - 5.0;
    y_ref_data[i] = i - 5.0;
  }

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  y.Assign<float, lite::DDim, TARGET(kCUDA)>(y_cpu_data, y_cpu.dims());

  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  param.axis = -1;
  elementwise_add_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  elementwise_add_kernel.SetContext(std::move(ctx));
  elementwise_add_kernel.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);
  ElementwiseBroadcastRef(x_ref_data, y_ref_data, out_ref_data, n, c, h * w);
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_cpu_data[i], out_ref_data[i], 1e-5);
  }
}

TEST(elementwise_add_nhwc, bias) {
  ElementwiseAddComputeNHWC elementwise_add_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::ElementwiseParam param;
  Tensor x, y, out;
  Tensor x_cpu, y_cpu, out_cpu;
  Tensor x_ref, y_ref, out_ref;

  const int n = 1;
  const int c = 3;
  const int h = 2000;
  const int w = 2000;

  x.Resize({n, h, w, c});
  y.Resize({c, 1, 1});
  out.Resize({n, h, w, c});
  x_cpu.Resize({n, h, w, c});
  y_cpu.Resize({c, 1, 1});
  out_cpu.Resize({n, h, w, c});
  x_ref.Resize({n, h, w, c});
  y_ref.Resize({c, 1, 1});
  out_ref.Resize({n, h, w, c});

  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));

  auto* x_cpu_data = x_cpu.mutable_data<float>();
  auto* y_cpu_data = y_cpu.mutable_data<float>();
  auto* out_cpu_data = out_cpu.mutable_data<float>();

  auto* x_ref_data = x_ref.mutable_data<float>();
  auto* y_ref_data = y_ref.mutable_data<float>();
  auto* out_ref_data = out_ref.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = i + 5.0;
    x_ref_data[i] = i + 5.0;
  }
  for (int i = 0; i < y_cpu.numel(); ++i) {
    y_cpu_data[i] = i - 5.0;
    y_ref_data[i] = i - 5.0;
  }

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  y.Assign<float, lite::DDim, TARGET(kCUDA)>(y_cpu_data, y_cpu.dims());

  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  param.axis = -1;
  elementwise_add_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  elementwise_add_kernel.SetContext(std::move(ctx));
  elementwise_add_kernel.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);
  ElementwiseBroadcastRef(
      x_ref_data, y_ref_data, out_ref_data, n * h * w, c, 1);
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_cpu_data[i], out_ref_data[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
