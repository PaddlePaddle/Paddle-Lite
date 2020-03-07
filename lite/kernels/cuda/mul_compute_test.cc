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

#include "lite/kernels/cuda/mul_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include "lite/backends/cuda/blas.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

TEST(mul_compute, normal) {
  MulCompute mul_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();
  context.InitOnce();

  Tensor x, y, out, x_cpu, y_cpu, out_cpu;
  int x_h = 2, x_w_y_h = 3, y_w = 4;
  out.Resize({x_h, y_w});
  x_cpu.Resize({x_h, x_w_y_h});
  y_cpu.Resize({x_w_y_h, y_w});
  out_cpu.Resize({x_h, y_w});

  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));
  float* x_cpu_data = x_cpu.mutable_data<float>();
  float* y_cpu_data = y_cpu.mutable_data<float>();
  float* out_cpu_data = out_cpu.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); i++) {
    x_cpu_data[i] = i + 1.0;
  }
  for (int i = 0; i < y_cpu.numel(); i++) {
    y_cpu_data[i] = i + 1.0;
  }

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  y.Assign<float, lite::DDim, TARGET(kCUDA)>(y_cpu_data, y_cpu.dims());

  operators::MulParam param;
  param.x = &x;
  param.y = &y;
  param.output = &out;
  mul_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  mul_kernel.SetContext(std::move(ctx));
  mul_kernel.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);
  for (int i = 0; i < out_cpu.numel(); i++) {
    LOG(INFO) << out_cpu_data[i];
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
