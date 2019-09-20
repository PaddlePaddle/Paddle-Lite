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

#include "lite/kernels/cuda/leaky_relu_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

TEST(leaky_relu, normal) {
  LeakyReluCompute leaky_relu_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::ActivationParam param;

  Tensor x, y, x_cpu, y_cpu;
  int h = 3, w = 3;
  y.Resize({h, w});
  x_cpu.Resize({h, w});
  y_cpu.Resize({h, w});

  auto* y_data = y.mutable_data<float>(TARGET(kCUDA));
  float* x_cpu_data = x_cpu.mutable_data<float>();
  float* y_cpu_data = x_cpu.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); i++) {
    x_cpu_data[i] = i - 5.0;
  }

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());

  param.X = &x;
  param.Out = &y;
  param.Leaky_relu_alpha = 10.0;
  leaky_relu_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  leaky_relu_kernel.SetContext(std::move(ctx));
  leaky_relu_kernel.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      y_cpu_data, y_data, sizeof(float) * y.numel(), IoDirection::DtoH);
  for (int i = 0; i < y.numel(); i++) {
    LOG(INFO) << y_cpu_data[i];
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
