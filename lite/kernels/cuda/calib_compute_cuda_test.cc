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

#include "lite/kernels/cuda/calib_compute.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

static void int8_to_fp32_basic(const int8_t* din,
                               float* dout,
                               const float scale,
                               int num) {
  for (int j = 0; j < num; ++j) {
    dout[j] = din[j] * scale;
  }
}

static void fp32_to_int8_basic(const float* din,
                               int8_t* dout,
                               const float scale,
                               int num) {
  for (int j = 0; j < num; ++j) {
    auto v = din[j] / scale;
    v = std::max(v, static_cast<float>(INT8_MIN));
    v = std::min(v, static_cast<float>(INT8_MAX));
    v = roundf(v);
    dout[j] = static_cast<int8_t>(v);
  }
}

void calib_ref(const operators::CalibParam& param, bool to_float = true) {
  auto scale = param.scale;
  if (to_float) {
    const auto* din = param.input->data<int8_t>();
    auto* dout = param.output->mutable_data<float>();
    int8_to_fp32_basic(din, dout, scale, param.input->numel());
  } else {
    const auto* din = param.input->data<float>();
    auto* dout = param.output->mutable_data<int8_t>();
    fp32_to_int8_basic(din, dout, scale, param.input->numel());
  }
}

TEST(calib_cuda, int8_to_fp32) {
  CalibComputeInt8ToFp32 calib;
  const int n = 64, c = 32, h = 18, w = 18;
  Tensor x;
  Tensor x_cpu;
  Tensor output;
  Tensor output_cpu;
  // set the dims of input, output tensors
  x.Resize({n, c, h, w});
  x_cpu.Resize({n, c, h, w});
  output.Resize({n, c, h, w});
  output_cpu.Resize({n, c, h, w});
  // initialize the data of input tensors
  auto* x_cpu_data = x_cpu.mutable_data<int8_t>();
  for (int i = 0; i < x.dims().production(); i++) {
    float sign = i % 3 == 0 ? -1.0f : 1.0f;
    x_cpu_data[i] = static_cast<int8_t>(sign * (i % 127));
  }
  x.Assign<int8_t, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  // prepare kernel params and run
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);
  calib.SetContext(std::move(ctx));

  operators::CalibParam param;
  param.scale = 0.013f;
  param.input = &x;
  param.output = &output;
  calib.SetParam(param);
  calib.Launch();
  cudaDeviceSynchronize();
  // invoking ref implementation and compare results
  param.input = &x_cpu;
  param.output = &output_cpu;
  calib_ref(param);
  auto* output_data = output.mutable_data<float>();
  std::unique_ptr<float[]> output_gpu_copy(new float[output.numel()]);
  CopySync<TARGET(kCUDA)>(output_gpu_copy.get(),
                          output_data,
                          sizeof(float) * output.numel(),
                          IoDirection::DtoH);
  const auto* output_cpu_data = output_cpu.data<float>();
  for (int i = 0; i < output.dims().production(); i++) {
    EXPECT_NEAR(output_gpu_copy[i], output_cpu_data[i], 1e-5);
  }
}

TEST(calib_cuda, fp32_to_int8) {
  CalibComputeFp32ToInt8 calib;
  const int n = 64, c = 32, h = 18, w = 18;
  Tensor x;
  Tensor x_cpu;
  Tensor output;
  Tensor output_cpu;
  // set the dims of input, output tensors
  x.Resize({n, c, h, w});
  x_cpu.Resize({n, c, h, w});
  output.Resize({n, c, h, w});
  output_cpu.Resize({n, c, h, w});
  // initialize the data of input tensors
  auto* x_cpu_data = x_cpu.mutable_data<float>();
  for (int i = 0; i < x.dims().production(); i++) {
    float sign = i % 3 == 0 ? -1.0f : 1.0f;
    x_cpu_data[i] = sign * (i % 127) * 0.013f;
  }
  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  // prepare kernel params and run
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);
  calib.SetContext(std::move(ctx));

  operators::CalibParam param;
  param.scale = 0.013f;
  param.input = &x;
  param.output = &output;
  calib.SetParam(param);
  calib.Launch();
  cudaDeviceSynchronize();
  // invoking ref implementation and compare results
  param.input = &x_cpu;
  param.output = &output_cpu;
  calib_ref(param, false);
  auto* output_data = output.mutable_data<int8_t>();
  std::unique_ptr<int8_t[]> output_gpu_copy(new int8_t[output.numel()]);
  CopySync<TARGET(kCUDA)>(output_gpu_copy.get(),
                          output_data,
                          sizeof(int8_t) * output.numel(),
                          IoDirection::DtoH);
  const auto* output_cpu_data = output_cpu.data<int8_t>();
  for (int i = 0; i < output.dims().production(); i++) {
    EXPECT_EQ(output_gpu_copy[i], output_cpu_data[i]);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
