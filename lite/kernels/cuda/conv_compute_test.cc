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

#include "lite/kernels/cuda/conv_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

float random(float low, float high) {
  static std::mt19937 mt(100);
  std::uniform_real_distribution<double> dist(low, high);
  return dist(mt);
}

TEST(conv_compute, fp32) {
  ConvCompute conv_fp32;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::ActivationParam act_param;
  act_param.has_active = true;
  // act_param.active_type = core::ActiveType::Active_relu;
  act_param.active_type = lite_api::ActivationType::kLeakyRelu;
  act_param.Leaky_relu_alpha = 0.1;
  operators::ConvParam param;
  param.activation_param = act_param;
  param.paddings = {1, 1};
  param.groups = 1;

  Tensor x, filter, bias, y, x_cpu, filter_cpu, bias_cpu, y_cpu;
  int n = 1, c = 1, h = 3, w = 3;
  int c_o = 1, h_o = 3, w_o = 3;
  y.Resize({n, c_o, h_o, w_o});
  x_cpu.Resize({n, c, h, w});
  filter_cpu.Resize({c_o, c / param.groups, 3, 3});
  y_cpu.Resize({n, c_o, h_o, w_o});
  bias_cpu.Resize({c_o});

  auto* x_data = x.mutable_data<float>(TARGET(kCUDA));
  auto* y_data = y.mutable_data<float>(TARGET(kCUDA));
  float* x_cpu_data = x_cpu.mutable_data<float>();
  float* filter_cpu_data = filter_cpu.mutable_data<float>();
  float* y_cpu_data = y_cpu.mutable_data<float>();
  float* bias_cpu_data = bias_cpu.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); i++) {
    x_cpu_data[i] = i;
  }
  std::vector<float> weight = {-0.2209115,
                               -0.17199445,
                               -0.2059412,
                               0.6763207,
                               -0.12260777,
                               -0.43123743,
                               -0.49696392,
                               -0.27471393,
                               -0.81017196};
  for (int i = 0; i < filter_cpu.numel(); i++) {
    filter_cpu_data[i] = weight[i];
  }
  for (int i = 0; i < bias_cpu.numel(); i++) {
    bias_cpu_data[i] = 0;
  }

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  filter.Assign<float, lite::DDim, TARGET(kCUDA)>(filter_cpu_data,
                                                  filter_cpu.dims());
  bias.Assign<float, lite::DDim, TARGET(kCUDA)>(bias_cpu_data, bias_cpu.dims());

  param.x = &x;
  param.filter = &filter;
  param.output = &y;
  // param.bias = &bias;

  conv_fp32.SetParam(param);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  conv_fp32.SetContext(std::move(ctx));
  conv_fp32.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      y_cpu_data, y_data, sizeof(float) * y.numel(), IoDirection::DtoH);

  std::vector<float> real_results = {-0.8, -0.7};
  for (int i = 0; i < y.numel(); i++) {
    LOG(INFO) << y_cpu_data[i];
  }
}

TEST(conv_compute, int8) {
  ConvComputeInt8<PRECISION(kFloat)> int8_conv_fp32out;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::ActivationParam act_param;
  act_param.has_active = true;
  act_param.active_type = lite_api::ActivationType::kRelu;
  operators::ConvParam param;
  // param.activation_param = act_param;
  param.groups = 1;

  Tensor x, filter, bias, y, x_cpu, filter_cpu, bias_cpu, y_cpu;
  int n = 1, c = 4, h = 3, w = 3;
  y.Resize({1, 1, 1, c});
  x_cpu.Resize({n, h, w, c});
  filter_cpu.Resize({c, 3, 3, c / param.groups});
  y_cpu.Resize({1, 1, 1, c});
  bias_cpu.Resize({c});

  auto* x_data = x.mutable_data<int8_t>(TARGET(kCUDA));
  auto* y_data = y.mutable_data<float>(TARGET(kCUDA));
  auto* x_cpu_data = x_cpu.mutable_data<int8_t>();
  auto* filter_cpu_data = filter_cpu.mutable_data<int8_t>();
  auto* y_cpu_data = x_cpu.mutable_data<float>();
  auto* bias_cpu_data = bias_cpu.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); i++) {
    x_cpu_data[i] = static_cast<int8_t>(1);
  }
  for (int i = 0; i < filter_cpu.numel(); i++) {
    filter_cpu_data[i] = static_cast<int8_t>(1);
  }
  for (int i = 0; i < bias_cpu.numel(); i++) {
    bias_cpu_data[i] = i + 1.0;
  }

  x.Assign<int8_t, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  filter.Assign<int8_t, lite::DDim, TARGET(kCUDA)>(filter_cpu_data,
                                                   filter_cpu.dims());
  bias.Assign<float, lite::DDim, TARGET(kCUDA)>(bias_cpu_data,
                                                filter_cpu.dims());

  param.x = &x;
  param.filter = &filter;
  param.output = &y;
  param.weight_scale = {1, 2, 3, 4};

  int8_conv_fp32out.SetParam(param);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  int8_conv_fp32out.SetContext(std::move(ctx));
  int8_conv_fp32out.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      y_cpu_data, y_data, sizeof(float) * y.numel(), IoDirection::DtoH);
  std::vector<float> real_results = {36, 72, 108, 144};
  for (int i = 0; i < y.numel(); i++) {
    EXPECT_NEAR(y_cpu_data[i], real_results[i], 1e-5);
  }
}

TEST(conv_compute, int8_int8_out) {
  ConvComputeInt8<PRECISION(kInt8)> int8_conv_fp32out;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::ActivationParam act_param;
  act_param.has_active = true;
  // act_param.active_type = core::ActiveType::Active_relu;
  act_param.active_type = lite_api::ActivationType::kLeakyRelu;
  act_param.Leaky_relu_alpha = 0.1;
  operators::ConvParam param;
  param.activation_param = act_param;
  param.groups = 1;

  Tensor x, filter, bias, y, x_cpu, filter_cpu, bias_cpu, y_cpu;
  int n = 1, c = 4, h = 3, w = 3;
  y.Resize({1, 1, 1, c});
  x_cpu.Resize({n, h, w, c});
  filter_cpu.Resize({c, 3, 3, c / param.groups});
  y_cpu.Resize({1, 1, 1, c});
  bias_cpu.Resize({c});

  auto* x_data = x.mutable_data<int8_t>(TARGET(kCUDA));
  auto* y_data = y.mutable_data<int8_t>(TARGET(kCUDA));
  auto* x_cpu_data = x_cpu.mutable_data<int8_t>();
  auto* filter_cpu_data = filter_cpu.mutable_data<int8_t>();
  auto* y_cpu_data = x_cpu.mutable_data<int8_t>();
  auto* bias_cpu_data = bias_cpu.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); i++) {
    x_cpu_data[i] = static_cast<int8_t>(random(-36, 36));
  }
  for (int i = 0; i < filter_cpu.numel(); i++) {
    filter_cpu_data[i] = static_cast<int8_t>(random(-10, 10));
  }
  for (int i = 0; i < bias_cpu.numel(); i++) {
    bias_cpu_data[i] = i + 1.0;
  }

  x.Assign<int8_t, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  filter.Assign<int8_t, lite::DDim, TARGET(kCUDA)>(filter_cpu_data,
                                                   filter_cpu.dims());
  bias.Assign<float, lite::DDim, TARGET(kCUDA)>(bias_cpu_data,
                                                filter_cpu.dims());

  param.x = &x;
  param.filter = &filter;
  param.output = &y;
  param.weight_scale = {0.01, 0.02, 0.03, 0.04};
  param.bias = &bias;

  int8_conv_fp32out.SetParam(param);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  int8_conv_fp32out.SetContext(std::move(ctx));
  int8_conv_fp32out.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      y_cpu_data, y_data, sizeof(int8_t) * y.numel(), IoDirection::DtoH);

  std::vector<float> real_results = {-1, 4, 0, -2};
  for (int i = 0; i < y.numel(); i++) {
    // EXPECT_NEAR(y_cpu_data[i], real_results[i], 1e-5);
    LOG(INFO) << float(y_cpu_data[i]);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
