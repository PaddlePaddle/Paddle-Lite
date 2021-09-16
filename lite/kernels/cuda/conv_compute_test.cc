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
#include <random>
#include <utility>
#include <vector>

#include "lite/api/test/test_helper.h"
#include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

static float random_num(float low, float high) {
  static std::mt19937 mt(100);
  std::uniform_real_distribution<double> dist(low, high);
  return dist(mt);
}

class Conv2dTest : public ::testing::Test {
 protected:
  Conv2dTest()
      : batch(16),
        in_channels(32),
        out_channels(128),
        height(64),
        width(64),
        kernel_h(5),
        kernel_w(5),
        stride_h(2),
        stride_w(2),
        pad_h(1),
        pad_w(1),
        dilation_h(2),
        dilation_w(2),
        groups(1),
        x_shape({batch, in_channels, height, width}),
        w_shape({out_channels, in_channels, kernel_h, kernel_w}),
        b_shape({out_channels}) {
    calc_output_shape();

    X_gpu.Resize(lite::DDim(x_shape));
    X_ref.Resize(lite::DDim(x_shape));

    W_gpu.Resize(lite::DDim(w_shape));
    W_ref.Resize(lite::DDim(w_shape));

    b_gpu.Resize(lite::DDim(b_shape));
    b_ref.Resize(lite::DDim(b_shape));

    auto x_ref_data = X_ref.mutable_data<float>();
    auto w_ref_data = W_ref.mutable_data<float>();
    auto b_ref_data = b_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < X_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < W_ref.numel(); i++) {
      w_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < b_ref.numel(); i++) {
      b_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }

    Out_ref.Resize(lite::DDim(out_shape));
    Out_gpu.Resize(lite::DDim(out_shape));
    Out_cpu.Resize(lite::DDim(out_shape));

    device_init();
  }

  int ConvOutputSize(
      int input_size, int filter_size, int dilation, int pad, int stride) {
    const int dkernel = dilation * (filter_size - 1) + 1;
    int output_size = (input_size + pad * 2 - dkernel) / stride + 1;
    return output_size;
  }

  void calc_output_shape() {
    out_shape.clear();
    out_shape.push_back(batch);
    out_shape.push_back(out_channels);
    out_shape.push_back(
        ConvOutputSize(height, kernel_h, dilation_h, pad_h, stride_h));
    out_shape.push_back(
        ConvOutputSize(width, kernel_w, dilation_w, pad_w, stride_w));
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    param.x = &X_gpu;
    param.filter = &W_gpu;
    param.output = &Out_gpu;
    param.bias = &b_gpu;
    param.paddings.reset(new std::vector<int>);
    param.paddings->push_back(pad_h);
    param.paddings->push_back(pad_h);
    param.paddings->push_back(pad_w);
    param.paddings->push_back(pad_w);
    param.dilations.reset(new std::vector<int>);
    param.dilations->push_back(dilation_h);
    param.dilations->push_back(dilation_w);
    param.strides[0] = stride_h;
    param.strides[1] = stride_w;
  }

  void float_data_init() {
    X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref.data<float>(),
                                                   X_gpu.dims());
    X_gpu.set_lod(X_ref.lod());
    W_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(W_ref.data<float>(),
                                                   W_gpu.dims());
    b_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(b_ref.data<float>(),
                                                   b_gpu.dims());
  }

  void half_data_init() {
    X_half.Resize(lite::DDim(x_shape));
    auto x_half_data = X_half.mutable_data<half>();
    for (int64_t i = 0; i < X_half.numel(); i++) {
      x_half_data[i] = half(lite::float16(X_ref.data<float>()[i]));
    }
    X_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, X_gpu.dims());
    X_gpu.set_lod(X_ref.lod());

    W_half.Resize(W_ref.dims());
    auto w_half_data = W_half.mutable_data<half>();
    for (int64_t i = 0; i < W_half.numel(); i++) {
      w_half_data[i] = half(lite::float16(W_ref.data<float>()[i]));
    }
    W_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(w_half_data, W_gpu.dims());

    b_half.Resize(b_ref.dims());
    auto b_half_data = b_half.mutable_data<half>();
    for (int64_t i = 0; i < b_half.numel(); i++) {
      b_half_data[i] = half(lite::float16(b_ref.data<float>()[i]));
    }
    b_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(b_half_data, b_gpu.dims());
  }

  void conv_cpu_base(const lite::Tensor* X,
                     const lite::Tensor* W,
                     lite::Tensor* Out,
                     lite::Tensor* Col) {}

  int batch, in_channels, out_channels, height, width;
  int kernel_h, kernel_w;
  int stride_h, stride_w;
  int pad_h, pad_w;
  int dilation_h, dilation_w, groups;
  std::vector<int64_t> x_shape, w_shape, b_shape, out_shape;
  lite::Tensor X_ref, W_ref, b_ref, Out_ref;
  lite::Tensor X_gpu, W_gpu, b_gpu;
  lite::Tensor X_half, W_half, b_half;
  lite::Tensor Out_cpu, Out_gpu;

  operators::ConvParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(Conv2dTest, fp32) {
  float_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  ConvCompute<float, PRECISION(kFloat)> conv_2d_kernel;
  conv_2d_kernel.SetParam(param);
  conv_2d_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    conv_2d_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  conv_2d_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    conv_2d_kernel.Run();
  }
  cudaDeviceSynchronize();
  auto duration = (GetCurrentUS() - start) / 1000.0;
  LOG(INFO) << "fp32, warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << duration / FLAGS_repeats << " ms in average.";
}

TEST_F(Conv2dTest, fp16) {
  half_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  ConvCompute<half, PRECISION(kFP16)> conv_2d_kernel;
  conv_2d_kernel.SetParam(param);
  conv_2d_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    conv_2d_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  conv_2d_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    conv_2d_kernel.Run();
  }
  cudaDeviceSynchronize();
  auto duration = (GetCurrentUS() - start) / 1000.0;
  LOG(INFO) << "fp16, warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << duration / FLAGS_repeats << " ms in average.";
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

  std::vector<int> pads = {0, 0, 0, 0};
  std::vector<int> dilations = {1, 1, 1, 1};
  param.paddings = std::make_shared<std::vector<int>>(pads);
  param.dilations = std::make_shared<std::vector<int>>(dilations);
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
  // for (int i = 0; i < y.numel(); i++) {
  //   EXPECT_NEAR(y_cpu_data[i], real_results[i], 1e-5);
  // }
}

TEST(conv_compute, int8_int8_out) {
  ConvComputeInt8<PRECISION(kInt8)> int8_conv_fp32out;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::ActivationParam act_param;
  act_param.has_active = true;
  act_param.active_type = lite_api::ActivationType::kRelu;
  // act_param.active_type = lite_api::ActivationType::kLeakyRelu;
  act_param.Leaky_relu_alpha = 0.1;
  operators::ConvParam param;
  param.activation_param = act_param;
  param.groups = 1;

  Tensor x, filter, bias, y, x_cpu, filter_cpu, bias_cpu, y_cpu;
  int c_i = 3, h_i = 3, w_i = 3;
  int n = 1, c = 4;
  y.Resize({1, 1, 1, c});
  x_cpu.Resize({n, h_i, w_i, c_i});
  filter_cpu.Resize({c, 3, 3, c_i / param.groups});
  y_cpu.Resize({1, 1, 1, c});
  bias_cpu.Resize({c});

  auto* y_data = y.mutable_data<int8_t>(TARGET(kCUDA));
  auto* x_cpu_data = x_cpu.mutable_data<int8_t>();
  auto* filter_cpu_data = filter_cpu.mutable_data<int8_t>();
  auto* y_cpu_data = x_cpu.mutable_data<int8_t>();
  auto* bias_cpu_data = bias_cpu.mutable_data<float>();

  std::cout << "input" << std::endl;
  for (int i = 0; i < x_cpu.numel(); i++) {
    x_cpu_data[i] = static_cast<int8_t>(random_num(-36, 36));
  }
  std::cout << "filter" << std::endl;
  for (int i = 0; i < filter_cpu.numel(); i++) {
    filter_cpu_data[i] = static_cast<int8_t>(random_num(-10, 10));
  }
  for (int i = 0; i < bias_cpu.numel(); i++) {
    bias_cpu_data[i] = i + 1.0;
    //  bias_cpu_data[i] = 0;
  }

  x.Assign<int8_t, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  filter.Assign<int8_t, lite::DDim, TARGET(kCUDA)>(filter_cpu_data,
                                                   filter_cpu.dims());
  bias.Assign<float, lite::DDim, TARGET(kCUDA)>(bias_cpu_data,
                                                filter_cpu.dims());

  std::vector<int> pads = {0, 0, 0, 0};
  std::vector<int> dilations = {1, 1, 1, 1};
  param.paddings = std::make_shared<std::vector<int>>(pads);
  param.dilations = std::make_shared<std::vector<int>>(dilations);
  param.x = &x;
  param.filter = &filter;
  param.output = &y;
  param.weight_scale = {0.01, 0.02, 0.03, 0.04};
  param.output_scale = 2;
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

  std::vector<float> real_results = {0, 7, 8, 1};
  for (int i = 0; i < y.numel(); i++) {
    // EXPECT_NEAR(y_cpu_data[i], real_results[i], 1e-5);
    LOG(INFO) << float(y_cpu_data[i]);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
