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
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"

#define FP16_MAX_DIFF (5e-1)

namespace paddle {
namespace lite {

void pool_avg(const int padding_height,
              const int padding_width,
              const int stride_height,
              const int stride_width,
              const int ksize_height,
              const int ksize_width,
              const float* input_data,
              const DDim& in_dim,
              float* output_data,
              const DDim& out_dim,
              const bool adaptive = false) {
  const int batch_size = in_dim[0];
  const int input_height = in_dim[2];
  const int input_width = in_dim[3];
  const int output_channels = out_dim[1];
  const int output_height = out_dim[2];
  const int output_width = out_dim[3];
  LOG(INFO) << "adaptive: " << adaptive;

  const size_t input_spatial_size = input_height * input_width;
  const size_t output_spatial_size = output_height * output_width;

  for (int i = 0; i < batch_size; i++) {
    for (int c = 0; c < output_channels; ++c) {
      int channel = i * output_channels + c;
      const float* input_ptr = input_data + channel * input_spatial_size;
      float* output_ptr = output_data + channel * output_spatial_size;

      for (int ph = 0; ph < output_height; ++ph) {
        int hstart = ph * stride_height - padding_height;
        int hend = std::min(hstart + ksize_height, input_height);
        hstart = std::max(hstart, 0);
        if (adaptive) {
          hstart = ph * input_height / output_height;
          hend = (ph + 1) * input_height / output_height;
        }
        for (int pw = 0; pw < output_width; ++pw) {
          int wstart = pw * stride_width - padding_width;
          int wend = std::min(wstart + ksize_width, input_width);
          wstart = std::max(wstart, 0);
          if (adaptive) {
            wstart = pw * input_width / output_width;
            wend = (pw + 1) * input_width / output_width;
          }

          float val = 0.f;
          int count = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              val += input_ptr[h * input_width + w];
              ++count;
            }
          }
          output_ptr[ph * output_width + pw] =
              (count > 0) ? val * (1.f / count) : 0.f;
        }
      }
    }
  }
}

TEST(pool2d_image2d, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "pool2d", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel:" << kernel->doc();

  lite::Tensor x, out;
  operators::PoolParam param;
  param.x = &x;
  param.output = &out;
  param.global_pooling = false;
  param.pooling_type = "avg";
  std::vector<int> paddings = {1, 1, 1, 1};
  param.strides = std::vector<int>{1, 1};
  param.ksize = std::vector<int>{3, 3};
  param.paddings = std::make_shared<std::vector<int>>(paddings);

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> pool_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(pool_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(pool_context));

  const DDim in_dim = DDim(std::vector<DDim::value_type>{1, 384, 25, 25});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{1, 384, 25, 25});
  x.Resize(in_dim);
  out.Resize(out_dim);

  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);
  std::vector<float> input_v(4 * 11 * 107 * 107);
  for (auto& i : input_v) {
    i = dist(engine);
  }

  LOG(INFO) << "prepare input";
  CLImageConverterDefault* default_converter = new CLImageConverterDefault();
  DDim x_image_shape = default_converter->InitImageDimInfoWith(in_dim);
  LOG(INFO) << "x_image_shape = " << x_image_shape[0] << " "
            << x_image_shape[1];
  std::vector<half_t> x_image_data(x_image_shape.production() * 4);  // 4 : RGBA
  default_converter->NCHWToImage(input_v.data(), x_image_data.data(), in_dim);
  auto* x_image = x.mutable_data<half_t, cl::Image2D>(
      x_image_shape[0], x_image_shape[1], x_image_data.data());
  LOG(INFO) << "x_image:" << x_image;

  DDim out_image_shape = default_converter->InitImageDimInfoWith(out_dim);
  LOG(INFO) << "out_image_shape = " << out_image_shape[0] << " "
            << out_image_shape[1];
  auto* out_image = out.mutable_data<half_t, cl::Image2D>(out_image_shape[0],
                                                          out_image_shape[1]);
  LOG(INFO) << "out_image:" << out_image;
  kernel->Launch();

  CLRuntime::Global()->command_queue().finish();

  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  pool_avg(1,
           1,
           1,
           1,
           3,
           3,
           input_v.data(),
           in_dim,
           out_ref.get(),
           out_dim,
           param.adaptive);

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  half_t* out_image_data = new half_t[out_image_shape.production() * 4];
  TargetWrapperCL::ImgcpySync(out_image_data,
                              out_image,
                              out_image_shape[0],
                              out_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  float* out_data = new float[out_image_shape.production() * 4];
  default_converter->ImageToNCHW(
      out_image_data, out_data, out_image_shape, out_dim);

  for (int i = 0; i < out_dim.production(); i++) {
    auto abs_diff = abs(out_data[i] - out_ref[i]);
    auto relative_diff = COMPUTE_RELATIVE_DIFF(out_data[i], out_ref[i]);
    EXPECT_EQ((relative_diff <= FP16_MAX_DIFF) || (abs_diff <= FP16_MAX_DIFF),
              true);
    if ((relative_diff > FP16_MAX_DIFF) && (abs_diff > FP16_MAX_DIFF)) {
      LOG(ERROR) << "error idx:" << i << " out_data[" << i
                 << "]:" << out_data[i] << " "
                                           "out_ref["
                 << i << "]:" << out_ref[i] << " abs_diff:" << abs_diff
                 << " relative_diff:" << relative_diff
                 << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
    }
  }
}

TEST(pool2d_global_pooling_false, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "pool2d", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel:" << kernel->doc();

  lite::Tensor x, out;
  operators::PoolParam param;
  param.x = &x;
  param.output = &out;
  param.global_pooling = false;
  param.pooling_type = "avg";
  std::vector<int> paddings = {0, 0, 0, 0};
  param.strides = std::vector<int>{1, 1};
  param.ksize = std::vector<int>{1, 1};
  param.paddings = std::make_shared<std::vector<int>>(paddings);
  param.adaptive = true;

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> pool_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(pool_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(pool_context));

  const DDim in_dim = DDim(std::vector<DDim::value_type>{1, 256, 16, 16});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{1, 256, 1, 1});
  x.Resize(in_dim);
  out.Resize(out_dim);

  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);
  std::vector<float> input_v(1 * 256 * 16 * 16);
  for (auto& i : input_v) {
    i = dist(engine);
  }

  LOG(INFO) << "prepare input";
  CLImageConverterDefault* default_converter = new CLImageConverterDefault();
  DDim x_image_shape = default_converter->InitImageDimInfoWith(in_dim);
  LOG(INFO) << "x_image_shape = " << x_image_shape[0] << " "
            << x_image_shape[1];
  std::vector<half_t> x_image_data(x_image_shape.production() * 4);  // 4 : RGBA
  default_converter->NCHWToImage(input_v.data(), x_image_data.data(), in_dim);
  auto* x_image = x.mutable_data<half_t, cl::Image2D>(
      x_image_shape[0], x_image_shape[1], x_image_data.data());
  LOG(INFO) << "x_image:" << x_image;

  DDim out_image_shape = default_converter->InitImageDimInfoWith(out_dim);
  LOG(INFO) << "out_image_shape = " << out_image_shape[0] << " "
            << out_image_shape[1];
  auto* out_image = out.mutable_data<half_t, cl::Image2D>(out_image_shape[0],
                                                          out_image_shape[1]);
  LOG(INFO) << "out_image:" << out_image;
  kernel->Launch();

  CLRuntime::Global()->command_queue().finish();

  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  pool_avg(0,
           0,
           1,
           1,
           1,
           1,
           input_v.data(),
           in_dim,
           out_ref.get(),
           out_dim,
           param.adaptive);

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  half_t* out_image_data = new half_t[out_image_shape.production() * 4];
  TargetWrapperCL::ImgcpySync(out_image_data,
                              out_image,
                              out_image_shape[0],
                              out_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  float* out_data = new float[out_image_shape.production() * 4];
  default_converter->ImageToNCHW(
      out_image_data, out_data, out_image_shape, out_dim);

  for (int i = 0; i < out_dim.production(); i++) {
    auto abs_diff = abs(out_data[i] - out_ref[i]);
    auto relative_diff = COMPUTE_RELATIVE_DIFF(out_data[i], out_ref[i]);
    EXPECT_EQ((relative_diff <= FP16_MAX_DIFF) || (abs_diff <= FP16_MAX_DIFF),
              true);
    if ((relative_diff > FP16_MAX_DIFF) && (abs_diff > FP16_MAX_DIFF)) {
      LOG(ERROR) << "error idx:" << i << " out_data[" << i
                 << "]:" << out_data[i] << " "
                                           "out_ref["
                 << i << "]:" << out_ref[i] << " abs_diff:" << abs_diff
                 << " relative_diff:" << relative_diff
                 << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
    }
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(pool2d, kOpenCL, kFP16, kImageDefault, image2d);
