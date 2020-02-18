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
              const DDim& out_dim) {
  const int batch_size = in_dim[0];
  const int input_height = in_dim[2];
  const int input_width = in_dim[3];
  const int output_channels = out_dim[1];
  const int output_height = out_dim[2];
  const int output_width = out_dim[3];

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
        for (int pw = 0; pw < output_width; ++pw) {
          int wstart = pw * stride_width - padding_width;
          int wend = std::min(wstart + ksize_width, input_width);
          wstart = std::max(wstart, 0);

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

// buffer
#if 0   // pool_buffer
TEST(pool2d_buffer_fp32, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "pool2d", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());
  LOG(INFO) << "get kernel:" << kernel->doc();

  lite::Tensor x, out;
  operators::PoolParam param;
  param.x = &x;
  param.output = &out;
  param.global_pooling = true;
  param.pooling_type = "avg";
  std::vector<int> paddings = {0, 0, 0, 0};
  param.strides = std::vector<int>{1, 1};
  param.ksize = std::vector<int>{7, 7};
  param.paddings = std::make_shared<std::vector<int>>(paddings);

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> pool_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(pool_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(pool_context));

  const DDim in_dim = DDim(std::vector<DDim::value_type>{4, 1024, 7, 7});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{4, 1024, 1, 1});
  x.Resize(in_dim);
  out.Resize(out_dim);

  auto* x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);
  auto* mapped_x = static_cast<float*>(
      TargetWrapperCL::Map(x_data, 0, sizeof(float) * in_dim.production()));
  for (int i = 0; i < in_dim.production(); i++) {
    mapped_x[i] = dist(engine);
  }

  kernel->Launch();

  auto* wait_list = context->As<OpenCLContext>().cl_wait_list();
  auto* out_ptr = param.output->data<float, cl::Buffer>();
  auto it = wait_list->find(out_ptr);
  if (it != wait_list->end()) {
    VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
    auto& event = *(it->second);
    event.wait();
  } else {
    LOG(FATAL) << "Could not find the sync event for the target cl tensor.";
  }

  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  pool_avg(0, 0, 1, 1, 7, 7, mapped_x, in_dim, out_ref.get(), out_dim);
  TargetWrapperCL::Unmap(x_data, mapped_x);
  auto* out_data = out.mutable_data<float, cl::Buffer>();
  auto* mapped_out = static_cast<float*>(
      TargetWrapperCL::Map(out_data, 0, sizeof(float) * out_dim.production()));
  for (int i = 0; i < out_dim.production(); i++) {
    EXPECT_NEAR(mapped_out[i], out_ref[i], 1e-6);
  }
  TargetWrapperCL::Unmap(out_data, mapped_out);
}
#endif  // pool_buffer

TEST(pool2d_image2d_fp32, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "pool2d", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kImageDefault));
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
  param.ksize = std::vector<int>{7, 7};
  param.paddings = std::make_shared<std::vector<int>>(paddings);

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> pool_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(pool_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(pool_context));

  const DDim in_dim = DDim(std::vector<DDim::value_type>{4, 11, 107, 107});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{4, 11, 101, 101});
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
  std::vector<float> x_image_data(x_image_shape.production() * 4);  // 4 : RGBA
  default_converter->NCHWToImage(input_v.data(), x_image_data.data(), in_dim);
  auto* x_image = x.mutable_data<float, cl::Image2D>(
      x_image_shape[0], x_image_shape[1], x_image_data.data());
  LOG(INFO) << "x_image:" << x_image;

  DDim out_image_shape = default_converter->InitImageDimInfoWith(out_dim);
  LOG(INFO) << "out_image_shape = " << out_image_shape[0] << " "
            << out_image_shape[1];
  auto* out_image = out.mutable_data<float, cl::Image2D>(out_image_shape[0],
                                                         out_image_shape[1]);
  LOG(INFO) << "out_image:" << out_image;
  kernel->Launch();

  auto* wait_list = context->As<OpenCLContext>().cl_wait_list();
  auto* out_ptr = param.output->data<float, cl::Image2D>();
  auto it = wait_list->find(out_ptr);
  if (it != wait_list->end()) {
    VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
    auto& event = *(it->second);
    event.wait();
  } else {
    LOG(FATAL) << "Could not find the sync event for the target cl tensor.";
  }

  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  pool_avg(0, 0, 1, 1, 7, 7, input_v.data(), in_dim, out_ref.get(), out_dim);

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  float* out_image_data = new float[out_image_shape.production() * 4];
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
    EXPECT_NEAR(out_data[i], out_ref[i], 1e-6);
  }
}

}  // namespace lite
}  // namespace paddle

// USE_LITE_KERNEL(pool2d, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(pool2d, kOpenCL, kFloat, kImageDefault, image2d);
