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
#include <random>
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

template <typename T, int STRIDE_H = 1, int STRIDE_W = 1>
void depth_conv(const T* input_data,
                const lite::DDim& input_dims,
                const T* filter_data,
                const lite::DDim& filter_dims,
                T* output_data,
                const lite::DDim& output_dims) {
  int stride_h = STRIDE_H, stride_w = STRIDE_W;

  int64_t batches = input_dims[0];
  int64_t channels = input_dims[1];
  int64_t h = input_dims[2];
  int64_t w = input_dims[3];

  int64_t num_output = output_dims[1];
  int64_t outh = output_dims[2];
  int64_t outw = output_dims[3];

  int64_t filter_h = filter_dims[2];
  int64_t filter_w = filter_dims[3];

  const int64_t in_batch_size = channels * h * w;
  const int64_t out_batch_size = num_output * outh * outw;

  auto kernel_offset = std::unique_ptr<int[]>(new int[filter_h * filter_w]);
  {
    int p = 0;
    int offset = 0;
    int gap = w - filter_w;
    for (int i = 0; i < filter_h; i++) {
      for (int j = 0; j < filter_w; j++) {
        kernel_offset[p++] = offset;
        offset += 1;
      }
      offset += gap;
    }
  }

  for (int b = 0; b < batches; b++) {
    auto* input_batch_start = input_data + b * in_batch_size;
    auto* output_batch_start = output_data + b * out_batch_size;
    for (int p = 0; p < num_output; p++) {
      float* output_ptr = output_batch_start + p * outh * outw;
      const float* filter_ptr = filter_data + p * filter_h * filter_w;
      const float* input_ptr = input_batch_start + p * h * w;

      for (int i = 0; i < outh; i++) {
        for (int j = 0; j < outw; j++) {
          float sum = 0;
          const float* input_ch_start =
              input_ptr + i * stride_h * w + j * stride_w;

          for (int fh = 0; fh < filter_h; ++fh) {
            for (int fw = 0; fw < filter_w; ++fw) {
              float val = input_ch_start[kernel_offset[fh * filter_w + fw]];
              float w = filter_ptr[fh * filter_w + fw];
              sum += val * w;
            }
          }
          output_ptr[j] = sum;
        }

        output_ptr += outw;
      }
    }
  }
}

// buffer
#if 0
TEST(depthwise_conv2d_buffer_fp32, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create("depthwise_conv2d",
                                                 TARGET(kOpenCL),
                                                 PRECISION(kFloat),
                                                 DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel";
  lite::Tensor input, filter, output;
  operators::ConvParam param;
  param.x = &input;
  param.filter = &filter;
  param.output = &output;
  std::vector<int> paddings = {0, 0};
  param.paddings = std::make_shared<std::vector<int>>(paddings);
  param.strides = std::vector<int>{1, 1};

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> dep_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(dep_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(dep_context));

  std::default_random_engine engine;
  std::uniform_real_distribution<float> gen(-5, 5);
  std::vector<float> input_v(4 * 32 * 112 * 112);
  std::vector<float> filter_v(32 * 1 * 3 * 3);
  for (auto& i : input_v) {
    i = gen(engine);
  }
  for (auto& f : filter_v) {
    f = gen(engine);
  }

  input.Assign<float, lite::DDim, TARGET(kOpenCL)>(
      input_v.data(), lite::DDim{std::vector<int64_t>({4, 32, 112, 112})});
  filter.Assign<float, lite::DDim, TARGET(kOpenCL)>(
      filter_v.data(), lite::DDim{std::vector<int64_t>({32, 1, 3, 3})});
  output.Resize({4, 32, 110, 110});
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

  lite::Tensor output_ref;
  output_ref.Resize({4, 32, 110, 110});
  auto* output_ref_data = output_ref.mutable_data<float>(TARGET(kARM));
  auto* input_data = input.mutable_data<float, cl::Buffer>();
  auto* filter_data = filter.mutable_data<float, cl::Buffer>();
  auto* mapped_input = static_cast<float*>(TargetWrapperCL::Map(
      input_data, 0, sizeof(float) * input.dims().production()));
  auto* mapped_filter = static_cast<float*>(TargetWrapperCL::Map(
      filter_data, 0, sizeof(float) * filter.dims().production()));
  depth_conv<float, 1, 1>(mapped_input,
                          input.dims(),
                          mapped_filter,
                          filter.dims(),
                          output_ref_data,
                          output_ref.dims());

  auto* output_data = output.mutable_data<float, cl::Buffer>();
  auto* mapped_output = static_cast<float*>(TargetWrapperCL::Map(
      output_data, 0, sizeof(float) * output.dims().production()));

  for (int i = 0; i < output.dims().production(); i++) {
    EXPECT_NEAR(mapped_output[i], output_ref_data[i], 1e-4);
  }

  TargetWrapperCL::Unmap(output_data, mapped_output);
  TargetWrapperCL::Unmap(filter_data, mapped_filter);
  TargetWrapperCL::Unmap(input_data, mapped_input);
}
#endif

TEST(depthwise_conv2d_image2d_fp16, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create("depthwise_conv2d",
                                                 TARGET(kOpenCL),
                                                 PRECISION(kFP16),
                                                 DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel";
  lite::Tensor input, filter, output;
  operators::ConvParam param;
  param.x = &input;
  param.filter = &filter;
  param.output = &output;
  std::vector<int> paddings = {0, 0};
  param.paddings = std::make_shared<std::vector<int>>(paddings);
  param.strides = std::vector<int>{1, 1};
  std::vector<int> dilations = {1, 1};
  param.dilations = std::make_shared<std::vector<int>>(dilations);

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> dep_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(dep_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(dep_context));

  LOG(INFO) << "kernel ready";
  std::default_random_engine engine;
  std::uniform_real_distribution<float> gen(-5, 5);
  std::vector<float> input_v(1 * 32 * 112 * 112);
  std::vector<float> filter_v(32 * 1 * 3 * 3);
  for (auto& i : input_v) {
    i = gen(engine);
  }
  for (auto& f : filter_v) {
    f = gen(engine);
  }

  LOG(INFO) << "prepare input";
  input.Resize({1, 32, 112, 112});
  CLImageConverterDefault* default_converter = new CLImageConverterDefault();
  DDim input_image_shape =
      default_converter->InitImageDimInfoWith(input.dims());
  LOG(INFO) << "input_image_shape = " << input_image_shape[0] << " "
            << input_image_shape[1];
  std::vector<float> input_image_data(input_image_shape.production() *
                                      4);  // 4 : RGBA
  default_converter->NCHWToImage(
      input_v.data(), input_image_data.data(), input.dims());
  auto* input_image = input.mutable_data<int16_t, cl::Image2D>(
      input_image_shape[0], input_image_shape[1], input_image_data.data());

  LOG(INFO) << "prepare kernel";
  filter.Resize({32, 1, 3, 3});
  CLImageConverterNWBlock* nw_converter = new CLImageConverterNWBlock();
  DDim filter_image_shape = nw_converter->InitImageDimInfoWith(filter.dims());
  LOG(INFO) << "filter_image_shape = " << filter_image_shape[0] << " "
            << filter_image_shape[1];
  std::vector<float> filter_image_data(filter_image_shape.production() *
                                       4);  // 4 : RGBA
  nw_converter->NCHWToImage(
      filter_v.data(), filter_image_data.data(), filter.dims());
  auto* filter_image = filter.mutable_data<int16_t, cl::Image2D>(
      filter_image_shape[0], filter_image_shape[1], filter_image_data.data());

  LOG(INFO) << "launch";
  output.Resize({1, 32, 110, 110});
  DDim output_image_shape =
      default_converter->InitImageDimInfoWith(output.dims());
  LOG(INFO) << "output_image_shape = " << output_image_shape[0] << " "
            << output_image_shape[1];
  auto* output_image = output.mutable_data<int16_t, cl::Image2D>(
      output_image_shape[0], output_image_shape[1]);

  kernel->Launch();

  auto* wait_list = context->As<OpenCLContext>().cl_wait_list();
  auto* out_ptr = param.output->data<int16_t, cl::Image2D>();
  auto it = wait_list->find(out_ptr);
  if (it != wait_list->end()) {
    VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
    LOG(INFO) << "--- Find the sync event for the target cl tensor. ---";
    auto& event = *(it->second);
    event.wait();
  } else {
    LOG(FATAL) << "Could not find the sync event for the target cl tensor.";
    LOG(INFO) << "Could not find the sync event for the target cl tensor.";
  }

  lite::Tensor output_ref;
  output_ref.Resize({1, 32, 110, 110});
  auto* output_ref_data = output_ref.mutable_data<float>(TARGET(kARM));
  depth_conv<float, 1, 1>(input_v.data(),
                          input.dims(),
                          filter_v.data(),
                          filter.dims(),
                          output_ref_data,
                          output_ref.dims());

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};

  float* output_image_data = new float[output_image_shape.production() * 4];
  TargetWrapperCL::ImgcpySync(output_image_data,
                              output_image,
                              output_image_shape[0],
                              output_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);

  float* output_data = new float[output_image_shape.production() * 4];
  default_converter->ImageToNCHW(
      output_image_data, output_data, output_image_shape, output.dims());

  LOG(INFO) << "output_data vs output_ref_data";
  for (int i = 0; i < output.dims().production(); i++) {
    EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-4);
    LOG(INFO) << output_data[i] << " " << output_ref_data[i];
  }
}

}  // namespace lite
}  // namespace paddle

// USE_LITE_KERNEL(depthwise_conv2d, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(depthwise_conv2d, kOpenCL, kFP16, kImageDefault, image2d);
