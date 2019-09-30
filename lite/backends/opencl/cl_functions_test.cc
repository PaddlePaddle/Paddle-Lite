/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <memory>
#include <random>
#include <vector>
#include "lite/backends/opencl/cl_caller.h"
#include "lite/backends/opencl/cl_context.h"
#include "lite/backends/opencl/cl_image.h"
#include "lite/backends/opencl/cl_runtime.h"
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/tensor.h"
#include "lite/utils/cp_logging.h"

DEFINE_string(cl_path, "/data/local/tmp/opencl", "The OpenCL kernels path.");

namespace paddle {
namespace lite {

TEST(cl_test, runtime_test) {
  auto *runtime = CLRuntime::Global();
  CHECK(runtime->IsInitSuccess());
  runtime->set_cl_path(FLAGS_cl_path);
  runtime->platform();
  runtime->device();
  runtime->command_queue();
  auto &context = runtime->context();
  auto program = runtime->CreateProgram(
      context,
      runtime->cl_path() + "/cl_kernel/" + "image/elementwise_add_kernel.cl");
  auto event = runtime->CreateEvent(context);
  CHECK(runtime->BuildProgram(program.get()));
}

TEST(cl_test, context_test) {
  auto *runtime = CLRuntime::Global();
  CHECK(runtime->IsInitSuccess());
  runtime->set_cl_path(FLAGS_cl_path);
  CLContext context;
  context.AddKernel("pool_max", "image/pool_kernel.cl", "");
  context.AddKernel("elementwise_add", "image/elementwise_add_kernel.cl", "");
  context.AddKernel("elementwise_add", "image/elementwise_add_kernel.cl", "");
}

TEST(cl_test, kernel_test) {
  auto *runtime = CLRuntime::Global();
  CHECK(runtime->IsInitSuccess());
  runtime->set_cl_path(FLAGS_cl_path);
  std::unique_ptr<CLContext> context(new CLContext);
  context->AddKernel("elementwise_add", "image/elementwise_add_kernel.cl");
  context->AddKernel("pool_max", "image/pool_kernel.cl");
  context->AddKernel("elementwise_add", "image/elementwise_add_kernel.cl");
  auto kernel = context->GetKernel(2);

  std::unique_ptr<float[]> in_data(new float[4 * 3 * 256 * 512]);
  for (int i = 0; i < 4 * 3 * 256 * 512; i++) {
    in_data[i] = 1.f;
  }
  const DDim in_dim = DDim(std::vector<DDim::value_type>{4, 3, 256, 512});
  CLImage in_image;
  in_image.set_tensor_data(in_data.get(), in_dim);
  in_image.InitNormalCLImage(context->GetContext());
  LOG(INFO) << in_image;

  std::unique_ptr<float[]> bias_data(new float[4 * 3 * 256 * 512]);
  for (int i = 0; i < 4 * 3 * 256 * 512; i++) {
    bias_data[i] = 2.f;
  }
  const DDim bias_dim = DDim(std::vector<DDim::value_type>{4, 3, 256, 512});
  CLImage bias_image;
  bias_image.set_tensor_data(bias_data.get(), bias_dim);
  bias_image.InitNormalCLImage(context->GetContext());
  LOG(INFO) << bias_image;

  CLImage out_image;
  const DDim out_dim = DDim(std::vector<DDim::value_type>{4, 3, 256, 512});
  out_image.InitEmptyImage(context->GetContext(), out_dim);
  LOG(INFO) << out_image;

  cl_int status;
  status = kernel.setArg(0, *in_image.cl_image());
  CL_CHECK_FATAL(status);
  status = kernel.setArg(1, *bias_image.cl_image());
  CL_CHECK_FATAL(status);
  status = kernel.setArg(2, *out_image.cl_image());
  CL_CHECK_FATAL(status);

  size_t width = in_image.ImageWidth();
  size_t height = in_image.ImageHeight();
  auto global_work_size = cl::NDRange{width, height};
  cl::Event event;
  status = context->GetCommandQueue().enqueueNDRangeKernel(
      kernel, cl::NullRange, global_work_size, cl::NullRange, nullptr, &event);
  CL_CHECK_FATAL(status);
  status = context->GetCommandQueue().finish();
  CL_CHECK_FATAL(status);
  double start_nanos = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  double stop_nanos = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  double elapsed_micros = (stop_nanos - start_nanos) / 1000.0;
  LOG(INFO) << "Kernel Run Cost Time: " << elapsed_micros << " us.";
  LOG(INFO) << out_image;
}

TEST(cl_test, channel_add_test) {
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);

  const DDim in_dim = DDim(std::vector<DDim::value_type>{4, 16, 256, 512});
  std::unique_ptr<float[]> in_data(new float[4 * 16 * 256 * 512]);
  for (int i = 0; i < 4 * 16 * 256 * 512; i++) {
    in_data[i] = dist(engine);
  }

  const DDim bias_dim = DDim(std::vector<DDim::value_type>{16});
  std::unique_ptr<float[]> bias_data(new float[16]);
  for (int i = 0; i < 16; i++) {
    bias_data[i] = dist(engine);
  }

  std::unique_ptr<float[]> out_ref(new float[4 * 16 * 256 * 512]);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 16; j++) {
      float b = bias_data[j];
      for (int k = 0; k < 256 * 512; k++) {
        int index = (i * 16 + j) * 256 * 512 + k;
        out_ref[index] = in_data[index] + b;
      }
    }
  }

  const DDim out_dim = DDim(std::vector<DDim::value_type>{4, 16, 256, 512});
  std::unique_ptr<float[]> out(new float[4 * 16 * 256 * 512]);

  bool status = InitOpenCLRuntime(FLAGS_cl_path);
  CHECK(status) << "Fail to initialize OpenCL runtime.";
  std::unique_ptr<CLContext> context(new CLContext);
  context->AddKernel("elementwise_add", "image/elementwise_add_kernel.cl");
  context->AddKernel("channel_add", "image/channel_add_kernel.cl");
  elementwise_add(context.get(),
                  in_data.get(),
                  in_dim,
                  bias_data.get(),
                  bias_dim,
                  out.get(),
                  out_dim);

  int stride = 4 * 16 * 256 * 512 / 20;
  for (int i = 0; i < 4 * 16 * 256 * 512; i += stride) {
    std::cout << out[i] << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < 4 * 16 * 256 * 512; i++) {
    EXPECT_NEAR(out[i], out_ref[i], 1e-6);
  }
}

TEST(cl_test, elementwise_add_test) {
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);

  const DDim in_dim = DDim(std::vector<DDim::value_type>{4, 16, 256, 512});
  std::unique_ptr<float[]> in_data(new float[4 * 16 * 256 * 512]);
  for (int i = 0; i < 4 * 16 * 256 * 512; i++) {
    in_data[i] = dist(engine);
  }

  const DDim bias_dim = DDim(std::vector<DDim::value_type>{4, 16, 256, 512});
  std::unique_ptr<float[]> bias_data(new float[4 * 16 * 256 * 512]);
  for (int i = 0; i < 4 * 16 * 256 * 512; i++) {
    bias_data[i] = dist(engine);
  }

  std::unique_ptr<float[]> out_ref(new float[4 * 16 * 256 * 512]);
  for (int i = 0; i < 4 * 16 * 256 * 512; i++) {
    out_ref[i] = in_data[i] + bias_data[i];
  }

  const DDim out_dim = DDim(std::vector<DDim::value_type>{4, 16, 256, 512});
  std::unique_ptr<float[]> out(new float[4 * 16 * 256 * 512]);

  bool status = InitOpenCLRuntime(FLAGS_cl_path);
  CHECK(status) << "Fail to initialize OpenCL runtime.";
  std::unique_ptr<CLContext> context(new CLContext);
  context->AddKernel("elementwise_add", "image/elementwise_add_kernel.cl");
  context->AddKernel("channel_add", "image/channel_add_kernel.cl");
  elementwise_add(context.get(),
                  in_data.get(),
                  in_dim,
                  bias_data.get(),
                  bias_dim,
                  out.get(),
                  out_dim);

  int stride = 4 * 16 * 256 * 512 / 20;
  for (int i = 0; i < 4 * 16 * 256 * 512; i += stride) {
    std::cout << out[i] << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < 4 * 16 * 256 * 512; i++) {
    EXPECT_NEAR(out[i], out_ref[i], 1e-6);
  }
}

void pool_avg(const int padding_height,
              const int padding_width,
              const int stride_height,
              const int stride_width,
              const int ksize_height,
              const int ksize_width,
              const float *input_data,
              const DDim &in_dim,
              float *output_data,
              const DDim &out_dim) {
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
      const float *input_ptr = input_data + channel * input_spatial_size;
      float *output_ptr = output_data + channel * output_spatial_size;

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

TEST(cl_test, pool_test) {
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);

  const DDim in_dim = DDim(std::vector<DDim::value_type>{4, 1024, 7, 7});
  std::unique_ptr<float[]> in_data(new float[4 * 1024 * 7 * 7]);
  for (int i = 0; i < 4 * 1024 * 7 * 7; i++) {
    in_data[i] = dist(engine);
  }

  const DDim out_dim = DDim(std::vector<DDim::value_type>{4, 1024, 1, 1});
  std::unique_ptr<float[]> out(new float[4 * 1024 * 1 * 1]);
  std::unique_ptr<float[]> out_ref(new float[4 * 1024 * 1 * 1]);

  bool status = InitOpenCLRuntime(FLAGS_cl_path);
  CHECK(status) << "Fail to initialize OpenCL runtime.";
  std::unique_ptr<CLContext> context(new CLContext);
  context->AddKernel("pool_max", "image/pool_kernel.cl");
  context->AddKernel("pool_avg", "image/pool_kernel.cl");
  pool(context.get(),
       "avg",
       0,
       0,
       1,
       1,
       7,
       7,
       in_data.get(),
       in_dim,
       out.get(),
       out_dim);
  pool_avg(0, 0, 1, 1, 7, 7, in_data.get(), in_dim, out_ref.get(), out_dim);

  for (int i = 0; i < 4 * 1024 * 1 * 1; i++) {
    EXPECT_NEAR(out[i], out_ref[i], 1e-6);
  }
}

TEST(cl_test, target_wrapper_buffer_test) {
  bool inited = InitOpenCLRuntime(FLAGS_cl_path);
  CHECK(inited) << "Fail to initialize OpenCL runtime.";
  std::unique_ptr<CLContext> context(new CLContext);
  std::string kernel_name = "elementwise_add";
  std::string build_options = "-DCL_DTYPE=float";
  context->AddKernel(
      kernel_name, "buffer/elementwise_add_kernel.cl", build_options);
  std::vector<float> h_a;
  std::vector<float> h_b;
  std::vector<float> h_out;
  std::vector<float> h_ref;
  for (int i = 0; i < 10; i++) {
    h_a.push_back(3.14f * i);
    h_b.push_back(6.28f * i);
    h_out.push_back(0);
    h_ref.push_back((3.14f + 6.28f) * i);
  }
  auto *d_a = static_cast<cl::Buffer *>(
      TargetWrapperCL::Malloc(sizeof(float) * h_a.size()));
  auto *d_b = static_cast<cl::Buffer *>(
      TargetWrapperCL::Malloc(sizeof(float) * h_b.size()));
  auto *d_out =
      static_cast<cl::Buffer *>(TargetWrapperCL::Malloc(sizeof(float) * 10));
  auto *d_copy =
      static_cast<cl::Buffer *>(TargetWrapperCL::Malloc(sizeof(float) * 10));
  TargetWrapperCL::MemcpySync(
      d_a, h_a.data(), sizeof(float) * h_a.size(), IoDirection::HtoD);
  TargetWrapperCL::MemcpySync(
      d_b, h_b.data(), sizeof(float) * h_b.size(), IoDirection::HtoD);
  // x + y: x[n=1, c=10, h=1, w=1], y[c=10]
  auto kernel = context->GetKernel(kernel_name + build_options);
  cl_int status = kernel.setArg(0, *d_a);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(1, *d_b);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(2, *d_out);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(3, 1);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(4, 10);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(5, 1);
  CL_CHECK_FATAL(status);
  auto global_work_size = cl::NDRange{10, 1};
  status = context->GetCommandQueue().enqueueNDRangeKernel(
      kernel, cl::NullRange, global_work_size, cl::NullRange, nullptr, nullptr);
  CL_CHECK_FATAL(status);
  status = context->GetCommandQueue().finish();
  CL_CHECK_FATAL(status);
  TargetWrapperCL::MemcpySync(
      h_out.data(), d_out, sizeof(float) * 10, IoDirection::DtoH);

  for (int i = 0; i < 10; i++) {
    std::cout << h_out[i] << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < 10; i++) {
    EXPECT_NEAR(h_out[i], h_ref[i], 1e-5);
  }

  TargetWrapperCL::MemcpySync(
      d_copy, d_out, sizeof(float) * 10, IoDirection::DtoD);
  std::fill(h_out.begin(), h_out.end(), 0);
  for (int i = 0; i < 10; i++) {
    EXPECT_NEAR(h_out[i], 0, 1e-5);
  }
  TargetWrapperCL::MemcpySync(
      h_out.data(), d_copy, sizeof(float) * 10, IoDirection::DtoH);
  for (int i = 0; i < 10; i++) {
    EXPECT_NEAR(h_out[i], h_ref[i], 1e-5);
  }

  auto *mapped_ptr =
      static_cast<float *>(TargetWrapperCL::Map(d_copy, 0, sizeof(float) * 10));
  for (int i = 0; i < 10; i++) {
    EXPECT_NEAR(mapped_ptr[i], h_ref[i], 1e-5);
  }
  TargetWrapperCL::Unmap(d_copy, mapped_ptr);

  TargetWrapperCL::Free(d_copy);
  TargetWrapperCL::Free(d_out);
  TargetWrapperCL::Free(d_b);
  TargetWrapperCL::Free(d_a);
}

TEST(cl_test, target_wrapper_image_test) {
  const std::array<size_t, 2> image_shape{28, 32};
  auto *d_image = static_cast<cl::Image2D *>(
      TargetWrapperCL::MallocImage(image_shape, PRECISION(kFloat)));
  std::array<size_t, 2> image_pitch;
  // Map/Unmap test
  auto *h_image = static_cast<float *>(
      TargetWrapperCL::MapImage(d_image, image_shape, &image_pitch));
  // row_pitch = 448 = 28 * 4 (RGBA: 4 floats) * 4 (float in bytes)
  // slice_pitch = 0
  size_t row_pitch = image_pitch[0];
  size_t slice_pitch = image_pitch[1];
  CHECK_EQ(row_pitch, 448);
  CHECK_EQ(slice_pitch, 0);
  LOG(INFO) << "row_pitch = " << row_pitch << ", slice_pitch " << slice_pitch;

  for (int i = 0; i < 10; i++) {
    h_image[i] = 3.14f * i;
  }
  TargetWrapperCL::Unmap(d_image, h_image);

  auto *h_ptr = static_cast<float *>(
      TargetWrapperCL::MapImage(d_image, image_shape, &image_pitch));
  for (int i = 0; i < 10; i++) {
    EXPECT_NEAR(h_ptr[i], 3.14f * i, 1e-6);
  }
  TargetWrapperCL::Unmap(d_image, h_ptr);

  // Imagecpy test
  std::vector<float> h_image_cpy(28 * 4 * 32);
  for (int i = 0; i < 28 * 4 * 32; i++) {
    h_image_cpy[i] = 3.14f;
  }
  TargetWrapperCL::ImgcpySync(
      d_image, h_image_cpy.data(), image_shape, image_pitch, IoDirection::HtoD);
  auto *d_image_cpy = static_cast<cl::Image2D *>(
      TargetWrapperCL::MallocImage(image_shape, PRECISION(kFloat)));
  TargetWrapperCL::ImgcpySync(
      d_image_cpy, d_image, image_shape, image_pitch, IoDirection::DtoD);
  std::fill(h_image_cpy.begin(), h_image_cpy.end(), 0);
  TargetWrapperCL::ImgcpySync(h_image_cpy.data(),
                              d_image_cpy,
                              image_shape,
                              image_pitch,
                              IoDirection::DtoH);
  for (int i = 0; i < 28 * 4 * 32; i++) {
    EXPECT_NEAR(h_image_cpy[i], 3.14f, 1e-6);
  }

  TargetWrapperCL::FreeImage(d_image_cpy);
  TargetWrapperCL::FreeImage(d_image);
}

}  // namespace lite
}  // namespace paddle
