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

#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <random>
#include <vector>
#include "lite/backends/opencl/cl_caller.h"
#include "lite/backends/opencl/cl_context.h"
#include "lite/backends/opencl/cl_image.h"
#include "lite/backends/opencl/cl_runtime.h"
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/tensor.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {

TEST(cl_test, runtime_test) {
  auto *runtime = CLRuntime::Global();
  CHECK(runtime->IsInitSuccess());
  runtime->platform();
  runtime->device();
  runtime->command_queue();
  auto &context = runtime->context();
  auto program = runtime->CreateProgramFromSource(
      context, "buffer/elementwise_add_kernel.cl");
  auto event = runtime->CreateEvent(context);
  const std::string build_option("-DCL_DTYPE_float");
  CHECK(runtime->BuildProgram(program.get(), build_option));
}

TEST(cl_test, context_test) {
  auto *runtime = CLRuntime::Global();
  CHECK(runtime->IsInitSuccess());
  CLContext context;
  context.AddKernel("pool_max", "image/pool_kernel.cl", "-DCL_DTYPE_float");
  context.AddKernel(
      "elementwise_add", "image/elementwise_add_kernel.cl", "-DCL_DTYPE_float");
  context.AddKernel(
      "elementwise_add", "image/elementwise_add_kernel.cl", "-DCL_DTYPE_float");
}

TEST(cl_test, kernel_test) {
  auto *runtime = CLRuntime::Global();
  CHECK(runtime->IsInitSuccess());
  std::unique_ptr<CLContext> context(new CLContext);
  context->AddKernel(
      "elementwise_add", "image/elementwise_add_kernel.cl", "-DCL_DTYPE_float");
  context->AddKernel("pool_max", "image/pool_kernel.cl", "-DCL_DTYPE_float");
  context->AddKernel(
      "elementwise_add", "image/elementwise_add_kernel.cl", "-DCL_DTYPE_float");
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
  status = context->GetCommandQueue().enqueueNDRangeKernel(
      kernel, cl::NullRange, global_work_size, cl::NullRange, nullptr, nullptr);
  CL_CHECK_FATAL(status);
  status = context->GetCommandQueue().finish();
  CL_CHECK_FATAL(status);
#if 0
  double start_nanos = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  double stop_nanos = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  double elapsed_micros = (stop_nanos - start_nanos) / 1000.0;
  LOG(INFO) << "Kernel Run Cost Time: " << elapsed_micros << " us.";
#endif

  LOG(INFO) << out_image;
}

TEST(cl_test, target_wrapper_buffer_test) {
  bool inited = InitOpenCLRuntime();
  CHECK(inited) << "Fail to initialize OpenCL runtime.";
  std::unique_ptr<CLContext> context(new CLContext);
  std::string kernel_name = "elementwise_add";
  std::string build_options = "-DCL_DTYPE_float";
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
  const size_t cl_image2d_width = 28;
  const size_t cl_image2d_height = 32;
  const size_t cl_image2d_elem_size =
      cl_image2d_width * cl_image2d_height * 4;  // 4 for RGBA channels
  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  auto *d_image = static_cast<cl::Image2D *>(
      TargetWrapperCL::MallocImage<float>(cl_image2d_width, cl_image2d_height));

  // Map/Unmap test
  auto *h_image =
      static_cast<float *>(TargetWrapperCL::MapImage(d_image,
                                                     cl_image2d_width,
                                                     cl_image2d_height,
                                                     cl_image2d_row_pitch,
                                                     cl_image2d_slice_pitch));
  CHECK_EQ(cl_image2d_slice_pitch, 0);
  LOG(INFO) << "cl_image2d_row_pitch = " << cl_image2d_row_pitch
            << ", cl_image2d_slice_pitch " << cl_image2d_slice_pitch;

  for (int i = 0; i < cl_image2d_elem_size; i++) {
    h_image[i] = 3.14f * i;
  }
  TargetWrapperCL::Unmap(d_image, h_image);

  auto *h_ptr =
      static_cast<float *>(TargetWrapperCL::MapImage(d_image,
                                                     cl_image2d_width,
                                                     cl_image2d_height,
                                                     cl_image2d_row_pitch,
                                                     cl_image2d_slice_pitch));
  for (int i = 0; i < cl_image2d_elem_size; i++) {
    EXPECT_NEAR(h_ptr[i], 3.14f * i, 1e-6);
  }
  TargetWrapperCL::Unmap(d_image, h_ptr);

  // Imagecpy test
  std::vector<float> h_image_cpy(cl_image2d_elem_size);
  for (int i = 0; i < cl_image2d_elem_size; i++) {
    h_image_cpy[i] = 3.14f;
  }
  TargetWrapperCL::ImgcpySync(d_image,
                              h_image_cpy.data(),
                              cl_image2d_width,
                              cl_image2d_height,
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::HtoD);
  auto *d_image_cpy = static_cast<cl::Image2D *>(
      TargetWrapperCL::MallocImage<float>(cl_image2d_width, cl_image2d_height));

  // device to device
  TargetWrapperCL::ImgcpySync(d_image_cpy,
                              d_image,
                              cl_image2d_width,
                              cl_image2d_height,
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoD);
  std::fill(h_image_cpy.begin(), h_image_cpy.end(), 0);

  // host to device
  TargetWrapperCL::ImgcpySync(h_image_cpy.data(),
                              d_image_cpy,
                              cl_image2d_width,
                              cl_image2d_height,
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  for (int i = 0; i < cl_image2d_elem_size; i++) {
    EXPECT_NEAR(h_image_cpy[i], 3.14f, 1e-6);
  }

  TargetWrapperCL::FreeImage(d_image_cpy);
  TargetWrapperCL::FreeImage(d_image);
}

}  // namespace lite
}  // namespace paddle
