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

#include "lite/backends/opencl/cl_context.h"
#include <memory>
#include <string>
#include <utility>
#include "lite/backends/opencl/cl_runtime.h"
#include "lite/backends/opencl/cl_utility.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {

cl::CommandQueue &CLContext::GetCommandQueue() {
  return CLRuntime::Global()->command_queue();
}

cl::Context &CLContext::GetContext() { return CLRuntime::Global()->context(); }

cl::Program &CLContext::GetProgram(const std::string &file_name,
                                   const std::string &options) {
  STL::stringstream program_key_ss;
  program_key_ss << file_name << options;
  std::string program_key = program_key_ss.str();
  auto it = programs_.find(program_key);
  if (it != programs_.end()) {
    VLOG(3) << " --- program -> " << program_key << " has been built --- ";
    return *(it->second);
  }

  auto program = CLRuntime::Global()->CreateProgram(
      GetContext(), CLRuntime::Global()->cl_path() + "/cl_kernel/" + file_name);

  VLOG(3) << " --- begin build program -> " << program_key << " --- ";
  CLRuntime::Global()->BuildProgram(program.get(), options);
  VLOG(3) << " --- end build program -> " << program_key << " --- ";

  programs_[program_key] = std::move(program);

  return *(programs_[program_key]);
}

void CLContext::AddKernel(const std::string &kernel_name,
                          const std::string &file_name,
                          const std::string &options) {
  cl_int status{CL_SUCCESS};
  VLOG(3) << " --- to get program " << file_name << " --- ";
  auto program = GetProgram(file_name, options);
  VLOG(3) << " --- end get program --- ";
  VLOG(3) << " --- to create kernel: " << kernel_name << " --- ";
  std::unique_ptr<cl::Kernel> kernel(
      new cl::Kernel(program, kernel_name.c_str(), &status));
  CL_CHECK_FATAL(status);
  VLOG(3) << " --- end create kernel --- ";
  kernels_.emplace_back(std::move(kernel));
  STL::stringstream kernel_key;
  kernel_key << kernel_name << options;
  kernel_offset_[kernel_key.str()] = kernels_.size() - 1;
}

cl::Kernel &CLContext::GetKernel(const int index) {
  VLOG(3) << " --- kernel count: " << kernels_.size() << " --- ";
  CHECK(static_cast<size_t>(index) < kernels_.size())
      << "The index must be less than the size of kernels.";
  CHECK(kernels_[index] != nullptr)
      << "The target kernel pointer cannot be null.";
  return *(kernels_[index]);
}

cl::Kernel &CLContext::GetKernel(const std::string &name) {
  auto it = kernel_offset_.find(name);
  CHECK(it != kernel_offset_.end()) << "Cannot find the kernel function: "
                                    << name;
  return GetKernel(it->second);
}

cl::NDRange CLContext::DefaultWorkSize(const CLImage &image) {
  // n c h w
  auto image_dim = image.tensor_dims();
  if (image_dim.size() == 4) {
    auto n = image_dim[0];
    auto h = image_dim[2];
    auto w = image_dim[3];
    auto image_width = image.ImageWidth();
    auto work_size_0 = image_width / w;
    auto work_size_1 = w;
    auto work_size_2 = n * h;
    return cl::NDRange{static_cast<size_t>(work_size_0),
                       static_cast<size_t>(work_size_1),
                       static_cast<size_t>(work_size_2)};
  } else if (image_dim.size() == 2) {
    return cl::NDRange{static_cast<size_t>(1),
                       static_cast<size_t>(image.ImageWidth()),
                       static_cast<size_t>(image.ImageHeight())};
  } else if (image_dim.size() == 1) {
    return cl::NDRange{static_cast<size_t>(1),
                       static_cast<size_t>(image.ImageWidth()),
                       static_cast<size_t>(1)};
  } else if (image_dim.size() == 3) {
    auto c = image_dim[0];
    auto h = image_dim[1];
    auto w = image_dim[2];
    return cl::NDRange{static_cast<size_t>((c + 3) / 4),
                       static_cast<size_t>(w),
                       static_cast<size_t>(h)};
  } else {
    LOG(FATAL) << "Not support this dimension, need to be implemented!";
    return cl::NDRange{};
  }
}

}  // namespace lite
}  // namespace paddle
