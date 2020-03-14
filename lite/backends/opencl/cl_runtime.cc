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

#include "lite/backends/opencl/cl_runtime.h"
#include <string>
#include <utility>
#include <vector>
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

CLRuntime* CLRuntime::Global() {
  static CLRuntime cl_runtime_;
  cl_runtime_.Init();
  return &cl_runtime_;
}

CLRuntime::~CLRuntime() {
  if (command_queue_ != nullptr) {
    command_queue_->finish();
  }
  // For controlling the destruction order:
  command_queue_.reset();
  context_.reset();
  device_.reset();
  platform_.reset();
}

bool CLRuntime::Init() {
  if (initialized_) {
    return true;
  }
  bool is_platform_init = InitializePlatform();
  bool is_device_init = InitializeDevice();
  is_init_success_ = is_platform_init && is_device_init;
  initialized_ = true;
  return initialized_;
}

cl::Platform& CLRuntime::platform() {
  CHECK(platform_ != nullptr) << "platform_ is not initialized!";
  return *platform_;
}

cl::Context& CLRuntime::context() {
  if (context_ == nullptr) {
    context_ = CreateContext();
  }
  return *context_;
}

cl::Device& CLRuntime::device() {
  CHECK(device_ != nullptr) << "device_ is not initialized!";
  return *device_;
}

cl::CommandQueue& CLRuntime::command_queue() {
  if (command_queue_ == nullptr) {
    command_queue_ = CreateCommandQueue(context());
  }
  return *command_queue_;
}

std::unique_ptr<cl::Program> CLRuntime::CreateProgram(
    const cl::Context& context, std::string file_name) {
  auto cl_file = opencl_kernels_files.find(file_name);
  std::string content(cl_file->second.begin(), cl_file->second.end());
  cl::Program::Sources sources;
  sources.push_back(content);
  auto prog =
      std::unique_ptr<cl::Program>(new cl::Program(context, sources, &status_));
  VLOG(4) << "OpenCL kernel file name: " << file_name;
  VLOG(4) << "Program source size: " << content.size();
  CL_CHECK_FATAL(status_);
  return std::move(prog);
}

std::unique_ptr<cl::UserEvent> CLRuntime::CreateEvent(
    const cl::Context& context) {
  auto event =
      std::unique_ptr<cl::UserEvent>(new cl::UserEvent(context, &status_));
  CL_CHECK_FATAL(status_);
  return std::move(event);
}

bool CLRuntime::BuildProgram(cl::Program* program, const std::string& options) {
  /* -I +CLRuntime::Global()->cl_path() + "/cl_kernel"*/
  std::string build_option = options + " -cl-fast-relaxed-math ";
  VLOG(4) << "OpenCL build_option: " << build_option;
  status_ = program->build({*device_}, build_option.c_str());
  CL_CHECK_ERROR(status_);

  if (status_ != CL_SUCCESS) {
    if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device()) ==
        CL_BUILD_ERROR) {
      std::string log = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device());
      LOG(FATAL) << "Program build error: " << log;
    }
    return false;
  }

  return true;
}

bool CLRuntime::InitializePlatform() {
  std::vector<cl::Platform> all_platforms;
  status_ = cl::Platform::get(&all_platforms);
  CL_CHECK_ERROR(status_);
  if (all_platforms.empty()) {
    LOG(FATAL) << "No OpenCL platform found!";
    return false;
  }
  platform_ = std::make_shared<cl::Platform>();
  *platform_ = all_platforms[0];
  return true;
}

bool CLRuntime::InitializeDevice() {
  std::vector<cl::Device> all_devices;
  status_ = platform_->getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
  CL_CHECK_ERROR(status_);
  if (all_devices.empty()) {
    LOG(FATAL) << "No OpenCL GPU device found!";
    return false;
  }
  device_ = std::make_shared<cl::Device>();
  *device_ = all_devices[0];

  auto device_name = device_->getInfo<CL_DEVICE_NAME>();
  LOG(INFO) << "Using device: " << device_name;
  auto image_support = device_->getInfo<CL_DEVICE_IMAGE_SUPPORT>();
  if (image_support) {
    LOG(INFO) << "The chosen device supports image processing.";
  } else {
    LOG(INFO) << "The chosen device doesn't support image processing!";
    return false;
  }
  auto ext_data = device_->getInfo<CL_DEVICE_EXTENSIONS>();
  VLOG(4) << "The extensions supported by this device: " << ext_data;
  if (ext_data.find("cl_khr_fp16") != std::string::npos) {
    LOG(INFO) << "The chosen device supports the half data type.";
  } else {
    LOG(INFO) << "The chosen device doesn't support the half data type!";
  }
  auto max_units = device_->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  LOG(INFO) << "The chosen device has " << max_units << " compute units.";
  auto local_mem = device_->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
  LOG(INFO) << "The local memory size of the chosen device is "
            << static_cast<float>(local_mem) / 1024 << " KB.";
  return true;
}

}  // namespace lite
}  // namespace paddle
