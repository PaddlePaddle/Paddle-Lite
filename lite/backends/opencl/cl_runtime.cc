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
    command_queue_->flush();
    command_queue_->finish();
  }
  // For controlling the destruction order
  command_queue_.reset();
  context_.reset();
  device_.reset();
  platform_.reset();
  device_info_.clear();
}

bool CLRuntime::Init() {
  if (initialized_) {
    return true;
  }
  bool is_platform_init = InitializePlatform();
  bool is_device_init = InitializeDevice();
  is_init_success_ = is_platform_init && is_device_init;
  initialized_ = true;

  context_ = CreateContext();
  command_queue_ = CreateCommandQueue(context());
  return initialized_;
}

cl::Platform& CLRuntime::platform() {
  CHECK(platform_ != nullptr) << "platform_ is not initialized!";
  return *platform_;
}

cl::Context& CLRuntime::context() {
  if (context_ == nullptr) {
    LOG(FATAL) << "context_ create failed. ";
  }
  return *context_;
}

cl::Device& CLRuntime::device() {
  CHECK(device_ != nullptr) << "device_ is not initialized!";
  return *device_;
}

cl::CommandQueue& CLRuntime::command_queue() {
  if (command_queue_ == nullptr) {
    LOG(FATAL) << "command_queue_ create failed. ";
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
  std::string build_option = options + " -cl-fast-relaxed-math -cl-mad-enable";
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

GpuType CLRuntime::ParseGpuTypeFromDeviceName(std::string device_name) {
  const std::string kMALI_PATTERN_STR = "Mali";
  const std::string kADRENO_PATTERN_STR = "QUALCOMM Adreno(TM)";
  const std::string kPOWERVR_PATTERN_STR = "PowerVR";

  if (device_name == kADRENO_PATTERN_STR) {
    LOG(INFO) << "adreno gpu";
    return GpuType::QUALCOMM_ADRENO;
  } else if (device_name.find(kMALI_PATTERN_STR) != std::string::npos) {
    LOG(INFO) << "mali gpu";
    return GpuType::ARM_MALI;
  } else if (device_name.find(kPOWERVR_PATTERN_STR) != std::string::npos) {
    LOG(INFO) << "powerVR gpu";
    return GpuType::IMAGINATION_POWERVR;
  } else {
    LOG(INFO) << "others gpu";
    return GpuType::UNKNOWN;
  }
}

bool CLRuntime::InitializeDevice() {
  // ===================== BASIC =====================
  // CL_DEVICE_TYPE_GPU
  // CL_DEVICE_NAME
  // CL_DEVICE_SUPPORT
  // CL_DEVICE_MAX_COMPUTE_UNITS
  // CL_DEVICE_MAX_CLOCK_FREQUENCY
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
  gpu_type_ = ParseGpuTypeFromDeviceName(device_name);

  cl_device_type device_type = device_->getInfo<CL_DEVICE_TYPE>();
  auto device_type_to_str = [](cl_device_type t) -> std::string {
    std::string t_str{""};
    switch (t) {
      case CL_DEVICE_TYPE_CPU:
        t_str = "CPU";
        break;
      case CL_DEVICE_TYPE_GPU:
        t_str = "GPU";
        break;
      case CL_DEVICE_TYPE_ACCELERATOR:
        t_str = "Accelerator";
        break;
      case CL_DEVICE_TYPE_DEFAULT:
        t_str = "Default";
        break;
      default:
        t_str = "Unknown";
    }
    return t_str;
  };
  LOG(INFO) << "device_type:" << device_type_to_str(device_type);
  device_info_["CL_DEVICE_TYPE"] = device_type;

  auto max_units = device_->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  LOG(INFO) << "The chosen device has " << max_units << " compute units.";
  device_info_["CL_DEVICE_MAX_COMPUTE_UNITS"] = max_units;

  auto max_clock_freq = device_->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
  LOG(INFO) << "CL_DEVICE_MAX_CLOCK_FREQUENCY:" << max_clock_freq;
  device_info_["CL_DEVICE_MAX_CLOCK_FREQUENCY"] = max_clock_freq;

  // ===================== MEMORY =====================
  // CL_DEVICE_LOCAL_MEM_SIZE
  // CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
  // CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
  // CL_DEVICE_GLOBAL_MEM_SIZE
  auto local_mem_kb =
      static_cast<float>(device_->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()) / 1024;
  LOG(INFO) << "The local memory size of the chosen device is " << local_mem_kb
            << " KB.";
  device_info_["CL_DEVICE_LOCAL_MEM_SIZE_KB"] = local_mem_kb;

  auto global_mem_cache_size_kb =
      static_cast<float>(device_->getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>()) /
      1024;
  LOG(INFO) << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE(KB):"
            << global_mem_cache_size_kb << " KB.";
  device_info_["CL_DEVICE_GLOBAL_MEM_CACHE_SIZE_KB"] = global_mem_cache_size_kb;

  auto global_mem_cacheline_size_kb =
      static_cast<float>(
          device_->getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>()) /
      1024;
  LOG(INFO) << "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE(KB):"
            << global_mem_cacheline_size_kb << " KB.";
  device_info_["CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE_KB"] =
      global_mem_cacheline_size_kb;

  auto global_mem_size_kb =
      static_cast<float>(device_->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()) / 1024;
  LOG(INFO) << "CL_DEVICE_GLOBAL_MEM_SIZE(KB):" << global_mem_size_kb << " KB.";
  device_info_["CL_DEVICE_GLOBAL_MEM_SIZE_KB"] = global_mem_size_kb;

  // ===================== WORK_GROUP =====================
  // CL_DEVICE_MAX_WORK_GROUP_SIZE
  // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  // CL_DEVICE_MAX_WORK_ITEM_SIZES
  auto max_work_group_size = device_->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  LOG(INFO) << "CL_DEVICE_MAX_WORK_GROUP_SIZE:" << max_work_group_size;
  device_info_["CL_DEVICE_MAX_WORK_GROUP_SIZE"] = max_work_group_size;

  auto max_dims_num = device_->getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
  LOG(INFO) << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:" << max_dims_num;
  device_info_["CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS"] = max_dims_num;

  auto max_work_item_sizes = device_->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  for (size_t i = 0; i < max_work_item_sizes.size(); ++i) {
    LOG(INFO) << "max_work_item_sizes[" << i << "]:" << max_work_item_sizes[i];
    std::string dim_key = "CL_DEVICE_MAX_WORK_ITEM_SIZES_" + std::to_string(i);
    device_info_[dim_key] = max_work_item_sizes[i];
  }

  // ===================== BUFFER =====================
  // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
  auto max_constant_buffer_size_kb =
      static_cast<float>(
          device_->getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>()) /
      1024;
  LOG(INFO) << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:"
            << max_constant_buffer_size_kb;
  device_info_["CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE"] =
      max_constant_buffer_size_kb;

  // ===================== IMAGE =====================
  // CL_DEVICE_IMAGE_SUPPORT
  // CL_DEVICE_IMAGE2D_MAX_HEIGHT
  // CL_DEVICE_IMAGE2D_MAX_WIDTH
  auto image_support = device_->getInfo<CL_DEVICE_IMAGE_SUPPORT>();
  if (image_support) {
    LOG(INFO) << "The chosen device supports image processing.";
    device_info_["CL_DEVICE_IMAGE_SUPPORT"] = 1;
  } else {
    LOG(INFO) << "The chosen device doesn't support image processing!";
    device_info_["CL_DEVICE_IMAGE_SUPPORT"] = 0;
    return false;
  }

  auto image2d_max_height = device_->getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();
  LOG(INFO) << "CL_DEVICE_IMAGE2D_MAX_HEIGHT:" << image2d_max_height;
  device_info_["CL_DEVICE_IMAGE2D_MAX_HEIGHT"] = image2d_max_height;

  auto image2d_max_width = device_->getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
  LOG(INFO) << "CL_DEVICE_IMAGE2D_MAX_WIDTH:" << image2d_max_width;
  device_info_["CL_DEVICE_IMAGE2D_MAX_WIDTH"] = image2d_max_width;

  // ===================== OTHERS / EXTENSION / VERSION =====================
  // CL_DEVICE_EXTENSIONS
  // CL_DEVICE_ADDRESS_BITS
  auto ext_data = device_->getInfo<CL_DEVICE_EXTENSIONS>();
  VLOG(4) << "The extensions supported by this device: " << ext_data;
  if (ext_data.find("cl_khr_fp16") != std::string::npos) {
    LOG(INFO) << "The chosen device supports the half data type.";
    device_info_["CL_DEVICE_EXTENSIONS_FP16"] = 1;
  } else {
    LOG(INFO) << "The chosen device doesn't support the half data type!";
    device_info_["CL_DEVICE_EXTENSIONS_FP16"] = 0;
  }

  auto address_bits = device_->getInfo<CL_DEVICE_ADDRESS_BITS>();
  LOG(INFO) << "CL_DEVICE_ADDRESS_BITS:" << address_bits;
  device_info_["CL_DEVICE_ADDRESS_BITS"] = address_bits;

  auto driver_version = device_->getInfo<CL_DRIVER_VERSION>();
  LOG(INFO) << "CL_DRIVER_VERSION:" << driver_version;

  return true;
}

std::map<std::string, size_t>& CLRuntime::GetDeviceInfo() {
  if (0 != device_info_.size()) {
    return device_info_;
  }
  InitializeDevice();
  return device_info_;
}

void CLRuntime::GetAdrenoContextProperties(
    std::vector<cl_context_properties>* properties,
    GPUPerfMode gpu_perf_mode,
    GPUPriorityLevel gpu_priority_level) {
  CHECK(properties) << "cl_context_properties is nullptr";
  properties->reserve(5);
  switch (gpu_perf_mode) {
    case GPUPerfMode::PERF_LOW:
      LOG(INFO) << "GPUPerfMode::PERF_LOW";
      properties->push_back(CL_CONTEXT_PERF_MODE_QCOM);
      properties->push_back(CL_PERF_MODE_LOW_QCOM);
      break;
    case GPUPerfMode::PERF_NORMAL:
      LOG(INFO) << "GPUPerfMode::PERF_NORMAL";
      properties->push_back(CL_CONTEXT_PERF_MODE_QCOM);
      properties->push_back(CL_PERF_MODE_NORMAL_QCOM);
      break;
    case GPUPerfMode::PERF_HIGH:
      LOG(INFO) << "GPUPerfMode::PERF_HIGH";
      properties->push_back(CL_CONTEXT_PERF_MODE_QCOM);
      properties->push_back(CL_PERF_MODE_HIGH_QCOM);
      break;
    default:
      break;
  }
  switch (gpu_priority_level) {
    case GPUPriorityLevel::PRIORITY_LOW:
      LOG(INFO) << "GPUPriorityLevel::PRIORITY_LOW";
      properties->push_back(CL_CONTEXT_PRIORITY_LEVEL_QCOM);
      properties->push_back(CL_PRIORITY_HINT_LOW_QCOM);
      break;
    case GPUPriorityLevel::PRIORITY_NORMAL:
      LOG(INFO) << "GPUPriorityLevel::PRIORITY_NORMAL";
      properties->push_back(CL_CONTEXT_PRIORITY_LEVEL_QCOM);
      properties->push_back(CL_PRIORITY_HINT_NORMAL_QCOM);
      break;
    case GPUPriorityLevel::PRIORITY_HIGH:
      LOG(INFO) << "GPUPriorityLevel::PRIORITY_HIGH";
      properties->push_back(CL_CONTEXT_PRIORITY_LEVEL_QCOM);
      properties->push_back(CL_PRIORITY_HINT_HIGH_QCOM);
      break;
    default:
      break;
  }
  // The properties list should be terminated with 0
  properties->push_back(0);
}

}  // namespace lite
}  // namespace paddle
