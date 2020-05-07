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

#pragma once

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/opencl/cl_include.h"
#include "lite/backends/opencl/cl_utility.h"

typedef enum {
  UNKNOWN = 0,
  QUALCOMM_ADRENO = 1,
  ARM_MALI = 2,
  IMAGINATION_POWERVR = 3,
  OTHERS = 4,
} GpuType;

typedef enum {
  PERF_DEFAULT = 0,
  PERF_LOW = 1,
  PERF_NORMAL = 2,
  PERF_HIGH = 3
} GPUPerfMode;

typedef enum {
  PRIORITY_DEFAULT = 0,
  PRIORITY_LOW = 1,
  PRIORITY_NORMAL = 2,
  PRIORITY_HIGH = 3
} GPUPriorityLevel;

// Adreno extensions
// Adreno performance hints
typedef cl_uint cl_perf_hint;
#define CL_CONTEXT_PERF_MODE_QCOM 0x40C2
#define CL_PERF_MODE_HIGH_QCOM 0x40C3
#define CL_PERF_MODE_NORMAL_QCOM 0x40C4
#define CL_PERF_MODE_LOW_QCOM 0x40C5

// Adreno priority hints
typedef cl_uint cl_priority_hint;

#define CL_PRIORITY_HINT_NONE_QCOM 0
#define CL_CONTEXT_PRIORITY_LEVEL_QCOM 0x40C9
#define CL_PRIORITY_HINT_HIGH_QCOM 0x40CA
#define CL_PRIORITY_HINT_NORMAL_QCOM 0x40CB
#define CL_PRIORITY_HINT_LOW_QCOM 0x40CC

namespace paddle {
namespace lite {

extern const std::map<std::string, std::vector<unsigned char>>
    opencl_kernels_files;

class CLRuntime {
 public:
  static CLRuntime* Global();

  bool Init();

  cl::Platform& platform();

  cl::Context& context();

  cl::Device& device();

  cl::CommandQueue& command_queue();

  std::unique_ptr<cl::Program> CreateProgram(const cl::Context& context,
                                             std::string file_name);

  std::unique_ptr<cl::UserEvent> CreateEvent(const cl::Context& context);

  bool BuildProgram(cl::Program* program, const std::string& options = "");

  bool IsInitSuccess() { return is_init_success_; }

  std::string cl_path() { return cl_path_; }

  void set_cl_path(std::string cl_path) { cl_path_ = cl_path; }

  std::map<std::string, size_t>& GetDeviceInfo();

 private:
  CLRuntime() { Init(); }

  ~CLRuntime();

  bool InitializePlatform();

  bool InitializeDevice();

  void GetAdrenoContextProperties(
      std::vector<cl_context_properties>* properties,
      GPUPerfMode gpu_perf_mode,
      GPUPriorityLevel gpu_priority_level);

  std::shared_ptr<cl::Context> CreateContext() {
    // note(ysh329): gpu perf mode and priority level of adreno gpu referred
    // from xiaomi/mace.
    // However, no performance gain after `PERF_HIGH` and `PRIORITY_HIGH` set.
    auto perf_mode = GPUPerfMode::PERF_HIGH;
    auto priority_level = GPUPriorityLevel::PRIORITY_HIGH;
    std::vector<cl_context_properties> context_properties;
    if (gpu_type_ == GpuType::QUALCOMM_ADRENO) {
      GetAdrenoContextProperties(
          &context_properties, perf_mode, priority_level);
    }
    auto context =
        std::make_shared<cl::Context>(std::vector<cl::Device>{device()},
                                      context_properties.data(),
                                      nullptr,
                                      nullptr,
                                      &status_);
    CL_CHECK_FATAL(status_);
    return context;
  }

  std::shared_ptr<cl::CommandQueue> CreateCommandQueue(
      const cl::Context& context) {
    cl_command_queue_properties properties = 0;

#ifdef LITE_WITH_PROFILE
    properties |= CL_QUEUE_PROFILING_ENABLE;
#endif  // LITE_WITH_PROFILE
    auto queue = std::make_shared<cl::CommandQueue>(
        context, device(), properties, &status_);
    CL_CHECK_FATAL(status_);
    return queue;
  }

  GpuType ParseGpuTypeFromDeviceName(std::string device_name);

  std::map<std::string, size_t> device_info_;

  GpuType gpu_type_{GpuType::UNKNOWN};

  std::string cl_path_;

  std::shared_ptr<cl::Platform> platform_{nullptr};

  std::shared_ptr<cl::Context> context_{nullptr};

  std::shared_ptr<cl::Device> device_{nullptr};

  std::shared_ptr<cl::CommandQueue> command_queue_{nullptr};

  cl_int status_{CL_SUCCESS};

  bool initialized_{false};

  bool is_init_success_{false};
};

}  // namespace lite
}  // namespace paddle
