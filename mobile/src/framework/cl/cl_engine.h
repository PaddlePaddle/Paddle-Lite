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

#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "CL/cl.h"
#include "common/enforce.h"
#include "common/log.h"
#include "framework/cl/cl_deleter.h"
#include "framework/cl/cl_tool.h"

namespace paddle_mobile {
namespace framework {

class CLLocalWorkSizeInfo {
 public:
  CLLocalWorkSizeInfo() {
    max_work_group_size = 0;
    max_work_item_size0 = 0;
    max_work_item_size1 = 0;
    max_work_item_size2 = 0;
  }
  CLLocalWorkSizeInfo(size_t total_size, size_t size0, size_t size1,
                      size_t size2) {
    max_work_group_size = total_size;
    max_work_item_size0 = size0;
    max_work_item_size1 = size1;
    max_work_item_size2 = size2;
  }
  bool isEmpty() {
    return max_work_group_size == 0 && max_work_item_size0 == 0 &&
           max_work_item_size1 == 0 && max_work_item_size2 == 0;
  }

  // max total number of work-items in the work-group
  size_t max_work_group_size;
  // max number of work-items in local_work_size in dim 0
  size_t max_work_item_size0;
  // max number of work-items in local_work_size in dim 1
  size_t max_work_item_size1;
  // max number of work-items in local_work_size in dim 2
  size_t max_work_item_size2;
};
inline void ctx_info(const char *errinfo, const void *private_info, size_t cb,
                     void *user_data) {
  fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}
class CLEngine {
 public:
  static CLEngine *Instance();

  bool Init();
  bool isInitSuccess();

  std::shared_ptr<_cl_context> CreateContext() {
    DLOG << "CreateContext ---";
    DLOG << "platform: " << platform_;
    DLOG << "devices_[0]: " << devices_[0];

    cl_int status;
    cl_context c = clCreateContext(NULL, 1, devices_, &ctx_info, NULL, &status);
    std::shared_ptr<_cl_context> context(c, CLContextDeleter());
    CL_CHECK_ERRORS(status);
    return std::move(context);
  }

  std::unique_ptr<_cl_command_queue, CLCommQueueDeleter> CreateClCommandQueue(
      cl_context context) {
    cl_int status;
    cl_command_queue queue =
        clCreateCommandQueue(context, devices_[0], 0, &status);
    std::unique_ptr<_cl_command_queue, CLCommQueueDeleter> command_queue_ptr(
        queue);
    CL_CHECK_ERRORS(status);
    return std::move(command_queue_ptr);
  }

  cl_context getContext() {
    if (context_.get() == nullptr) {
      context_ = CreateContext();
    }
    return context_.get();
  }

  cl_command_queue getClCommandQueue() {
    if (command_queue_.get() == nullptr) {
      command_queue_ = CreateClCommandQueue(getContext());
    }
    return command_queue_.get();
  }

  CLLocalWorkSizeInfo getLocalWorkSizeInfo() {
    if (!localWorkSizeInfo_.isEmpty()) {
      return localWorkSizeInfo_;
    }
    cl_int status;
    size_t max_work_group_size = 0;
    status = clGetDeviceInfo(devices_[0], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                             sizeof(size_t), &max_work_group_size, NULL);
    if (status != CL_SUCCESS) {
      return CLLocalWorkSizeInfo(0, 0, 0, 0);
    }
    cl_uint max_dims_num = 0;
    status = clGetDeviceInfo(devices_[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                             sizeof(cl_uint), &max_dims_num, NULL);
    if (status != CL_SUCCESS) {
      return CLLocalWorkSizeInfo(0, 0, 0, 0);
    }
    DLOG << "max_work_item_sizes max_dims_num: " << max_dims_num;
    size_t *max_work_item_sizes =
        reinterpret_cast<size_t *>(calloc(max_dims_num, sizeof(size_t)));
    size_t ret_size = 0;
    status = clGetDeviceInfo(devices_[0], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                             max_dims_num * sizeof(size_t), max_work_item_sizes,
                             &ret_size);
    if (status != CL_SUCCESS || ret_size / sizeof(size_t) < 3) {
      return CLLocalWorkSizeInfo(0, 0, 0, 0);
    }
    DLOG << " max_work_item_sizes {" << max_work_item_sizes[0] << ", "
         << max_work_item_sizes[1] << ", " << max_work_item_sizes[2] << "}";

    localWorkSizeInfo_ =
        CLLocalWorkSizeInfo(max_work_group_size, max_work_item_sizes[0],
                            max_work_item_sizes[1], max_work_item_sizes[2]);
    free(max_work_item_sizes);
    return localWorkSizeInfo_;
  }
  size_t GetKernelWorkSize(cl_kernel kernel) {
    cl_int status;
    size_t kernel_work_size = 0;
    status =
        clGetKernelWorkGroupInfo(kernel, devices_[0], CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(size_t), &kernel_work_size, NULL);
    if (status != CL_SUCCESS) {
      return 0;
    }
    DLOG << "kernel_work_size: " << kernel_work_size;
    return kernel_work_size;
  }

  std::unique_ptr<_cl_program, CLProgramDeleter> CreateProgramWith(
      cl_context context, std::string file_name) {
    FILE *file = fopen(file_name.c_str(), "rb");
    PADDLE_MOBILE_ENFORCE(file != nullptr, "can't open file: %s ",
                          file_name.c_str());
    fseek(file, 0, SEEK_END);
    int64_t size = ftell(file);
    PADDLE_MOBILE_ENFORCE(size > 0, "size is too small");
    rewind(file);
    char *data = new char[size + 1];
    size_t bytes_read = fread(data, 1, size, file);
    data[size] = '\0';
    PADDLE_MOBILE_ENFORCE(bytes_read == size,
                          "read binary file bytes do not match with fseek");
    fclose(file);

    const char *source = data;
    size_t sourceSize[] = {strlen(source)};
    cl_program p =
        clCreateProgramWithSource(context, 1, &source, sourceSize, &status_);

    DLOG << " cl kernel file name: " << file_name;
    DLOG << " source size: " << sourceSize[0];
    CL_CHECK_ERRORS(status_);

    std::unique_ptr<_cl_program, CLProgramDeleter> program_ptr(p);

    return std::move(program_ptr);
  }

  std::unique_ptr<_cl_program, CLProgramDeleter> CreateProgramWithSource(
      cl_context context, const char *source) {
    size_t sourceSize[] = {strlen(source)};
    cl_program p =
        clCreateProgramWithSource(context, 1, &source, sourceSize, &status_);

    LOG(kLOG_DEBUG4) << " cl kernel from source";
    LOG(kLOG_DEBUG4) << " source size: " << sourceSize[0];
    CL_CHECK_ERRORS(status_);

    std::unique_ptr<_cl_program, CLProgramDeleter> program_ptr(p);

    return std::move(program_ptr);
  }

  std::unique_ptr<_cl_event, CLEventDeleter> CreateEvent(cl_context context) {
    cl_event event = clCreateUserEvent(context, &status_);
    std::unique_ptr<_cl_event, CLEventDeleter> event_ptr(event);
    CL_CHECK_ERRORS(status_);
    return std::move(event_ptr);
  }

  bool BuildProgram(cl_program program, const std::string &options = "") {
    cl_int status;
    std::string path = options + " -cl-fast-relaxed-math";

    status = clBuildProgram(program, 0, 0, path.c_str(), 0, 0);

    CL_CHECK_ERRORS(status);

    if (status == CL_BUILD_PROGRAM_FAILURE) {
      size_t log_size;
      clGetProgramBuildInfo(program, CLEngine::Instance()->DeviceID(),
                            CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      char *log = reinterpret_cast<char *>(malloc(log_size));
      clGetProgramBuildInfo(program, CLEngine::Instance()->DeviceID(),
                            CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
      DLOG << " program build error: " << log;
    }

    return status == CL_SUCCESS;
  }

  cl_device_id DeviceID(int index = 0) { return devices_[index]; }

  std::string GetCLPath() { return cl_path_; }
  void setClPath(std::string cl_path) { cl_path_ = cl_path; }

 private:
  CLEngine() { initialized_ = false; }

  bool SetPlatform();

  bool SetClDeviceId();

  bool initialized_;

  CLLocalWorkSizeInfo localWorkSizeInfo_;

  cl_int status_;
  std::string cl_path_;
  bool is_init_success_ = false;
  std::unique_ptr<_cl_command_queue, CLCommQueueDeleter> command_queue_;
  std::shared_ptr<_cl_context> context_;
  cl_device_id devices_[10];
  cl_platform_id platform_;
};

}  // namespace framework
}  // namespace paddle_mobile
