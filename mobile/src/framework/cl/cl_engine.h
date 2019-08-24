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

#include "CL/cl.h"
#include "common/enforce.h"
#include "common/log.h"
#include "framework/cl/cl_deleter.h"
#include "framework/cl/cl_tool.h"

namespace paddle_mobile {
namespace framework {

class CLEngine {
 public:
  static CLEngine *Instance();

  bool Init();
  bool isInitSuccess();
  std::unique_ptr<_cl_context, CLContextDeleter> CreateContext() {
    cl_int status;
    cl_context c = clCreateContext(NULL, 1, devices_, NULL, NULL, &status);
    std::unique_ptr<_cl_context, CLContextDeleter> context_ptr(c);
    CL_CHECK_ERRORS(status);
    return std::move(context_ptr);
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
    if (context_ == nullptr) {
      context_ = CreateContext();
    }
    return context_.get();
  }

  cl_command_queue getClCommandQueue() {
    if (command_queue_ == nullptr) {
      command_queue_ = CreateClCommandQueue(getContext());
    }
    return command_queue_.get();
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

    DLOG << " cl kernel from source";
    DLOG << " source size: " << sourceSize[0];
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
    std::string path = options + " -cl-fast-relaxed-math -I " +
                       CLEngine::Instance()->GetCLPath() + "/cl_kernel";

    status = clBuildProgram(program, 0, 0, path.c_str(), 0, 0);

    CL_CHECK_ERRORS(status);

    if (status_ == CL_BUILD_PROGRAM_FAILURE) {
      size_t log_size;
      clGetProgramBuildInfo(program, CLEngine::Instance()->DeviceID(),
                            CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      char *log = reinterpret_cast<char *>(malloc(log_size));
      clGetProgramBuildInfo(program, CLEngine::Instance()->DeviceID(),
                            CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
      DLOG << " program build error: " << log;
    }

    if (status == CL_SUCCESS) {
      return true;
    } else {
      return false;
    }
  }

  cl_device_id DeviceID(int index = 0) { return devices_[index]; }

  std::string GetCLPath() { return cl_path_; }
  void setClPath(std::string cl_path) { cl_path_ = cl_path; }

 private:
  CLEngine() { initialized_ = false; }

  bool SetPlatform();

  bool SetClDeviceId();

  bool initialized_;

  cl_platform_id platform_;

  cl_device_id *devices_;

  cl_int status_;

  std::string cl_path_;
  std::unique_ptr<_cl_program, CLProgramDeleter> program_;

  std::unique_ptr<_cl_context, CLContextDeleter> context_ = nullptr;

  std::unique_ptr<_cl_command_queue, CLCommQueueDeleter> command_queue_ =
      nullptr;

  //  bool SetClContext();

  //  bool SetClCommandQueue();

  //  bool LoadKernelFromFile(const char *kernel_file);

  //  bool BuildProgram();
  bool is_init_success_ = false;
};

}  // namespace framework
}  // namespace paddle_mobile
