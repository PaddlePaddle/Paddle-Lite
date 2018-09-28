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

#include <memory>
#include <string>
#include <fstream>

#include "common/enforce.h"
#include "framework/cl/cl_deleter.h"
#include "framework/cl/CL/cl.h"

namespace paddle_mobile {
namespace framework {

class CLEngine {
 public:
  static CLEngine *Instance();

  bool Init();

  std::unique_ptr<_cl_context, CLContextDeleter> CreateContext() {
    cl_context c = clCreateContext(NULL, 1, devices_, NULL, NULL, NULL);
    std::unique_ptr<_cl_context, CLContextDeleter> context_ptr(c);
    return std::move(context_ptr);
  }

  std::unique_ptr<_cl_command_queue, CLCommQueueDeleter> CreateClCommandQueue() {
    cl_int status;
    cl_command_queue = clCreateCommandQueue(context_.get(), devices_[0], 0, &status);
    std::unique_ptr<_cl_command_queue, CLCommQueueDeleter> command_queue_ptr(cl_command_queue);
    return std::move(command_queue_ptr);
  }

  std::unique_ptr<_cl_program, CLProgramDeleter> CreateProgramWith(cl_context context, std::string file_name) {
    const char *kernel_file = file_name.c_str();
    size_t size;
    char *str;
    std::fstream f(kernel_file, (std::fstream::in | std::fstream::binary));

    PADDLE_MOBILE_ENFORCE(f.is_open(), " file open failed")

    size_t fileSize;
    f.seekg(0, std::fstream::end);
    size = fileSize = (size_t)f.tellg();
    f.seekg(0, std::fstream::beg);
    str = new char[size+1];

    PADDLE_MOBILE_ENFORCE(str != NULL, " str null")

    f.read(str, fileSize);
    f.close();
    str[size] = '\0';
    const char *source = str;
    size_t sourceSize[] = {strlen(source)};
    cl_program p = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
    std::unique_ptr<_cl_program, CLProgramDeleter> program_ptr(p);
    return std::move(program_ptr);
  }

  bool CLEngine::BuildProgram(cl_program program) {
    cl_int status;
    status = clBuildProgram(program, 0, 0, "-cl-fast-relaxed-math", 0, 0);
    CL_CHECK_ERRORS(status);
    return true;
  }

 private:
  CLEngine() { initialized_ = false; }

  bool SetPlatform();

  bool SetClDeviceId();

//  bool SetClContext();

//  bool SetClCommandQueue();

//  bool LoadKernelFromFile(const char *kernel_file);

//  bool BuildProgram();

  bool initialized_;

  cl_platform_id platform_;

  cl_device_id *devices_;

  std::unique_ptr<_cl_context, CLContextDeleter> context_;

  std::unique_ptr<_cl_command_queue, CLCommQueueDeleter> command_queue_;

  std::unique_ptr<_cl_program, CLProgramDeleter> program_;
};

}  // namespace framework
}  // namespace paddle_mobile
