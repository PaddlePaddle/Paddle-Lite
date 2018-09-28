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


    FILE *file = fopen(file_name.c_str(), "rb");
    PADDLE_MOBILE_ENFORCE(file != nullptr, "can't open file: %s ",
                          filename.c_str());
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


  bool initialized_;

  cl_platform_id platform_;

  cl_device_id *devices_;

  std::unique_ptr<_cl_context, CLContextDeleter> context_;

  std::unique_ptr<_cl_command_queue, CLCommQueueDeleter> command_queue_;

  std::unique_ptr<_cl_program, CLProgramDeleter> program_;

//  bool SetClContext();

//  bool SetClCommandQueue();

//  bool LoadKernelFromFile(const char *kernel_file);

//  bool BuildProgram();

};

}  // namespace framework
}  // namespace paddle_mobile
