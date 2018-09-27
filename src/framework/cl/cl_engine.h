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

#include <CL/cl.h>
// #include "CL/cl.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace paddle_mobile {
namespace framework {

struct CLContext {};

struct CLKernelDeleter {
  template <class T>
  void operator()(T *clKernelObj) {
    clReleaseKernel(clKernelObj);
  }
};

struct CLMemDeleter {
  template <class T>
  void operator()(T *clMemObj) {
    clReleaseMemObject(clMemObj);
  }
};

struct CLCommQueueDeleter {
  template <class T>
  void operator()(T *clQueueObj) {
    clReleaseCommandQueue(clQueueObj);
  }
};

struct CLContextDeleter {
  template <class T>
  void operator()(T *clContextObj) {
    clReleaseContext(clContextObj);
  }
};

struct CLProgramDeleter {
  template <class T>
  void operator()(T *clProgramObj) {
    clReleaseProgram(clProgramObj);
  }
};

class CLEngine {
 public:
  static CLEngine *Instance();

  bool Init();

  std::unique_ptr<_cl_kernel, clKernel_deleter> GetKernel(
      const std::string &kernel_name);

  const cl_context GetContext() { return context_.get(); }

  const cl_program GetProgram() { return program_.get(); }

  const cl_command_queue GetCommandQueue() { return command_queue_.get(); }

 private:
  CLEngine() { initialized_ = false; }

  bool SetPlatform();

  bool SetClDeviceId();

  bool SetClContext();

  bool SetClCommandQueue();

  bool LoadKernelFromFile(const char *kernel_file);

  bool BuildProgram();

  bool initialized_;
  cl_platform_id platform_;
  cl_device_id *devices_;
  std::unique_ptr<_cl_context, CLContextDeleter> context_;
  std::unique_ptr<_cl_command_queue, CLCommQueueDeleter> command_queue_;
  std::unique_ptr<_cl_program, clProgram_deleter> program_;
};

}  // namespace framework
}  // namespace paddle_mobile
