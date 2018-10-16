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
#include <unordered_map>

#include "CL/cl.h"
#include "framework/cl/cl_deleter.h"
#include "framework/cl/cl_engine.h"
#include "framework/cl/cl_tool.h"

namespace paddle_mobile {
namespace framework {

class CLScope {
 public:
  CLScope() {
    CLEngine *engin = CLEngine::Instance();
    context_ = engin->CreateContext();
    command_queue_ = engin->CreateClCommandQueue(context_.get());
  }

  cl_command_queue CommandQueue() { return command_queue_.get(); }

  std::unique_ptr<_cl_kernel, CLKernelDeleter> GetKernel(
      const std::string &kernel_name, const std::string &file_name) {
    auto program = Program(file_name);
    DLOG << " get program ~ ";
    std::unique_ptr<_cl_kernel, CLKernelDeleter> kernel(
        clCreateKernel(program, kernel_name.c_str(), &status_));
    CL_CHECK_ERRORS(status_);
    DLOG << " create kernel ~ ";
    return std::move(kernel);
  }

  cl_context Context() { return context_.get(); }

  cl_program Program(const std::string &file_name) {
    auto it = programs_.find(file_name);
    if (it != programs_.end()) {
      return it->second.get();
    }

    auto program = CLEngine::Instance()->CreateProgramWith(
        context_.get(), "./cl_kernel/" + file_name);

    DLOG << " --- begin build program -> " << file_name << " --- ";
    status_ =
        clBuildProgram(program.get(), 0, 0, "-cl-fast-relaxed-math", 0, 0);

    CL_CHECK_ERRORS(status_);

    if (status_ == CL_BUILD_PROGRAM_FAILURE) {
      size_t log_size;
      clGetProgramBuildInfo(program.get(), CLEngine::Instance()->DeviceID(),
                            CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      char *log = (char *)malloc(log_size);
      clGetProgramBuildInfo(program.get(), CLEngine::Instance()->DeviceID(),
                            CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
      DLOG << " program build error: " << log;
    }

    DLOG << " --- end build program -> " << file_name << " --- ";

    programs_[file_name] = std::move(program);

    return programs_[file_name].get();
  }

 private:
  cl_int status_;
  std::unique_ptr<_cl_context, CLContextDeleter> context_;
  std::unique_ptr<_cl_command_queue, CLCommQueueDeleter> command_queue_;
  std::unordered_map<std::string,
                     std::unique_ptr<_cl_program, CLProgramDeleter>>
      programs_;
};

}  // namespace framework
}  // namespace paddle_mobile
