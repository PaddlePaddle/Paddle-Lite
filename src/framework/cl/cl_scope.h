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

#include "framework/cl/cl_tool.h"
#include "framework/cl/cl_engine.h"
#include "framework/cl/cl_deleter.h"
#include "CL/cl.h"

namespace paddle_mobile {
namespace framework {

class CLScope {
 public:
  CLScope() {
    CLEngine *engin = CLEngine::Instance();
    context_ = engin->CreateContext();
    command_queue_ = engin->CreateClCommandQueue();
  }

  cl_command_queue CommandQueue() {
    return command_queue_.get();
  }

  std::unique_ptr<_cl_kernel, CLKernelDeleter> GetKernel(const std::string &kernel_name, const std::string &file_name) {
    auto program = Program(file_name);
    std::unique_ptr<_cl_kernel, CLKernelDeleter> kernel(clCreateKernel(program, kernel_name.c_str(), NULL));
    return std::move(kernel);
  }

  cl_context Context() {
    return context_.get();
  }

  cl_program Program(const std::string &file_name) {
    auto it = programs_.find(file_name);
    if (it != programs_.end()) {
      return it->second.get();
    }

    auto program = CLEngine::Instance()->CreateProgramWith(context_.get(), file_name);
    programs_[file_name] = std::move(program);

    status_ =  clBuildProgram(program.get(), 0, 0, 0, 0, 0);
    CL_CHECK_ERRORS(status_);
    return program.get();
  }

 private:
  cl_int    status_;
  std::unique_ptr<_cl_context, CLContextDeleter> context_;
  std::unique_ptr<_cl_command_queue, CLCommQueueDeleter> command_queue_;
  std::unordered_map<std::string, std::unique_ptr<_cl_program, CLProgramDeleter>> programs_;
};

}
}
