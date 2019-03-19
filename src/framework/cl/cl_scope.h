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
    CLEngine *engine = CLEngine::Instance();
    context_ = engine->getContext();
    command_queue_ = engine->getClCommandQueue();
  }

  cl_command_queue CommandQueue() { return command_queue_; }

  std::unique_ptr<_cl_kernel, CLKernelDeleter> GetKernel(
      const std::string &kernel_name, const std::string &file_name) {
    DLOG << " to get program " << file_name;
    auto program = Program(file_name);
    DLOG << " end get program ~ ";
    DLOG << " to create kernel: " << kernel_name;
    std::unique_ptr<_cl_kernel, CLKernelDeleter> kernel(
        clCreateKernel(program, kernel_name.c_str(), &status_));
    CL_CHECK_ERRORS(status_);
    DLOG << " end create kernel ~ ";
    return std::move(kernel);
  }

  cl_context Context() { return context_; }

  cl_program Program(const std::string &file_name) {
    auto it = programs_.find(file_name);
    if (it != programs_.end()) {
      return it->second.get();
    }

    auto program = CLEngine::Instance()->CreateProgramWith(
        context_,
        CLEngine::Instance()->GetCLPath() + "/cl_kernel/" + file_name);

    DLOG << " --- begin build program -> " << file_name << " --- ";
    CLEngine::Instance()->BuildProgram(program.get());
    DLOG << " --- end build program -> " << file_name << " --- ";

    programs_[file_name] = std::move(program);

    return programs_[file_name].get();
  }

 private:
  cl_int status_;
  cl_context context_;
  cl_command_queue command_queue_;
  std::unordered_map<std::string,
                     std::unique_ptr<_cl_program, CLProgramDeleter>>
      programs_;
};

}  // namespace framework
}  // namespace paddle_mobile
