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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "CL/cl.h"
#include "framework/cl/cl_deleter.h"
#include "framework/cl/cl_engine.h"
#include "framework/cl/cl_tool.h"

namespace paddle_mobile {

extern const std::map<std::string, std::vector<unsigned char>> opencl_kernels;
extern const std::vector<std::string> need_conv_header_kernels;

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
      const std::string &kernel_name, const std::string &file_name,
      const std::string &options) {
    DLOG << " to get program " << file_name;
    auto program = Program(file_name, options);
    DLOG << " end get program ~ ";
    DLOG << " to create kernel: " << kernel_name;
    std::unique_ptr<_cl_kernel, CLKernelDeleter> kernel(
        clCreateKernel(program, kernel_name.c_str(), &status_));
    CL_CHECK_ERRORS(status_);
    DLOG << " end create kernel ~ ";
    return std::move(kernel);
  }

  cl_context Context() { return context_; }

  cl_program Program(const std::string &file_name, const std::string &options) {
    std::string program_key = file_name;
    if (!options.empty()) {
      program_key += options;
    }
    auto it = programs_.find(program_key);
    if (it != programs_.end()) {
      return it->second.get();
    }

    if (opencl_kernels.find(file_name) != opencl_kernels.end()) {
      auto it = opencl_kernels.find(file_name);
      std::string source(it->second.begin(), it->second.end());
      if (std::find(need_conv_header_kernels.begin(),
                    need_conv_header_kernels.end(),
                    file_name) != need_conv_header_kernels.end()) {
        auto it = opencl_kernels.find("conv_kernel.inc.cl");
        std::string header(it->second.begin(), it->second.end());
        source = header + source;
      }
      auto program = CLEngine::Instance()->CreateProgramWithSource(
          context_, source.c_str());

      DLOG << " --- begin build program -> " << program_key << " --- ";
      CLEngine::Instance()->BuildProgram(program.get(), options);
      DLOG << " --- end build program -> " << program_key << " --- ";

      programs_[program_key] = std::move(program);
    } else {
      auto program = CLEngine::Instance()->CreateProgramWith(
          context_,
          CLEngine::Instance()->GetCLPath() + "/cl_kernel/" + file_name);

      DLOG << " --- begin build program -> " << program_key << " --- ";
      CLEngine::Instance()->BuildProgram(program.get(), options);
      DLOG << " --- end build program -> " << program_key << " --- ";

      programs_[program_key] = std::move(program);
    }

    return programs_[program_key].get();
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
