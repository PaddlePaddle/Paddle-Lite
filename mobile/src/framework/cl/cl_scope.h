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
#include <utility>
#include <vector>

#include "CL/cl.h"
#include "framework/cl/cl_deleter.h"
#include "framework/cl/cl_engine.h"
#include "framework/cl/cl_tool.h"

namespace paddle_mobile {

extern const std::map<std::string, std::vector<unsigned char>> opencl_kernels;
extern const std::map<std::string, std::vector<unsigned char>> opencl_headers;

namespace framework {

class CLScope {
 public:
  CLScope() {}

  cl_command_queue CommandQueue() {
    return CLEngine::Instance()->getClCommandQueue();
  }

  std::unique_ptr<_cl_kernel, CLKernelDeleter> GetKernel(
      const std::string &kernel_name, const std::string &file_name,
      const std::string &options) {
    LOG(kLOG_DEBUG2) << " to get program " << file_name;
    auto program = Program(file_name, kernel_name, options);
    LOG(kLOG_DEBUG2) << " end get program ~ ";
    LOG(kLOG_DEBUG2) << " to create kernel: " << kernel_name;
    std::unique_ptr<_cl_kernel, CLKernelDeleter> kernel(
        clCreateKernel(program, kernel_name.c_str(), &status_));
    CL_CHECK_ERRORS(status_);
    LOG(kLOG_DEBUG2) << " end create kernel ~ ";
    return std::move(kernel);
  }

  cl_context Context() { return CLEngine::Instance()->getContext(); }

  cl_program Program(const std::string &file_name,
                     const std::string &kernel_name,
                     const std::string &options) {
    if (opencl_kernels.find(kernel_name) != opencl_kernels.end() &&
        opencl_headers.find(file_name) != opencl_headers.end()) {
      std::string program_key = file_name + kernel_name;
      if (!options.empty()) {
        program_key += options;
      }
      auto it = programs_.find(program_key);
      if (it != programs_.end()) {
        return it->second.get();
      }
      auto src_it = opencl_kernels.find(kernel_name);
      std::string source(src_it->second.begin(), src_it->second.end());
      auto header_it = opencl_headers.find(file_name);
      std::string header(header_it->second.begin(), header_it->second.end());
      source = header + "\n" + source;
      auto program = CLEngine::Instance()->CreateProgramWithSource(
          CLEngine::Instance()->getContext(), source.c_str());

      LOG(kLOG_DEBUG3) << " --- begin build program -> " << program_key
                       << " --- ";
      CLEngine::Instance()->BuildProgram(program.get(), options);
      LOG(kLOG_DEBUG3) << " --- end build program -> " << program_key
                       << " --- ";

      programs_[program_key] = std::move(program);
      return programs_[program_key].get();
    } else {
      std::string program_key = file_name;
      if (!options.empty()) {
        program_key += options;
      }
      auto it = programs_.find(program_key);
      if (it != programs_.end()) {
        return it->second.get();
      }
      auto program = CLEngine::Instance()->CreateProgramWith(
          CLEngine::Instance()->getContext(),
          CLEngine::Instance()->GetCLPath() + "/cl_kernel/" + file_name);

      LOG(kLOG_DEBUG3) << " --- begin build program ele-> " << program_key
                       << " --- ";
      CLEngine::Instance()->BuildProgram(program.get(), options);
      LOG(kLOG_DEBUG3) << " --- end build program ele-> " << program_key
                       << " --- ";

      programs_[program_key] = std::move(program);
      return programs_[program_key].get();
    }
  }

  CLLocalWorkSizeInfo LocalWorkSizeInfo() {
    return CLEngine::Instance()->getLocalWorkSizeInfo();
  }
  size_t KernelWorkSize(cl_kernel kernel) {
    size_t kernel_work_size = CLEngine::Instance()->GetKernelWorkSize(kernel);
    return kernel_work_size;
  }

 private:
  cl_int status_;
  std::unordered_map<std::string,
                     std::unique_ptr<_cl_program, CLProgramDeleter>>
      programs_;
};

}  // namespace framework
}  // namespace paddle_mobile
