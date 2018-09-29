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

#include <vector>
#include <type_traits>

#include "framework/cl/cl_scope.h"
#include "framework/cl/cl_deleter.h"

namespace paddle_mobile {
namespace framework {

class CLHelper {
 public:
  CLHelper() = default;

  CLHelper(CLScope *scope): scope_(scope) {
  }

  void AddKernel(const std::string &kernel_name, const std::string &file_name) {
    auto kernel = scope_->GetKernel(kernel_name, file_name);
    kernels.emplace_back(std::move(kernel));
  }

  cl_kernel KernelAt(const int index) {
    return kernels[index].get();
  }

  cl_command_queue CLCommandQueue() {
    return scope_->CommandQueue();
  }

  cl_context CLContext() {
    return scope_->Context();
  }

 private:
  CLScope *scope_;
  std::vector<std::unique_ptr<_cl_kernel, CLKernelDeleter>> kernels;
};

}
}
