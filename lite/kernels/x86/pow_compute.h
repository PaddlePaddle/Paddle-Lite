// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "lite/backends/x86/math/power.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

class PowCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::PowParam;
  void Run() override;
  virtual ~PowCompute() = default;

#ifdef LITE_WITH_PROFILE
#define PROFILE_INFO                            \
  template <>                                   \
  void PowCompute::SetProfileRuntimeKernelInfo( \
      paddle::lite::profile::OpCharacter* ch) { \
    ch->kernel_func_name = kernel_func_name_;   \
  }

#define KERNEL_FUNC_NAME(kernel_func_name) kernel_func_name_ = kernel_func_name;
#else
#define PROFILE_INFO(dtype1, dtype2)
#define KERNEL_FUNC_NAME(kernel_func_name)
#endif
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
