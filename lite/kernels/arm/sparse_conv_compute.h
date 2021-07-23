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
#include <string>
#include <vector>
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <PrecisionType Ptype, PrecisionType OutType>
class SparseConvCompute : public KernelLite<TARGET(kARM), Ptype> {
 public:
  virtual void PrepareForRun();
  virtual void ReInitWhenNeeded() {}
  virtual void Run();

  ~SparseConvCompute() {}

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
  }
  std::string kernel_func_name_{"NotImplForSparseConv"};
#define KERNEL_FUNC_NAME(kernel_func_name) kernel_func_name_ = kernel_func_name;
#else
#define KERNEL_FUNC_NAME(kernel_func_name)
#endif

 private:
  using param_t = operators::SparseConvParam;
  Tensor bias_;
  bool flag_trans_bias_{false};
  std::vector<float> w_scale_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
