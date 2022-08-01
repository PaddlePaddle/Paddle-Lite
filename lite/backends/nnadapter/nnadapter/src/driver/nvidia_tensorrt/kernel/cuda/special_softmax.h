// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <map>
#include <memory>
#include "driver/nvidia_tensorrt/kernel/kernel.h"
#include "driver/nvidia_tensorrt/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {
namespace cuda {

class SpecialSoftmaxKernel : public KernelBase {
 public:
  virtual ~SpecialSoftmaxKernel() {}

  int Run(core::Operation* operation,
          std::map<core::Operand*, std::shared_ptr<Tensor>>* operand_map,
          cudaStream_t stream = nullptr) override;
};

}  // namespace cuda
}  // namespace nvidia_tensorrt
}  // namespace nnadapter
