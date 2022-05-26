// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "core/types.h"
#include "runtime/context.h"

namespace nnadapter {
namespace runtime {

class Model {
 public:
  Model() : completed_{false} {}
  ~Model();
  int AddOperand(const NNAdapterOperandType& type, core::Operand** operand);
  int AddOperation(NNAdapterOperationType type, core::Operation** operation);
  int IdentifyInputsAndOutputs(uint32_t input_count,
                               core::Operand** input_operands,
                               uint32_t output_count,
                               core::Operand** output_operands);
  int Finish();
  // Get the supported operations for one device
  int GetSupportedOperations(Context::DeviceContext* device_context,
                             bool* supported_operations) const;
  // Get the supported operations for some devices, this operation is supported
  // as long as one device supports it.
  int GetSupportedOperations(Context* context,
                             bool* supported_operations) const;

  core::Model model_;
  bool completed_;
};

}  // namespace runtime
}  // namespace nnadapter
