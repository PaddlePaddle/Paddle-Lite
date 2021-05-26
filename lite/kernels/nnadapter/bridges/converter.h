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

#include <map>
#include <string>
#include <vector>
#include "lite/backends/nnadapter/nnadapter_wrapper.h"
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

class Converter {
 public:
  explicit Converter(NNAdapterModel* model) : model_(model) {}
  ~Converter() {}

  NNAdapterOperation* AddOperation(NNAdapterOperationType type);
  void SetOperation(NNAdapterOperation* operation,
                    std::vector<NNAdapterOperand*>* input_operands,
                    std::vector<NNAdapterOperand*>* output_operands);
  bool HasOperand(const std::string& name);
  NNAdapterOperand* GetOperand(std::string name);
  NNAdapterOperand* AddOperand(NNAdapterOperandType* type,
                               const std::string& name = "");
  NNAdapterOperand* AddOperand(NNAdapterOperand* operand,
                               const std::string& name);
  void SetOperandCopyFrom(NNAdapterOperand* operand,
                          void* buffer,
                          size_t length);
  void SetOperandReferenceTo(NNAdapterOperand* operand,
                             void* buffer,
                             size_t length);

 private:
  std::map<std::string, NNAdapterOperand*> operands_;
  NNAdapterModel* model_{nullptr};
};

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
