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

#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

NNAdapterOperation* Converter::AddOperation(NNAdapterOperationType type) {
  NNAdapterOperation* operation = nullptr;
  NNAdapterModel_addOperation_invoke(model_, type, &operation);
  return operation;
}

void Converter::SetOperation(NNAdapterOperation* operation,
                             std::vector<NNAdapterOperand*>* input_operands,
                             std::vector<NNAdapterOperand*>* output_operands) {
  NNAdapterModel_setOperation_invoke(operation,
                                     input_operands->size(),
                                     &((*input_operands)[0]),
                                     output_operands->size(),
                                     &((*output_operands)[0]));
}

bool Converter::HasOperand(const std::string& name) {
  return operands_.find(name) != operands_.end();
}

NNAdapterOperand* Converter::GetOperand(std::string name) {
  CHECK(HasOperand(name)) << "Operand '" << name << "' is not found!";
  return operands_[name];
}

NNAdapterOperand* Converter::AddOperand(NNAdapterOperandType* type,
                                        const std::string& name) {
  NNAdapterOperand* operand = nullptr;
  if (!name.empty()) {
    if (HasOperand(name)) {
      LOG(WARNING) << "Operand '" << name << "' already exists!";
      operand = operands_[name];
    } else {
      NNAdapterModel_addOperand_invoke(model_, type, &operand);
      operands_[name] = operand;
    }
  } else {
    // Anonymous operand
    NNAdapterModel_addOperand_invoke(model_, type, &operand);
  }
  return operand;
}

NNAdapterOperand* Converter::AddOperand(NNAdapterOperand* operand,
                                        const std::string& name) {
  CHECK(!operand);
  CHECK(!name.empty());
  operands_[name] = operand;
  return operand;
}

void Converter::SetOperandCopyFrom(NNAdapterOperand* operand,
                                   void* buffer,
                                   size_t length) {
  NNAdapterModel_setOperandCopyFrom_invoke(operand, buffer, length);
}

void Converter::SetOperandReferenceTo(NNAdapterOperand* operand,
                                      void* buffer,
                                      size_t length) {
  NNAdapterModel_setOperandReferenceTo_invoke(operand, buffer, length);
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
