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

int Converter::AddOperand(const std::string& name,
                          std::shared_ptr<Operand> operand) {
  auto it = operands_.find(name);
  if (it != operands_.end()) {
    // Only temporary variable operand can be shared with the same name
    if (!operand->is_temporary_variable() ||
        !it->second.back()->is_temporary_variable()) {
      LOG(FATAL) << "Constant, input or output operand " << name
                 << " is redefined.";
      return -1;
    }
  } else {
    auto ret = operands_.insert(
        std::make_pair(name, std::vector<std::shared_ptr<Operand>>()));
    CHECK(ret.second);
    it = ret.first;
  }
  it->second.push_back(operand);
  return it->second.size();
}

std::shared_ptr<Operand> Converter::AddOperand(const std::string& name,
                                               NNAdapterOperand* operand) {
  Operand::Lifetime lifetime = Operand::Lifetime::kTemporaryVariable;
  auto handle = std::make_shared<Operand>(operand, lifetime);
  auto index = AddOperand(name, handle);
  CHECK_GE(index, 1);
  return handle;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
