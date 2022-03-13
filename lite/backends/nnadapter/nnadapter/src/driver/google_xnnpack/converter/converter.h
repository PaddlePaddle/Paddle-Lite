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
#include <memory>
#include <string>
#include <vector>
#include "driver/google_xnnpack/utility.h"

namespace nnadapter {
namespace google_xnnpack {

const uint32_t INVALID_TENSOR_VALUE_ID = 0xFFFFFFFF;

class Converter {
 public:
  explicit Converter(
      xnn_subgraph_t subgraph,
      std::map<core::Operand*, std::vector<uint32_t>>* tensor_value_ids)
      : subgraph_(subgraph), tensor_value_ids_(tensor_value_ids) {}
  ~Converter() {}

  // Convert a NNAdapter model to a XNNPACK subgraph and tensor value ids
  int Apply(core::Model* model);
  // Mapping a XNNPACK tensor value id to a NNAdapter operand
  uint32_t GetMappedTensorValueId(core::Operand* operand);
  uint32_t UpdateTensorValueIdMap(core::Operand* operand,
                                  uint32_t tensor_value_id);
  // Convert a constant and model input operand and map to a XNNPACK tensor
  // value id
  uint32_t ConvertOperand(core::Operand* operand,
                          std::vector<int32_t> dimensions = {});

 private:
  xnn_subgraph_t subgraph_{nullptr};
  std::map<core::Operand*, std::vector<uint32_t>>* tensor_value_ids_{nullptr};
  uint32_t operand_index_{0};
};

}  // namespace google_xnnpack
}  // namespace nnadapter
