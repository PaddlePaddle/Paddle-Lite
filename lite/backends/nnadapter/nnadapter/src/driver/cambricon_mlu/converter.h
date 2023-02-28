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
#include <utility>
#include <vector>
#include "driver/cambricon_mlu/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

class Converter {
 public:
  explicit Converter(
      std::map<core::Operand*, std::vector<magicmind::ITensor*>>* tensors,
      magicmind::INetwork* network,
      const std::string& op_params)
      : tensors_(tensors), network_(network), op_params_(op_params) {}
  ~Converter() {}

  // Convert a NNAdapter model to MagicMind network.
  int Apply(core::Model* model);

  magicmind::INetwork* network();

  const std::string& op_params();

  magicmind::ITensor* GetMappedTensor(core::Operand* operand);

  void UpdateTensorMap(core::Operand* operand, magicmind::ITensor* tensor);

  magicmind::ITensor* AddInt32ConstantTensor(void* buffer,
                                             std::vector<int64_t> dimensions);

  magicmind::ITensor* AddFloat32ConstantTensor(void* buffer,
                                               std::vector<int64_t> dimensions);

  magicmind::ITensor* AddTensor(const NNAdapterOperandType* type,
                                void* buffer,
                                std::vector<int64_t> dimensions);

  magicmind::ITensor* AddTensor(const NNAdapterOperandType* type,
                                std::vector<int64_t> dimensions);

  magicmind::ITensor* ConvertOperand(core::Operand* operand,
                                     std::vector<int64_t> dimensions = {});

 private:
  // Map NNAdapter operand to magicmind itensor
  std::map<core::Operand*, std::vector<magicmind::ITensor*>>* tensors_;
  magicmind::INetwork* network_;
  std::string op_params_;
};

}  // namespace cambricon_mlu
}  // namespace nnadapter
