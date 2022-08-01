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
#include "utility.h"  // NOLINT

namespace nnadapter {
namespace fake_device {

class Converter {
 public:
  explicit Converter(
      fake_ddk::Graph* graph,
      std::map<core::Operand*, std::vector<fake_ddk::Tensor*>>* tensors)
      : graph_(graph), tensors_(tensors) {}
  ~Converter() {}

  // Convert a NNAdapter model to fake device graph and tensors
  int Apply(core::Model* model);
  // Mapping a fake device tensor to a NNAdapter operand
  std::string GetTensorName(core::Operand* operand);
  fake_ddk::Tensor* GetMappedTensor(core::Operand* operand);
  fake_ddk::Tensor* UpdateTensorMap(core::Operand* operand,
                                    fake_ddk::Tensor* tensor);
  // Create and add a fake device tensor
  fake_ddk::Tensor* AddTensor(
      int32_t* dimensions_data,
      uint32_t dimensions_count,
      fake_ddk::PrecisionType precision,
      const float* quant_scales = nullptr,
      const int32_t* zero_points = nullptr,
      uint32_t scale_count = 0,
      int channel_dim = -1,
      void* buffer = nullptr,
      fake_ddk::DataLayoutType layout = fake_ddk::DataLayoutType::NCHW);
  // Convert a NNAdapter operand to a fake device tensor
  fake_ddk::Tensor* ConvertOperand(core::Operand* operand,
                                   std::vector<int32_t> dimensions = {});
  // Create and add a fake device operator into fake device graph
  fake_ddk::Operator* AddOperator(fake_ddk::OperatorType type,
                                  std::vector<fake_ddk::Tensor*> input_tensors,
                                  std::vector<fake_ddk::Tensor*> output_tensors,
                                  void* attrs);

 private:
  fake_ddk::Graph* graph_{nullptr};
  std::map<core::Operand*, std::vector<fake_ddk::Tensor*>>* tensors_{nullptr};
};

}  // namespace fake_device
}  // namespace nnadapter
