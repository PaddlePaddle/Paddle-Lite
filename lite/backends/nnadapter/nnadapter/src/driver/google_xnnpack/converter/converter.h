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

class Converter {
 public:
  explicit Converter(
      xnn_subgraph_t subgraph,
      std::map<core::Operand*, std::vector<uint32_t>>* tensor_value_ids)
      : subgraph_(subgraph), tensor_value_ids_(tensor_value_ids) {}
  ~Converter() {}

  xnn_subgraph_t subgraph() { return subgraph_; }
  // Convert a NNAdapter model to a XNNPACK subgraph and tensor value ids
  int Apply(core::Model* model);
  // Mapping a XNNPACK tensor value id to a NNAdapter operand
  uint32_t GetMappedTensorValueId(core::Operand* operand);
  uint32_t UpdateTensorValueIdMap(core::Operand* operand,
                                  uint32_t tensor_value_id);
  uint32_t AddTensorValue(int32_t* dimensions_data,
                          uint32_t dimensions_count,
                          xnn_datatype datatype,
                          float* quant_scales = nullptr,
                          uint32_t quant_scale_count = 0,
                          uint32_t quant_channel_dim = 0,
                          void* buffer = nullptr,
                          uint32_t flags = 0);
  uint32_t AddFloat32ConstantTensorValue(float* values,
                                         int32_t* dimensions_data,
                                         uint32_t dimensions_count);
  uint32_t AddFloat32ConstantTensorValue(float* values, uint32_t num_values);
  uint32_t AddFloat32ConstantTensorValue(float value);
  uint32_t AddQuant8ConstantTensorValue(int8_t* values,
                                        int32_t* dimensions_data,
                                        uint32_t dimensions_count,
                                        float* quant_scales,
                                        uint32_t quant_scale_count,
                                        uint32_t quant_channel_dim);
  uint32_t AddQuant8ConstantTensorValue(int8_t* values,
                                        int32_t* dimensions_data,
                                        uint32_t dimensions_count,
                                        float quant_scale);
  uint32_t AddQuant32ConstantTensorValue(int32_t* values,
                                         int32_t* dimensions_data,
                                         uint32_t dimensions_count,
                                         float quant_scale);
  uint32_t AddFloat32VariableTensorValue(int32_t* dimensions_data,
                                         uint32_t dimensions_count,
                                         uint32_t flags = 0);
  uint32_t AddQuant8VariableTensorValue(int32_t* dimensions_data,
                                        uint32_t dimensions_count,
                                        float quant_scale,
                                        uint32_t flags = 0);
  // Convert a constant and model input operand, map it to a XNNPACK tensor
  // value id
  uint32_t ConvertOperand(core::Operand* operand,
                          std::vector<int32_t> dimensions = {});

 private:
  xnn_subgraph_t subgraph_{nullptr};
  std::map<core::Operand*, std::vector<uint32_t>>* tensor_value_ids_{nullptr};
};

#define ADD_OPERATOR(__operator__, ...)                                   \
  {                                                                       \
    xnn_status status = __operator__(converter->subgraph(), __VA_ARGS__); \
    NNADAPTER_CHECK(status == xnn_status_success)                         \
        << "Failed to add a operator into a XNNPACK subgraph(status = "   \
        << status << ")!";                                                \
  }

}  // namespace google_xnnpack
}  // namespace nnadapter
