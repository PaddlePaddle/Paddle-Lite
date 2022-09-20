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

#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "driver/android_nnapi/utility.h"

namespace nnadapter {
namespace android_nnapi {

const uint32_t INVALID_INDEX = 0xFFFFFFFF;

class Converter {
 public:
  explicit Converter(
      ANeuralNetworksModel* model,
      std::map<core::Operand*, std::vector<uint32_t>>* operand_indexes)
      : model_(model), operand_indexes_(operand_indexes) {}
  ~Converter() {}

  // Convert a NNAdapter model to a NNAPI model and operand index
  int Apply(core::Model* model);
  // Mapping a NNAPI operand index to a NNAdapter operand
  uint32_t GetMappedIndex(core::Operand* operand);
  uint32_t UpdateIndexMap(core::Operand* operand, uint32_t index);
  uint32_t AddOperand(int32_t* dimensions_data,
                      uint32_t dimensions_count,
                      int precision,
                      float* quant_scales = nullptr,
                      int32_t* zero_point = nullptr,
                      uint32_t quant_scale_count = 0,
                      uint32_t quant_channel_dim = 0,
                      void* buffer = nullptr,
                      bool copy = true);
  int AddOperation(ANeuralNetworksOperationType type,
                   const std::vector<uint32_t>& input_indexes,
                   const std::vector<uint32_t>& output_indexes);
  uint32_t AddBool8ConstantOperand(bool value);
  uint32_t AddInt32ConstantOperand(int32_t value);
  uint32_t AddFloat32ConstantOperand(float value);
  uint32_t AddInt32ConstantOperand(int32_t* values,
                                   uint32_t num_values,
                                   bool copy = true);
  uint32_t AddFloat32ConstantOperand(float* values,
                                     uint32_t num_values,
                                     bool copy = true);
  uint32_t AddInt32ConstantOperand(int32_t* values,
                                   int32_t* dimensions_data,
                                   uint32_t dimensions_count,
                                   bool copy = true);
  uint32_t AddFloat32ConstantOperand(float* values,
                                     int32_t* dimensions_data,
                                     uint32_t dimensions_count,
                                     bool copy = true);
  uint32_t AddInt32ConstantOperand(const std::vector<int32_t>& values);
  uint32_t AddFloat32ConstantOperand(const std::vector<float>& values);
  // Quant8 constant operand with symmetric per-channel quantizion
  uint32_t AddQuant8ConstantOperand(int8_t* values,
                                    uint32_t num_values,
                                    float* quant_scales,
                                    uint32_t quant_scale_count,
                                    uint32_t quant_channel_dim = 0,
                                    bool copy = true);
  uint32_t AddQuant8ConstantOperand(int8_t* values,
                                    int32_t* dimensions_data,
                                    uint32_t dimensions_count,
                                    float* quant_scales,
                                    uint32_t quant_scale_count,
                                    uint32_t quant_channel_dim = 0,
                                    bool copy = true);
  uint32_t AddQuant8ConstantOperand(const std::vector<int8_t>& values,
                                    float* quant_scales,
                                    uint32_t quant_scale_count,
                                    uint32_t quant_channel_dim = 0);
  // Quant8 constant operand with asymmetric per-layer quantizion
  uint32_t AddQuant8ConstantOperand(uint8_t* values,
                                    uint32_t num_values,
                                    float quant_scale,
                                    int32_t zero_point,
                                    bool copy = true);
  uint32_t AddQuant8ConstantOperand(uint8_t* values,
                                    int32_t* dimensions_data,
                                    uint32_t dimensions_count,
                                    float quant_scale,
                                    int32_t zero_point,
                                    bool copy = true);
  uint32_t AddQuant8ConstantOperand(const std::vector<uint8_t>& values,
                                    float quant_scale,
                                    int32_t zero_point);
  // Quant32 constant operand with symmetric per-layer quantizion
  uint32_t AddQuant32ConstantOperand(int32_t* values,
                                     uint32_t num_values,
                                     float quant_scale,
                                     bool copy = true);
  uint32_t AddQuant32ConstantOperand(int32_t* values,
                                     int32_t* dimensions_data,
                                     uint32_t dimensions_count,
                                     float quant_scale,
                                     bool copy = true);
  uint32_t AddQuant32ConstantOperand(const std::vector<int32_t>& values,
                                     float quant_scale);
  // Float32 variable operand
  uint32_t AddFloat32VariableOperand(int32_t* dimensions_data,
                                     uint32_t dimensions_count);
  // Quant8 variable operand with asymmetric per-layer quantizion
  uint32_t AddQuant8VariableOperand(int32_t* dimensions_data,
                                    uint32_t dimensions_count,
                                    float quant_scale,
                                    int32_t zero_point);
  // Convert a constant and model input operand and map to a NNAPI operand index
  uint32_t ConvertOperand(core::Operand* operand,
                          std::vector<int32_t> dimensions = {});

 private:
  ANeuralNetworksModel* model_{nullptr};
  std::map<core::Operand*, std::vector<uint32_t>>* operand_indexes_{nullptr};
  uint32_t operand_index_{0};
  std::list<std::vector<uint8_t>> operand_values_;
};

}  // namespace android_nnapi
}  // namespace nnadapter
