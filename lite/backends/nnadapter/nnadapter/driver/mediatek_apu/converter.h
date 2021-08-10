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
#include "driver/mediatek_apu/utility.h"

namespace nnadapter {
namespace mediatek_apu {

const uint32_t INVALID_INDEX = 0xFFFFFFFF;

class Device {
 public:
  Device() {}
  ~Device() {}
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  ~Context();

 private:
  void* device_{nullptr};
  void* context_{nullptr};
};

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program();

  int Build(hal::Model* model, hal::Cache* cache);
  int Execute(uint32_t input_count,
              hal::Argument* input_arguments,
              uint32_t output_count,
              hal::Argument* output_arguments);

 private:
  void Clear();
  // Build from model or cache
  int BuildFromModel(hal::Model* model);
  int BuildFromCache(hal::Cache* cache);

  // Operand converters
  uint32_t GetMappedIndex(hal::Operand* operand);
  uint32_t UpdateIndexMap(hal::Operand* operand, uint32_t index);
  uint32_t AddOperand(int32_t* dimensions,
                      uint32_t dimension_count,
                      int precision,
                      float* quant_scales = nullptr,
                      int32_t* zero_point = nullptr,
                      uint32_t quant_scale_count = 0,
                      uint32_t quant_channel_dim = 0,
                      void* buffer = nullptr);
  int AddOperation(NeuronOperationType type,
                   std::vector<uint32_t>* input_indexes,
                   std::vector<uint32_t>* output_indexes);
  uint32_t AddBool8ConstantOperand(bool value);
  uint32_t AddInt32ConstantOperand(int32_t value);
  uint32_t AddFloat32ConstantOperand(float value);
  uint32_t AddInt32ConstantOperand(int32_t* values, uint32_t num_values);
  uint32_t AddFloat32ConstantOperand(float* values, uint32_t num_values);
  uint32_t AddInt32ConstantOperand(int32_t* values,
                                   int32_t* dimensions,
                                   uint32_t dimension_count);
  uint32_t AddFloat32ConstantOperand(float* values,
                                     int32_t* dimensions,
                                     uint32_t dimension_count);
  // Quant8 constant operand with symmetric per-channel quantizion
  uint32_t AddQuant8ConstantOperand(int8_t* values,
                                    int32_t* dimensions,
                                    uint32_t dimension_count,
                                    float* quant_scales,
                                    uint32_t quant_scale_count,
                                    uint32_t quant_channel_dim = 0);
  // Quant8 constant operand with asymmetric per-layer quantizion
  uint32_t AddQuant8ConstantOperand(uint8_t* values,
                                    int32_t* dimensions,
                                    uint32_t dimension_count,
                                    float quant_scale,
                                    int32_t zero_point);
  // Quant32 constant operand with symmetric per-layer quantizion
  uint32_t AddQuant32ConstantOperand(int32_t* values,
                                     int32_t* dimensions,
                                     uint32_t dimension_count,
                                     float quant_scale);
  // Quant8 variable operand with asymmetric per-layer quantizion
  uint32_t AddQuant8VariableOperand(int32_t* dimensions,
                                    uint32_t dimension_count,
                                    float quant_scale,
                                    int32_t zero_point);
  // Convert a constant and model input operand and map to a Neuron operand
  // index
  uint32_t ConvertOperand(hal::Operand* operand,
                          std::vector<int32_t> dimensions = {});

  // Operation converters
  int ConvertConv2D(hal::Operation* operation);
  int ConvertFullyConnected(hal::Operation* operation);
  int ConvertPool2D(hal::Operation* operation);
  int ConvertElementwise(hal::Operation* operation);
  int ConvertSoftmax(hal::Operation* operation);
  int ConvertActivation(hal::Operation* operation);
  int ConvertReshape(hal::Operation* operation);
  int ConvertTranspose(hal::Operation* operation);
  int ConvertConcat(hal::Operation* operation);

 private:
  Context* context_{nullptr};
  // Map NNAdapter operand to Neuron operand index
  std::map<hal::Operand*, std::vector<uint32_t>> operand_indexes_;
  uint32_t operand_index_{0};
  std::vector<void*> operand_buffers_;
  NeuronModel* model_{nullptr};
  NeuronCompilation* compilation_{nullptr};
  NeuronExecution* execution_{nullptr};
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  std::string dump_graph_path_;
  std::vector<uint8_t>* dump_graph_buffer_{nullptr};
};

}  // namespace mediatek_apu
}  // namespace nnadapter
