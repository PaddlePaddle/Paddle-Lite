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
#include "driver/imagination_nna/imgdnn_manager.h"
#include "driver/imagination_nna/utility.h"

namespace nnadapter {
namespace imagination_nna {

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
  // Operand converters
  imgdnn_tensor GetMappedTensor(hal::Operand* operand);
  imgdnn_tensor UpdateTensorMap(hal::Operand* operand, imgdnn_tensor tensor);
  imgdnn_tensor AddTensor(int32_t* dimensions,
                          uint32_t dimension_count,
                          imgdnn_type type,
                          const float* quant_scales,
                          const int32_t* zero_point,
                          uint32_t quant_scale_count,
                          uint32_t quant_channel_dim,
                          void* buffer);
  imgdnn_tensor AddTensor(const NNAdapterOperandType* type,
                          void* buffer,
                          std::vector<int32_t> dimensions);
  // Quant8 constant operand with asymmetric per-layer quantizion
  imgdnn_tensor AddQuant8ConstantTensor(uint8_t* values,
                                        int32_t* dimensions,
                                        uint32_t dimension_count,
                                        float quant_scale,
                                        int32_t zero_point);
  // Quant8 constant operand with symmetric per-channel quantizion
  imgdnn_tensor AddQuant8ConstantTensor(int8_t* values,
                                        int32_t* dimensions,
                                        uint32_t dimension_count,
                                        float* quant_scales,
                                        uint32_t quant_scale_count,
                                        uint32_t quant_channel_dim);
  // Quant32 constant operand with symmetric per-layer quantizion
  imgdnn_tensor AddQuant32ConstantTensor(int32_t* values,
                                         int32_t* dimensions,
                                         uint32_t dimension_count,
                                         float quant_scale);
  imgdnn_tensor ConvertOperand(hal::Operand* operand,
                               std::vector<int32_t> dimensions = {});

  // Operation converters
  int ConvertConv2D(hal::Operation* operation);
  int ConvertFullyConnected(hal::Operation* operation);
  int ConvertPool2D(hal::Operation* operation);
  int ConvertSoftmax(hal::Operation* operation);
  int ConvertActivation(hal::Operation* operation);

 private:
  Context* context_{nullptr};
  // Map NNAdapter operand to imgdnn tensor
  std::map<hal::Operand*, std::vector<imgdnn_tensor>> tensors_;
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  std::vector<imgdnn_input> input_info_;
  std::vector<imgdnn_output> output_info_;
  std::vector<std::pair<imgdnn_memory, size_t>> input_memory_;
  std::vector<std::pair<imgdnn_memory, size_t>> output_memory_;
  ImgdnnManager imgdnn_mgr_;
};

}  // namespace imagination_nna
}  // namespace nnadapter
