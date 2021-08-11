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
#include "driver/rockchip_npu/utility.h"

namespace nnadapter {
namespace rockchip_npu {

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
  std::string GetTensorName(hal::Operand* operand);
  std::shared_ptr<rk::nn::Tensor> GetMappedTensor(hal::Operand* operand);
  std::shared_ptr<rk::nn::Tensor> UpdateTensorMap(
      hal::Operand* operand, std::shared_ptr<rk::nn::Tensor> tensor);
  std::shared_ptr<rk::nn::Tensor> AddTensor(
      const std::string& name,
      int32_t* dimensions,
      uint32_t dimension_count,
      rk::nn::PrecisionType precision,
      const float* quant_scale = nullptr,
      const int32_t* zero_point = nullptr,
      void* buffer = nullptr,
      rk::nn::DataLayoutType layout = rk::nn::DataLayoutType::NCHW);
  std::shared_ptr<rk::nn::Tensor> AddTensor(
      const std::string& name,
      const NNAdapterOperandType* type,
      void* buffer = nullptr,
      std::vector<int32_t> dimensions = {});
  std::shared_ptr<rk::nn::Tensor> AddConstantTensor(
      void* values,
      int32_t* dimensions,
      uint32_t dimension_count,
      rk::nn::PrecisionType precision,
      const float* quant_scale = nullptr,
      const int32_t* zero_point = nullptr);
  std::shared_ptr<rk::nn::Tensor> AddVariableTensor(
      const std::string& name,
      int32_t* dimensions,
      uint32_t dimension_count,
      rk::nn::PrecisionType precision,
      const float* quant_scale = nullptr,
      const int32_t* zero_point = nullptr);
  // Quant8 constant operand with asymmetric per-layer quantizion
  std::shared_ptr<rk::nn::Tensor> AddQuant8ConstantTensor(
      uint8_t* values,
      int32_t* dimensions,
      uint32_t dimension_count,
      float quant_scale,
      int32_t zero_point);
  // Quant32 constant operand with symmetric per-layer quantizion
  std::shared_ptr<rk::nn::Tensor> AddQuant32ConstantTensor(
      int32_t* values,
      int32_t* dimensions,
      uint32_t dimension_count,
      float quant_scale);
  // Quant8 variable operand with asymmetric per-layer quantizion
  std::shared_ptr<rk::nn::Tensor> AddQuant8VariableTensor(
      const std::string& name,
      int32_t* dimensions,
      uint32_t dimension_count,
      float quant_scale,
      int32_t zero_point);
  std::shared_ptr<rk::nn::Tensor> ConvertOperand(
      hal::Operand* operand, std::vector<int32_t> dimensions = {});

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
  // Map NNAdapter operand to rknpu tensor
  std::map<hal::Operand*, std::vector<std::shared_ptr<rk::nn::Tensor>>>
      tensors_;
  std::shared_ptr<rk::nn::Graph> graph_{nullptr};
  std::shared_ptr<rk::nn::Exection> execution_{nullptr};
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  std::string dump_graph_path_;
  std::vector<uint8_t>* dump_graph_buffer_{nullptr};
};

}  // namespace rockchip_npu
}  // namespace nnadapter
