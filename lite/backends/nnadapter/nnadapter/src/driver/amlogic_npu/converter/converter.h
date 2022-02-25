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
#include "driver/amlogic_npu/utility.h"

namespace nnadapter {
namespace amlogic_npu {

class Converter {
 public:
  explicit Converter(
      aml::nn::Graph* graph,
      std::map<core::Operand*, std::vector<std::shared_ptr<aml::nn::Tensor>>>*
          tensors)
      : graph_(graph), tensors_(tensors) {}
  ~Converter() {}

  // Convert a NNAdapter model to aml graph and tensors
  int Apply(core::Model* model);
  // Mapping a aml tensor to a NNAdapter operand
  std::string GetTensorName(core::Operand* operand);
  std::shared_ptr<aml::nn::Tensor> GetMappedTensor(core::Operand* operand);
  std::shared_ptr<aml::nn::Tensor> UpdateTensorMap(
      core::Operand* operand, std::shared_ptr<aml::nn::Tensor> tensor);
  // Create and add a named or anonymous aml tensor into the tensor pool
  std::shared_ptr<aml::nn::Tensor> AddTensor(
      const std::string& name,
      int32_t* dimensions_data,
      uint32_t dimensions_count,
      aml::nn::PrecisionType precision,
      const float* quant_scale = nullptr,
      const int32_t* zero_point = nullptr,
      void* buffer = nullptr,
      aml::nn::DataLayoutType layout = aml::nn::DataLayoutType::NCHW);
  std::shared_ptr<aml::nn::Tensor> AddTensor(
      const std::string& name,
      const NNAdapterOperandType* type,
      void* buffer = nullptr,
      std::vector<int32_t> dimensions = {});
  std::shared_ptr<aml::nn::Tensor> AddConstantTensor(
      void* values,
      int32_t* dimensions_data,
      uint32_t dimensions_count,
      aml::nn::PrecisionType precision,
      const float* quant_scale = nullptr,
      const int32_t* zero_point = nullptr);
  std::shared_ptr<aml::nn::Tensor> AddVariableTensor(
      const std::string& name,
      int32_t* dimensions_data,
      uint32_t dimensions_count,
      aml::nn::PrecisionType precision,
      const float* quant_scale = nullptr,
      const int32_t* zero_point = nullptr);
  // Quant8 constant operand with asymmetric per-layer quantizion
  std::shared_ptr<aml::nn::Tensor> AddQuant8ConstantTensor(
      uint8_t* values,
      int32_t* dimensions_data,
      uint32_t dimensions_count,
      float quant_scale,
      int32_t zero_point);
  // Quant32 constant operand with symmetric per-layer quantizion
  std::shared_ptr<aml::nn::Tensor> AddQuant32ConstantTensor(
      int32_t* values,
      int32_t* dimensions_data,
      uint32_t dimensions_count,
      float quant_scale);
  // Quant8 variable operand with asymmetric per-layer quantizion
  std::shared_ptr<aml::nn::Tensor> AddQuant8VariableTensor(
      const std::string& name,
      int32_t* dimensions_data,
      uint32_t dimensions_count,
      float quant_scale,
      int32_t zero_point);
  // Convert a NNAdapter operand to a aml tensor
  std::shared_ptr<aml::nn::Tensor> ConvertOperand(
      core::Operand* operand, std::vector<int32_t> dimensions = {});
  // Add a aml operator into aml graph
  std::shared_ptr<aml::nn::Operator> AddOperator(
      aml::nn::OperatorType type,
      std::vector<std::shared_ptr<aml::nn::Tensor>> input_tensors,
      std::vector<std::shared_ptr<aml::nn::Tensor>> output_tensors,
      void* attrs,
      std::string name = "");

 private:
  aml::nn::Graph* graph_{nullptr};
  std::map<core::Operand*, std::vector<std::shared_ptr<aml::nn::Tensor>>>*
      tensors_;
};

}  // namespace amlogic_npu
}  // namespace nnadapter
