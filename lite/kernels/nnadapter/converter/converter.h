// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/engine.h"
#include "lite/kernels/nnadapter/utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

const int NO_ERROR = 0;
const int PARAMETER_ERROR = 1;
const int UNSUPPORTED_FEATURE = 2;

class Converter {
 public:
  explicit Converter(NNAdapterModel* model) : model_(model) {
    sub_converter.reset(new subgraph::nnadapter::Converter(model_, &operands_));
  }
  ~Converter() {}

  // Convert a block_desc with tensors to a NNAdapter model
  int Apply(int block_idx,
            const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
            Scope* exec_scope,
            const std::vector<Variable>& input_vars,
            std::vector<Variable>* output_vars,
            std::vector<NNAdapterOperand*>* input_operands,
            std::vector<NNAdapterOperand*>* output_operands);

  // Mapping a string name to a operand
  NNAdapterOperand* GetMappedOperand(const std::string& name);
  NNAdapterOperand* UpdateOperandMap(const std::string& name,
                                     NNAdapterOperand* operand);
  // Add a constant operand from the scalar value or vector values
  // Set quant_scales to a symmetric per-layer quantizion operand if
  // quant_scales is not empty
  // Set quant_scales and quant_channel_dim to create a symmetric per-channel
  // quantizion operand
  template <typename T>
  NNAdapterOperand* AddConstantOperand(
      const T* values,
      const DDim& dimensions,
      bool copy = true,
      const std::vector<float>& quant_scales = {},
      uint32_t quant_channel_dim = 0) {
    auto precision_type = ConvertPODTypeToPrecisionType<T>();
    return AddOperand(dimensions,
                      precision_type,
                      quant_scales.data(),
                      quant_scales.size(),
                      quant_channel_dim,
                      values,
                      copy);
  }
  template <typename T>
  NNAdapterOperand* AddConstantOperand(
      T value,
      const std::vector<float>& quant_scales = {},
      uint32_t quant_channel_dim = 0) {
    return AddConstantOperand(&value,
                              DDim(std::vector<int64_t>({1})),
                              lite_api::PrecisionTypeTrait<T>::Type(),
                              true,
                              quant_scales,
                              quant_channel_dim);
  }
  template <typename T>
  NNAdapterOperand* AddConstantOperand(
      const std::vector<T>& values,
      DDim dimensions = {},
      const std::vector<float>& quant_scales = {},
      uint32_t quant_channel_dim = 0) {
    if (dimensions.empty()) {
      dimensions = DDim({static_cast<int64_t>(values.size())});
    }
    return AddConstantOperand(values.data(),
                              dimensions,
                              lite_api::PrecisionTypeTrait<T>::Type(),
                              true,
                              quant_scales,
                              quant_channel_dim);
  }
  // Add a constant operand from a tensor
  NNAdapterOperand* AddConstantOperand(
      const Tensor& tensor,
      DDim dimensions = {},
      bool copy = false,
      const std::vector<float>& quant_scales = {},
      uint32_t quant_channel_dim = 0);
  NNAdapterOperand* AddConstantOperand(
      const void* buffer,
      const DDim& dimensions,
      PrecisionType precision_type,
      bool copy = true,
      const std::vector<float>& quant_scales = {},
      uint32_t quant_channel_dim = 0);
  NNAdapterOperand* AddConstantOperand(
      const void* buffer,
      const DDim& dimensions,
      NNAdapterOperandPrecisionCode precision_code,
      bool copy = true,
      const std::vector<float>& quant_scales = {},
      uint32_t quant_channel_dim = 0);
  // Add a named input operand, should specify its dimensions and precision
  NNAdapterOperand* AddInputOperand(
      const std::string& name,
      const DDim& dimensions,
      const std::vector<std::vector<int64_t>>& dynamic_dimensions,
      PrecisionType precision_type,
      const std::vector<float>& quant_scales = {},
      uint32_t quant_channel_dim = 0);
  NNAdapterOperand* AddInputOperand(
      const std::string& name,
      const DDim& dimensions,
      const std::vector<std::vector<int64_t>>& dynamic_dimensions,
      NNAdapterOperandPrecisionCode precision_code,
      const std::vector<float>& quant_scales = {},
      uint32_t quant_channel_dim = 0);
  // Add a named variable operand, its dimension and precision can be infered
  // after AddOperation() is called
  NNAdapterOperand* AddOutputOperand(
      const std::string& name,
      const std::vector<float>& quant_scales = {},
      uint32_t quant_channel_dim = 0);
  // Add an anonymous variable operand, its dimension and precision can be
  // infered after AddOperation() is called
  NNAdapterOperand* AddOutputOperand(
      const std::vector<float>& quant_scales = {},
      uint32_t quant_channel_dim = 0);
  // Get the type of a operand, which includes precision, dimension and
  // quantization parameters
  const NNAdapterOperandType* GetOperandType(NNAdapterOperand* operand);
  // Add a operation with input and output operands
  NNAdapterOperation* AddOperation(
      NNAdapterOperationType type,
      std::vector<NNAdapterOperand*>* input_operands,
      std::vector<NNAdapterOperand*>* output_operands);
  // Add shape operation with input name and output name
  NNAdapterOperand* AddShapeOperation(
      const std::string& input_name,
      const std::string& output_name,
      NNAdapterOperandPrecisionCode output_precision = NNADAPTER_TENSOR_INT32);

 private:
  // Add a operand from a NNAdapter type, only for internal use
  NNAdapterOperand* AddOperand(NNAdapterOperandType* type,
                               const std::string& name = "");
  // Add a operand, only for internal use
  NNAdapterOperand* AddOperand(
      const DDim& dimensions,
      PrecisionType precision_type,
      const float* quant_scales = nullptr,
      uint32_t quant_scale_count = 0,
      uint32_t quant_channel_dim = 0,
      const void* buffer = nullptr,
      bool copy = true,
      const std::string& name = "",
      const std::vector<std::vector<int64_t>>& dynamic_dimensions = {});
  // Set the value of a constant operand
  void SetOperandValue(NNAdapterOperand* operand,
                       const void* buffer,
                       size_t length,
                       bool copy = true);
  NNAdapterModel* model_{nullptr};
  std::map<std::string, std::vector<NNAdapterOperand*>> operands_;
  std::shared_ptr<subgraph::nnadapter::Converter> sub_converter{nullptr};
};

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
