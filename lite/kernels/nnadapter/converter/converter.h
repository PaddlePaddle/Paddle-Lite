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

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/nnadapter/nnadapter_wrapper.h"
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"
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
  explicit Converter(NNAdapterModel* model, ::NNAdapterContext* context)
      : model_(model), context_(context) {}
  ~Converter() {}

  // Convert a block_desc with tensors to a NNAdapter model
  int Apply(const cpp::BlockDesc* block_desc,
            Scope* exec_scope,
            const std::vector<Variable>& input_vars,
            std::vector<Variable>* output_vars);

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
  NNAdapterOperand* AddInputOperand(Scope* scope,
                                    const std::string& input_name,
                                    DDim dimensions = {},
                                    const std::vector<float>& quant_scales = {},
                                    uint32_t quant_channel_dim = 0);
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
  // Add a operation with input and output operands
  NNAdapterOperation* AddOperation(
      NNAdapterOperationType type,
      std::vector<NNAdapterOperand*> input_operands,
      std::vector<NNAdapterOperand*> output_operands);
  // Add shape operation with input operand, output name, output precision
  NNAdapterOperand* AddShapeOperation(
      NNAdapterOperand* input_operand,
      const std::string& output_name = "",
      NNAdapterOperandPrecisionCode output_precision = NNADAPTER_INT32);
  // Add unsqueeze operation with input operand, axes, output_name, quant_scales
  NNAdapterOperand* AddUnsqueezeOperation(
      NNAdapterOperand* input_operand,
      const std::vector<int32_t>& axes,
      const std::string& output_name = "",
      const std::vector<float>& output_quant_scales = {},
      uint32_t output_quant_channel_dim = 0);
  // Add squeeze operation with input operand, axes, output_name, quant_scales
  NNAdapterOperand* AddSqueezeOperation(
      NNAdapterOperand* input_operand,
      const std::vector<int32_t>& axes,
      const std::string& output_name = "",
      const std::vector<float>& output_quant_scales = {},
      uint32_t output_quant_channel_dim = 0);
  // Add slice operation with input operand, axes, starts, ends, steps,
  // output_name, quant_scales
  NNAdapterOperand* AddSliceOperation(
      NNAdapterOperand* input_operand,
      const std::vector<int32_t>& axes,
      const std::vector<int32_t>& starts,
      const std::vector<int32_t>& ends,
      const std::vector<int32_t>& steps,
      const std::string& output_name = "",
      const std::vector<float>& output_quant_scales = {},
      uint32_t output_quant_channel_dim = 0);
  // Add reshape operation with input operand, shape, output_name, quant_scales
  NNAdapterOperand* AddReshapeOperation(
      NNAdapterOperand* input_operand,
      const std::vector<int32_t>& shape,
      const std::string& output_name = "",
      const std::vector<float>& output_quant_scales = {},
      uint32_t output_quant_channel_dim = 0);
  // Add flatten operation with input operand, start, end, output_name
  NNAdapterOperand* AddFlattenOperation(
      NNAdapterOperand* input_operand,
      const int32_t start_axis,
      const int32_t end_axis,
      const std::string& output_name = "",
      const std::vector<float>& output_quant_scales = {},
      uint32_t output_quant_channel_dim = 0);
  // Unpack the fused activations
  NNAdapterOperand* UnpackFusedActivations(
      NNAdapterOperand* input_operand,
      const std::string& act_type,
      OpInfo* op,
      Scope* scope,
      const std::string& output_name = "",
      const std::vector<float>& output_quant_scales = {},
      uint32_t output_quant_channel_dim = 0);

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
  ::NNAdapterContext* context_{nullptr};
  std::map<std::string, std::vector<NNAdapterOperand*>> operands_;
  size_t operation_count_{0};
};

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
