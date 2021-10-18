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

#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

bool Converter::HasOperand(const std::string& name) {
  return operands_->find(name) != operands_->end();
}

NNAdapterOperand* Converter::GetOperand(std::string name) {
  CHECK(HasOperand(name)) << "Operand '" << name << "' is not found!";
  return (*operands_)[name].back();
}

NNAdapterOperand* Converter::AddOperand(NNAdapterOperand* operand,
                                        const std::string& name) {
  CHECK(operand);
  CHECK(!name.empty());
  (*operands_)[name].emplace_back(operand);
  return operand;
}

NNAdapterOperand* Converter::AddBool8ConstantOperand(bool value) {
  int8_t int8_value = value ? static_cast<int8_t>(1) : static_cast<int8_t>(0);
  return AddOperand(DDim(std::vector<int64_t>({})),
                    NNADAPTER_BOOL8,
                    nullptr,
                    0,
                    0,
                    &int8_value);
}

NNAdapterOperand* Converter::AddInt32ConstantOperand(int32_t value) {
  return AddOperand(
      DDim(std::vector<int64_t>({})), NNADAPTER_INT32, nullptr, 0, 0, &value);
}

NNAdapterOperand* Converter::AddInt64ConstantOperand(int64_t value) {
  return AddOperand(
      DDim(std::vector<int64_t>({})), NNADAPTER_INT64, nullptr, 0, 0, &value);
}

NNAdapterOperand* Converter::AddFloat32ConstantOperand(float value) {
  return AddOperand(
      DDim(std::vector<int64_t>({})), NNADAPTER_FLOAT32, nullptr, 0, 0, &value);
}

NNAdapterOperand* Converter::AddFloat64ConstantOperand(double value) {
  return AddOperand(
      DDim(std::vector<int64_t>({})), NNADAPTER_FLOAT64, nullptr, 0, 0, &value);
}

NNAdapterOperand* Converter::AddBool8ConstantOperand(bool* values,
                                                     const DDim& dimensions,
                                                     bool copy) {
  return AddOperand(dimensions, NNADAPTER_BOOL8, nullptr, 0, 0, values, copy);
}

NNAdapterOperand* Converter::AddInt32ConstantOperand(int32_t* values,
                                                     const DDim& dimensions,
                                                     bool copy) {
  return AddOperand(dimensions, NNADAPTER_INT32, nullptr, 0, 0, values, copy);
}

NNAdapterOperand* Converter::AddInt64ConstantOperand(int64_t* values,
                                                     const DDim& dimensions,
                                                     bool copy) {
  return AddOperand(dimensions, NNADAPTER_INT64, nullptr, 0, 0, values, copy);
}

NNAdapterOperand* Converter::AddFloat32ConstantOperand(float* values,
                                                       const DDim& dimensions,
                                                       bool copy) {
  return AddOperand(dimensions, NNADAPTER_FLOAT32, nullptr, 0, 0, values, copy);
}

NNAdapterOperand* Converter::AddFloat64ConstantOperand(double* values,
                                                       const DDim& dimensions,
                                                       bool copy) {
  return AddOperand(dimensions, NNADAPTER_FLOAT64, nullptr, 0, 0, values, copy);
}

NNAdapterOperand* Converter::AddQuant8ConstantOperand(int8_t* values,
                                                      const DDim& dimensions,
                                                      float quant_scale,
                                                      bool copy) {
  return AddOperand(dimensions,
                    NNADAPTER_QUANT_INT8_SYMM_PER_LAYER,
                    &quant_scale,
                    1,
                    0,
                    values,
                    copy);
}

NNAdapterOperand* Converter::AddQuant8ConstantOperand(
    int8_t* values,
    const DDim& dimensions,
    float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim,
    bool copy) {
  return AddOperand(dimensions,
                    NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
                    quant_scales,
                    quant_scale_count,
                    quant_channel_dim,
                    values,
                    copy);
}

NNAdapterOperand* Converter::AddQuant32ConstantOperand(int32_t* values,
                                                       const DDim& dimensions,
                                                       float quant_scale,
                                                       bool copy) {
  return AddOperand(dimensions,
                    NNADAPTER_QUANT_INT32_SYMM_PER_LAYER,
                    &quant_scale,
                    1,
                    0,
                    values,
                    copy);
}

NNAdapterOperand* Converter::AddQuant32ConstantOperand(
    int32_t* values,
    const DDim& dimensions,
    float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim,
    bool copy) {
  return AddOperand(dimensions,
                    NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL,
                    quant_scales,
                    quant_scale_count,
                    quant_channel_dim,
                    values,
                    copy);
}

NNAdapterOperand* Converter::AddQuant8VariableOperand(const DDim& dimensions,
                                                      float quant_scale,
                                                      const std::string& name) {
  return AddOperand(dimensions,
                    NNADAPTER_QUANT_INT8_SYMM_PER_LAYER,
                    &quant_scale,
                    1,
                    0,
                    nullptr,
                    false,
                    name);
}

NNAdapterOperand* Converter::AddConstantOperand(const Tensor* tensor) {
  auto tensor_precision = tensor->precision();
  void* tensor_data = const_cast<void*>(tensor->raw_data());
  auto nnadapter_tensor_precison =
      Precision2NNAdapterTensorPrecisionCode(tensor_precision);
  auto tensor_dims = tensor->dims();
  return AddOperand(
      tensor_dims, nnadapter_tensor_precison, nullptr, 0, 0, tensor_data, true);
}

NNAdapterOperand* Converter::AddOperand(const Tensor* tensor,
                                        const std::string& name) {
  auto tensor_precision = tensor->precision();
  bool is_const_tensor = tensor->persistable();
  auto dims = tensor->dims();
  if (is_const_tensor) {
    return AddConstantOperand(tensor);
  } else {
    auto nnadapter_tensor_precison =
        Precision2NNAdapterTensorPrecisionCode(tensor_precision);
    auto dims = tensor->dims();
    return AddVariableOperand(dims, name, nnadapter_tensor_precison);
  }
}

NNAdapterOperand* Converter::AddVariableOperand(
    const DDim& dimensions,
    const std::string& name,
    NNAdapterOperandPrecisionCode precision) {
  return AddOperand(dimensions, precision, nullptr, 0, 0, nullptr, false, name);
}

NNAdapterOperand* Converter::AddFloat32VariableOperand(
    const DDim& dimensions, const std::string& name) {
  return AddVariableOperand(dimensions, name, NNADAPTER_FLOAT32);
}

NNAdapterOperand* Converter::AddFloat64VariableOperand(
    const DDim& dimensions, const std::string& name) {
  return AddVariableOperand(dimensions, name, NNADAPTER_FLOAT64);
}

NNAdapterOperand* Converter::AddInt32VariableOperand(const DDim& dimensions,
                                                     const std::string& name) {
  return AddVariableOperand(dimensions, name, NNADAPTER_INT32);
}

NNAdapterOperand* Converter::AddInt64VariableOperand(const DDim& dimensions,
                                                     const std::string& name) {
  return AddVariableOperand(dimensions, name, NNADAPTER_INT64);
}

NNAdapterOperation* Converter::AddOperation(
    NNAdapterOperationType type,
    std::vector<NNAdapterOperand*>* input_operands,
    std::vector<NNAdapterOperand*>* output_operands) {
  NNAdapterOperation* operation = nullptr;
  NNAdapterModel_addOperation_invoke(model_,
                                     type,
                                     input_operands->size(),
                                     input_operands->data(),
                                     output_operands->size(),
                                     output_operands->data(),
                                     &operation);
  return operation;
}

NNAdapterOperand* Converter::AddOperand(NNAdapterOperandType* type,
                                        const std::string& name) {
  NNAdapterOperand* operand = nullptr;
  NNAdapterModel_addOperand_invoke(model_, type, &operand);
  if (!name.empty()) {
    if (HasOperand(name)) {
      (*operands_)[name].emplace_back(operand);
    } else {
      (*operands_)[name] = std::vector<NNAdapterOperand*>{operand};
    }
  }
  return operand;
}

void Converter::SetOperandValue(NNAdapterOperand* operand,
                                void* buffer,
                                size_t length,
                                bool copy) {
  NNAdapterModel_setOperandValue_invoke(operand, buffer, length, copy);
}

NNAdapterOperand* Converter::AddOperand(const DDim& dimensions,
                                        NNAdapterOperandPrecisionCode precision,
                                        float* quant_scales,
                                        uint32_t quant_scale_count,
                                        uint32_t quant_channel_dim,
                                        void* buffer,
                                        bool copy,
                                        const std::string& name) {
  NNAdapterOperandType type;
  memset(&type, 0, sizeof(NNAdapterOperandType));
  bool is_scalar = dimensions.size() == 0;
  if (!is_scalar) {
    ConvertDimensions(dimensions, type.dimensions.data, &type.dimensions.count);
  }
  type.precision = precision;
  if (quant_scales && quant_scale_count > 0) {
    // Quant type
    if (quant_scale_count > 1) {
      // Symmetric per-channel quantization
      CHECK(precision == NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL ||
            precision == NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL);
      type.symm_per_channel_params.scales = quant_scales;
      type.symm_per_channel_params.scale_count = quant_scale_count;
      type.symm_per_channel_params.channel_dim = quant_channel_dim;
    } else {
      // Symmetric per-layer quantization
      CHECK(precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER ||
            precision == NNADAPTER_QUANT_INT32_SYMM_PER_LAYER);
      type.symm_per_layer_params.scale = quant_scales[0];
    }
  } else {
    // Basic type, without any quantization parameters
  }
  auto operand = AddOperand(&type, name);
  if (buffer) {
    // Constant operand
    auto length =
        PrecisionLength(precision) * (!is_scalar ? dimensions.production() : 1);
    SetOperandValue(operand, buffer, length, copy);
  } else {
    // Variable/Input/Output operand
  }
  return operand;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
