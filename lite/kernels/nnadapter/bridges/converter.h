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
#include <string>
#include <vector>
#include "lite/backends/nnadapter/nnadapter_wrapper.h"
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

class Converter {
 public:
  explicit Converter(
      NNAdapterModel* model,
      std::map<std::string, std::vector<NNAdapterOperand*>>* operands = nullptr)
      : model_(model), operands_(operands) {}
  ~Converter() {}

  // NNAdapter operand
  bool HasOperand(const std::string& name);
  NNAdapterOperand* GetOperand(std::string name);
  NNAdapterOperand* AddOperand(NNAdapterOperand* operand,
                               const std::string& name);
  // Scalar or tensor constant operand with basic type
  NNAdapterOperand* AddBool8ConstantOperand(bool value);
  NNAdapterOperand* AddInt32ConstantOperand(int32_t value);
  NNAdapterOperand* AddInt64ConstantOperand(int64_t value);
  NNAdapterOperand* AddFloat32ConstantOperand(float value);
  NNAdapterOperand* AddFloat64ConstantOperand(double value);
  NNAdapterOperand* AddBool8ConstantOperand(bool* values,
                                            const DDim& dimensions,
                                            bool copy = true);
  NNAdapterOperand* AddInt32ConstantOperand(int32_t* values,
                                            const DDim& dimensions,
                                            bool copy = true);
  NNAdapterOperand* AddInt64ConstantOperand(int64_t* values,
                                            const DDim& dimensions,
                                            bool copy = true);
  NNAdapterOperand* AddFloat32ConstantOperand(float* values,
                                              const DDim& dimensions,
                                              bool copy = true);
  NNAdapterOperand* AddFloat64ConstantOperand(double* values,
                                              const DDim& dimensions,
                                              bool copy = true);
  // Quant8 constant operand with symmetric per-layer quantizion
  NNAdapterOperand* AddQuant8ConstantOperand(int8_t* values,
                                             const DDim& dimensions,
                                             float quant_scale,
                                             bool copy = true);
  // Quant8 constant operand with symmetric per-channel quantizion
  NNAdapterOperand* AddQuant8ConstantOperand(int8_t* values,
                                             const DDim& dimensions,
                                             float* quant_scales,
                                             uint32_t quant_scale_count,
                                             uint32_t quant_channel_dim = 0,
                                             bool copy = true);
  // Quant32 constant operand with symmetric per-layer quantizion
  NNAdapterOperand* AddQuant32ConstantOperand(int32_t* values,
                                              const DDim& dimensions,
                                              float quant_scale,
                                              bool copy = true);
  // Quant32 constant operand with symmetric per-channel quantizion
  NNAdapterOperand* AddQuant32ConstantOperand(int32_t* values,
                                              const DDim& dimensions,
                                              float* quant_scales,
                                              uint32_t quant_scale_count,
                                              uint32_t quant_channel_dim = 0,
                                              bool copy = true);
  // Quant8 variable operand with symmetric per-layer quantizion
  NNAdapterOperand* AddQuant8VariableOperand(const DDim& dimensions,
                                             float quant_scale,
                                             const std::string& name = "");
  NNAdapterOperand* AddFloat32VariableOperand(const DDim& dimensions,
                                              const std::string& name = "");
  NNAdapterOperand* AddFloat64VariableOperand(const DDim& dimensions,
                                              const std::string& name = "");
  NNAdapterOperand* AddInt32VariableOperand(const DDim& dimensions,
                                            const std::string& name = "");
  NNAdapterOperand* AddInt64VariableOperand(const DDim& dimensions,
                                            const std::string& name = "");
  NNAdapterOperand* AddVariableOperand(
      const DDim& dimensions,
      const std::string& name = "",
      NNAdapterOperandPrecisionCode precision = NNADAPTER_FLOAT32);
  NNAdapterOperand* AddConstantOperand(const Tensor* tensor);
  NNAdapterOperand* AddOperand(const Tensor* tensor, const std::string& name);
  // NNAdapter operation
  NNAdapterOperation* AddOperation(
      NNAdapterOperationType type,
      std::vector<NNAdapterOperand*>* input_operands,
      std::vector<NNAdapterOperand*>* output_operands);

 private:
  NNAdapterOperand* AddOperand(NNAdapterOperandType* type,
                               const std::string& name = "");
  void SetOperandValue(NNAdapterOperand* operand,
                       void* buffer,
                       size_t length,
                       bool copy = true);
  NNAdapterOperand* AddOperand(const DDim& dimensions,
                               NNAdapterOperandPrecisionCode precision,
                               float* quant_scales = nullptr,
                               uint32_t quant_scale_count = 0,
                               uint32_t quant_channel_dim = 0,
                               void* buffer = nullptr,
                               bool copy = true,
                               const std::string& name = "");
  NNAdapterModel* model_{nullptr};
  std::map<std::string, std::vector<NNAdapterOperand*>>* operands_;
};

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
