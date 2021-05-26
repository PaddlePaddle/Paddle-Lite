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
#include <vector>
#include "../../nnadapter_driver.h"   // NOLINT
#include "../../nnadapter_logging.h"  // NOLINT
#include "context.h"                  // NOLINT
#include "neuron_adapter_wrapper.h"   // NOLINT

namespace nnadapter {
namespace driver {
namespace mediatek_apu {

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program();

  int Build(driver::Model* model, driver::Cache* cache);
  int Execute(uint32_t input_count,
              driver::Argument* input_arguments,
              uint32_t output_count,
              driver::Argument* output_arguments);

 private:
  // Utilities for building Neuron model
  template <typename T>
  uint32_t AddScalarConstantOperand(T value, int32_t code) {
    NeuronOperandType type;
    memset(&type, 0, sizeof(NeuronOperandType));
    type.type = code;
    type.dimensionCount = 0;
    NeuronModel_addOperand_invoke(model_, &type);
    auto index = operand_index_++;
    NNADAPTER_CHECK_EQ(
        NeuronModel_setOperandValue_invoke(model_, index, &value, sizeof(T)),
        NNADAPTER_NO_ERROR);
    return index;
  }
  template <typename T>
  uint32_t AddVectorConstantOperand(const T* values,
                                    uint32_t num_values,
                                    int32_t code,
                                    float scale,
                                    int32_t zero_point) {
    NeuronOperandType type;
    memset(&type, 0, sizeof(NeuronOperandType));
    type.type = code;
    type.dimensionCount = 1;
    type.dimensions = &num_values;
    type.scale = scale;
    type.zeroPoint = zero_point;
    NeuronModel_addOperand_invoke(model_, &type);
    auto index = operand_index_++;
    NNADAPTER_CHECK_EQ(NeuronModel_setOperandValue_invoke(
                           model_, index, values, sizeof(T) * num_values),
                       NNADAPTER_NO_ERROR);
    return index;
  }
  template <typename T>
  uint32_t AddVectorConstantOperand(const T* values,
                                    uint32_t num_values,
                                    int32_t code) {
    return AddVectorConstantOperand(
        values, num_values, code, /*scale=*/0.f, /*zero_point=*/0);
  }
  uint32_t AddScalarInt32ConstantOperand(int32_t value);
  uint32_t AddScalarFloat32ConstantOperand(float value);
  uint32_t AddVectorInt32ConstantOperand(const int32_t* values,
                                         uint32_t num_values);
  uint32_t AddVectorInt32ConstantOperand(const int32_t* values,
                                         uint32_t num_values,
                                         float scale,
                                         int32_t zero_point);
  uint32_t AddVectorFloat32ConstantOperand(const float* values,
                                           uint32_t num_values);
  // Operation converters
  uint32_t ConvertOperand(driver::Operand* operand);
  int ConvertConv2D(driver::Operation* operation);
  int ConvertFullyConnected(driver::Operation* operation);
  int ConvertAverageAndMaxPool2D(driver::Operation* operation);
  int ConvertElementwiseBinaryOperations(driver::Operation* operation);
  int ConvertSoftmax(driver::Operation* operation);
  int ConvertActivationUnaryOperations(driver::Operation* operation);
  int ConvertTranspose(driver::Operation* operation);

 private:
  Context* context_{nullptr};
  // NNAdapter operand to the index of Neuron operand
  std::map<driver::Operand*, uint32_t> operand_indexes_;
  uint32_t operand_index_{0};
  std::vector<void*> operand_buffers_;
  NeuronModel* model_{nullptr};
  NeuronCompilation* compilation_{nullptr};
  NeuronExecution* execution_{nullptr};
  std::vector<int32_t> input_zero_points_;
  std::vector<int32_t> output_zero_points_;
};

}  // namespace mediatek_apu
}  // namespace driver
}  // namespace nnadapter
