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

#include "runtime/model.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace runtime {

Model::~Model() {
  for (auto& operand : model_.operands) {
    if (operand.type.lifetime == NNADAPTER_CONSTANT_COPY && operand.buffer) {
      free(operand.buffer);
    }
    if ((operand.type.precision ==
             NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL ||
         operand.type.precision ==
             NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL) &&
        operand.type.symm_per_channel_params.scales) {
      free(operand.type.symm_per_channel_params.scales);
    }
  }
}

int Model::AddOperand(const NNAdapterOperandType& type,
                      hal::Operand** operand) {
  *operand = nnadapter::AddOperand(&model_);
  memcpy(&(*operand)->type, &type, sizeof(NNAdapterOperandType));
  if (type.precision == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL ||
      type.precision == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL) {
    uint32_t scale_size =
        type.symm_per_channel_params.scale_count * sizeof(float);
    float* scales = reinterpret_cast<float*>(malloc(scale_size));
    NNADAPTER_CHECK(scales)
        << "Failed to allocate the scale buffer for a operand.";
    memcpy(scales, type.symm_per_channel_params.scales, scale_size);
    (*operand)->type.symm_per_channel_params.scales = scales;
  }
  return NNADAPTER_NO_ERROR;
}

int Model::AddOperation(NNAdapterOperationType type,
                        hal::Operation** operation) {
  *operation = nnadapter::AddOperation(&model_);
  (*operation)->type = type;
  return NNADAPTER_NO_ERROR;
}

int Model::IdentifyInputsAndOutputs(uint32_t input_count,
                                    hal::Operand** input_operands,
                                    uint32_t output_count,
                                    hal::Operand** output_operands) {
  model_.input_operands.resize(input_count);
  for (uint32_t i = 0; i < input_count; i++) {
    model_.input_operands[i] = input_operands[i];
    model_.input_operands[i]->type.lifetime = NNADAPTER_MODEL_INPUT;
  }
  model_.output_operands.resize(output_count);
  for (uint32_t i = 0; i < output_count; i++) {
    model_.output_operands[i] = output_operands[i];
    model_.output_operands[i]->type.lifetime = NNADAPTER_MODEL_OUTPUT;
  }
  return NNADAPTER_NO_ERROR;
}

int Model::Finish() {
  // TODO(hong19860320) model validation
  completed_ = true;
  return NNADAPTER_NO_ERROR;
}

}  // namespace runtime
}  // namespace nnadapter
