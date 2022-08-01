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
#include <memory>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace runtime {

Model::~Model() { ClearModel(&model_); }

int Model::AddOperand(const NNAdapterOperandType& type,
                      core::Operand** operand) {
  *operand = nnadapter::AddOperand(&model_);
  memcpy(&(*operand)->type, &type, sizeof(NNAdapterOperandType));
  if (IsPerChannelQuantType(type.precision)) {
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
                        core::Operation** operation) {
  *operation = nnadapter::AddOperation(&model_);
  (*operation)->type = type;
  return NNADAPTER_NO_ERROR;
}

int Model::IdentifyInputsAndOutputs(uint32_t input_count,
                                    core::Operand** input_operands,
                                    uint32_t output_count,
                                    core::Operand** output_operands) {
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

int Model::GetSupportedOperations(Context::DeviceContext* device_context,
                                  bool* supported_operations) const {
  auto operation_count = model_.operations.size();
  std::fill(
      supported_operations, supported_operations + operation_count, false);
  NNADAPTER_CHECK(device_context);
  auto context = device_context->context;
  NNADAPTER_CHECK(context);
  auto device = device_context->device;
  NNADAPTER_CHECK(device);
  auto result = device->ValidateProgram(context, &model_, supported_operations);
  if (result == NNADAPTER_FEATURE_NOT_SUPPORTED) {
    NNADAPTER_LOG(WARNING)
        << "Warning: Failed to get the supported operations for device '"
        << device->GetName() << "', because the HAL interface "
                                "'validate_program' is not implemented!";
  }
  return result;
}

int Model::GetSupportedOperations(Context* context,
                                  bool* supported_operations) const {
  auto operation_count = model_.operations.size();
  std::fill(
      supported_operations, supported_operations + operation_count, false);
  auto device_count = context->GetDeviceCount();
  for (size_t i = 0; i < device_count; i++) {
    auto device_context = context->GetDeviceContext(i);
    NNADAPTER_CHECK(device_context);
    std::unique_ptr<bool[]> _supported_operations_(new bool[operation_count]);
    auto result =
        GetSupportedOperations(device_context, _supported_operations_.get());
    if (result != NNADAPTER_NO_ERROR) {
      return result;
    }
    for (size_t j = 0; j < operation_count; j++) {
      supported_operations[j] |= _supported_operations_[j];
    }
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace runtime
}  // namespace nnadapter
