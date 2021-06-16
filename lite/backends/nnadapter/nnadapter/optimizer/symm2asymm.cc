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

#include "optimizer/symm2asymm.h"
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

static void ConvertOperandFromSymmToAsymm(hal::Operand* operand,
                                          int32_t zero_point) {
  switch (operand->type.precision) {
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER: {
      operand->type.precision = NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER;
      auto scale = operand->type.symm_per_layer_params.scale;
      operand->type.asymm_per_layer_params = {.scale = scale,
                                              .zero_point = zero_point};
      auto is_constant_copy = operand->type.lifetime == NNADAPTER_CONSTANT_COPY;
      auto is_constant_reference =
          operand->type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
      if (zero_point != 0 && (is_constant_copy || is_constant_reference)) {
        auto transform_buffer = static_cast<uint8_t*>(operand->buffer);
        if (is_constant_reference) {
          transform_buffer = static_cast<uint8_t*>(malloc(operand->length));
        }
        for (uint32_t i = 0; i < operand->length; i++) {
          transform_buffer[i] = static_cast<uint8_t>(
              static_cast<int16_t>(static_cast<int8_t*>(operand->buffer)[i]) +
              zero_point);
        }
        if (is_constant_reference) {
          operand->buffer = transform_buffer;
          operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
        }
      }
    } break;
    default:
      break;
  }
}

static void PropagateAsymmZeroPoint(hal::Operand* reference_operand,
                                    hal::Operand* target_operand) {
  auto& reference_type = reference_operand->type;
  auto& target_type = target_operand->type;
  auto reference_precision = reference_type.precision;
  auto target_precision = target_type.precision;
  if (IsAsymmPerLayerQuantization(reference_precision) &&
      IsAsymmPerLayerQuantization(target_precision)) {
    target_type.asymm_per_layer_params.zero_point =
        reference_type.asymm_per_layer_params.zero_point;
  } else {
    NNADAPTER_LOG(FATAL) << "Unhandled case: reference_precision="
                         << OperandPrecisionCodeToString(
                                reference_type.precision)
                         << ", target_precision="
                         << OperandPrecisionCodeToString(target_precision);
  }
}

NNADAPTER_EXPORT void ConvertModelFromSymmToAsymmQuantization(
    hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    auto& input_operands = operation->input_operands;
    auto& output_operands = operation->output_operands;
    auto input_count = input_operands.size();
    auto output_count = output_operands.size();
    switch (operation->type) {
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_RESHAPE:
      case NNADAPTER_TANH:
      case NNADAPTER_TRANSPOSE:
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH: {
        ConvertOperandFromSymmToAsymm(input_operands[0], 128);
        ConvertOperandFromSymmToAsymm(output_operands[0], 128);
        PropagateAsymmZeroPoint(input_operands[0], output_operands[0]);
      } break;
      case NNADAPTER_SIGMOID:
      case NNADAPTER_SOFTMAX: {
        ConvertOperandFromSymmToAsymm(input_operands[0], 128);
        // The zeroPoint of the output of softmax and sigmoid must be 0.
        ConvertOperandFromSymmToAsymm(output_operands[0], 0);
      } break;
      case NNADAPTER_ADD:
      case NNADAPTER_DIV:
      case NNADAPTER_FULLY_CONNECTED:
      case NNADAPTER_MUL:
      case NNADAPTER_SUB: {
        ConvertOperandFromSymmToAsymm(input_operands[0], 128);
        ConvertOperandFromSymmToAsymm(input_operands[1], 128);
        ConvertOperandFromSymmToAsymm(output_operands[0], 128);
      } break;
      case NNADAPTER_CONV_2D: {
        ConvertOperandFromSymmToAsymm(input_operands[0], 128);
        ConvertOperandFromSymmToAsymm(input_operands[1], 128);
        ConvertOperandFromSymmToAsymm(input_operands[2], 128);
        ConvertOperandFromSymmToAsymm(output_operands[0], 128);
      } break;
      case NNADAPTER_CONCAT: {
        NNADAPTER_CHECK_GE(input_count, 2);
        for (int i = 0; i < input_count - 1; i++) {
          ConvertOperandFromSymmToAsymm(input_operands[i], 128);
        }
        ConvertOperandFromSymmToAsymm(output_operands[0], 128);
      } break;
      default:
        NNADAPTER_LOG(FATAL)
            << "Missing the processing of "
            << OperationTypeToString(operation->type)
            << " for the conversion of symm2asymm quantization.";
        break;
    }
  }
}

}  // namespace nnadapter
