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

#include "optimizer/convert_quantization_symm_to_asymm.h"
#include <algorithm>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

static void ConvertOperandSymmToAsymm(core::Operand* operand,
                                      int32_t zero_point) {
  switch (operand->type.precision) {
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER: {
      operand->type.precision = NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER;
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
          transform_buffer[i] = static_cast<uint8_t>(std::min(
              std::max(static_cast<int32_t>(
                           reinterpret_cast<int8_t*>(operand->buffer)[i]) +
                           zero_point,
                       0),
              255));
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

static void PropagateAsymmZeroPoint(core::Operand* reference_operand,
                                    core::Operand* target_operand) {
  auto& reference_type = reference_operand->type;
  auto& target_type = target_operand->type;
  if (IsAsymmPerLayerQuantType(reference_type.precision) &&
      IsAsymmPerLayerQuantType(target_type.precision)) {
    target_type.asymm_per_layer_params.zero_point =
        reference_type.asymm_per_layer_params.zero_point;
  }
}

NNADAPTER_EXPORT void ConvertQuantizationSymmToAsymm(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    auto& input_operands = operation->input_operands;
    auto& output_operands = operation->output_operands;
    auto input_count = input_operands.size();
    auto output_count = output_operands.size();
    switch (operation->type) {
      case NNADAPTER_ADD:
      case NNADAPTER_DIV:
      case NNADAPTER_EQUAL:
      case NNADAPTER_FULLY_CONNECTED:
      case NNADAPTER_GATHER:
      case NNADAPTER_GREATER:
      case NNADAPTER_GREATER_EQUAL:
      case NNADAPTER_LESS:
      case NNADAPTER_LESS_EQUAL:
      case NNADAPTER_MAT_MUL:
      case NNADAPTER_MAX:
      case NNADAPTER_MIN:
      case NNADAPTER_MUL:
      case NNADAPTER_NOT_EQUAL:
      case NNADAPTER_POW:
      case NNADAPTER_SUB: {
        ConvertOperandSymmToAsymm(input_operands[0], 128);
        ConvertOperandSymmToAsymm(input_operands[1], 128);
        ConvertOperandSymmToAsymm(output_operands[0], 128);
      } break;
      case NNADAPTER_ABS:
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_BATCH_NORMALIZATION:
      case NNADAPTER_CAST:
      case NNADAPTER_CHANNEL_SHUFFLE:
      case NNADAPTER_CLIP:
      case NNADAPTER_CUM_SUM:
      case NNADAPTER_FILL_LIKE:
      case NNADAPTER_FLATTEN:
      case NNADAPTER_GELU:
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH:
      case NNADAPTER_LAYER_NORMALIZATION:
      case NNADAPTER_LEAKY_RELU:
      case NNADAPTER_MAX_POOL_2D:
      case NNADAPTER_PAD:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_RESHAPE:
      case NNADAPTER_RESIZE_NEAREST:
      case NNADAPTER_RESIZE_LINEAR:
      case NNADAPTER_SLICE:
      case NNADAPTER_SQUEEZE:
      case NNADAPTER_SWISH:
      case NNADAPTER_TANH:
      case NNADAPTER_TILE:
      case NNADAPTER_TRANSPOSE:
      case NNADAPTER_UNSQUEEZE: {
        ConvertOperandSymmToAsymm(input_operands[0], 128);
        ConvertOperandSymmToAsymm(output_operands[0], 128);
        PropagateAsymmZeroPoint(input_operands[0], output_operands[0]);
      } break;
      case NNADAPTER_CONCAT:
      case NNADAPTER_STACK: {
        NNADAPTER_CHECK_GE(input_count, 2);
        for (int i = 0; i < input_count - 1; i++) {
          ConvertOperandSymmToAsymm(input_operands[i], 128);
        }
        NNADAPTER_CHECK_EQ(output_count, 1);
        ConvertOperandSymmToAsymm(output_operands[0], 128);
      } break;
      case NNADAPTER_CONV_2D:
      case NNADAPTER_CONV_2D_TRANSPOSE: {
        ConvertOperandSymmToAsymm(input_operands[0], 128);
        ConvertOperandSymmToAsymm(input_operands[1], 128);
        ConvertOperandSymmToAsymm(input_operands[2], 128);
        ConvertOperandSymmToAsymm(output_operands[0], 128);
      } break;
      case NNADAPTER_SIGMOID:
      case NNADAPTER_SOFTMAX: {
        ConvertOperandSymmToAsymm(input_operands[0], 128);
        // The zeroPoint of the output of softmax and sigmoid must be 0.
        ConvertOperandSymmToAsymm(output_operands[0], 0);
      } break;
      case NNADAPTER_SPLIT: {
        ConvertOperandSymmToAsymm(input_operands[0], 128);
        NNADAPTER_CHECK_GE(output_count, 1);
        for (uint32_t i = 0; i < output_count; i++) {
          ConvertOperandSymmToAsymm(output_operands[i], 128);
        }
      } break;
      case NNADAPTER_UNSTACK: {
        ConvertOperandSymmToAsymm(input_operands[0], 128);
        NNADAPTER_CHECK_GE(output_count, 1);
        for (uint32_t i = 0; i < output_count; i++) {
          ConvertOperandSymmToAsymm(output_operands[i], 128);
        }
      } break;
      case NNADAPTER_QUANTIZE:
      case NNADAPTER_DEQUANTIZE:
        break;
      default:
        NNADAPTER_LOG(FATAL) << "Missing the processing of "
                             << OperationTypeToString(operation->type)
                             << " for the conversion from the symmetric "
                                "quantization to the asymmetric quantization.";
        break;
    }
  }
}

}  // namespace nnadapter
