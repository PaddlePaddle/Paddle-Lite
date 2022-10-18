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

#include "driver/rockchip_npu/optimizer/restrict_input_output_quant_params.h"
#include <cmath>
#include <vector>
#include "driver/rockchip_npu/utility.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace rockchip_npu {

static bool RestrictInputOutputScale(core::Model* model,
                                     core::Operation* operation,
                                     core::Operand* target_operand,
                                     float quant_scale,
                                     int32_t zero_point,
                                     bool is_output = false,
                                     double threshold = 1e-5f) {
  if (!IsAsymmPerLayerQuantType(target_operand->type.precision)) {
    return false;
  }
  auto target_quant_scale = target_operand->type.asymm_per_layer_params.scale;
  auto target_zero_point =
      target_operand->type.asymm_per_layer_params.zero_point;
  if (fabs(target_quant_scale - quant_scale) <= threshold &&
      target_zero_point == zero_point) {
    return false;
  }
  NNADAPTER_VLOG(5) << "Requantize input/output operand "
                    << OperandIdToString(target_operand) << ": scale "
                    << target_quant_scale << " -> " << quant_scale
                    << ", zero_point " << target_zero_point << " -> "
                    << zero_point;
  NNAdapterAsymmPerLayerQuantParams asymm_per_layer_params{quant_scale,
                                                           zero_point};
  if (is_output) {
    auto requantized_operand =
        InsertRequantOperation(model, target_operand, &asymm_per_layer_params);
    UpdateOperationOutputOperands(
        operation, target_operand, requantized_operand);
  } else {
    auto requantized_operand =
        AppendRequantOperation(model, target_operand, &asymm_per_layer_params);
    UpdateOperationInputOperands(
        {operation}, target_operand, requantized_operand);
  }
  return true;
}

static bool RestrictInputOutputScale(core::Model* model,
                                     core::Operation* operation,
                                     core::Operand* target_operand,
                                     core::Operand* reference_operand,
                                     bool is_output = false,
                                     double threshold = 1e-5f) {
  if (!IsAsymmPerLayerQuantType(reference_operand->type.precision)) {
    return false;
  }
  return RestrictInputOutputScale(
      model,
      operation,
      target_operand,
      reference_operand->type.asymm_per_layer_params.scale,
      reference_operand->type.asymm_per_layer_params.zero_point,
      is_output,
      threshold);
}

void RestrictInputOutputQuantParams(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  auto operation_count = operations.size();
  // Add a dummy ADD operation that does requantization to make the input and
  // output share the same quantization parameters.
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    auto& input_operands = operation->input_operands;
    auto input_count = input_operands.size();
    auto& output_operands = operation->output_operands;
    auto output_count = output_operands.size();
    switch (operation->type) {
      case NNADAPTER_CONCAT:
        for (uint32_t i = 0; i < input_count - 1; i++) {
          RestrictInputOutputScale(
              model, operation, input_operands[i], output_operands[0], false);
        }
        break;
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_CHANNEL_SHUFFLE:
      case NNADAPTER_FLATTEN:
      case NNADAPTER_MAX_POOL_2D:
      // case NNADAPTER_REDUCE_MAX:
      // case NNADAPTER_REDUCE_MIN:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_RESHAPE:
      case NNADAPTER_RESIZE_LINEAR:
      case NNADAPTER_RESIZE_NEAREST:
      case NNADAPTER_SLICE:
      case NNADAPTER_SQUEEZE:
      case NNADAPTER_TRANSPOSE:
      case NNADAPTER_UNSQUEEZE:
        RestrictInputOutputScale(
            model, operation, input_operands[0], output_operands[0], false);
        break;
      case NNADAPTER_SPLIT:
        for (uint32_t i = 0; i < output_count; i++) {
          RestrictInputOutputScale(
              model, operation, output_operands[i], input_operands[0], true);
        }
        break;
      case NNADAPTER_ADD:
      case NNADAPTER_BATCH_NORMALIZATION:
      case NNADAPTER_CONV_2D:
      case NNADAPTER_CONV_2D_TRANSPOSE:
      case NNADAPTER_DIV:
      case NNADAPTER_FULLY_CONNECTED:
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH:
      case NNADAPTER_LEAKY_RELU:
      case NNADAPTER_MAT_MUL:
      case NNADAPTER_MUL:
      case NNADAPTER_SIGMOID:
      case NNADAPTER_SOFTMAX:
      case NNADAPTER_SUB:
      case NNADAPTER_TANH:
        break;
      default:
        NNADAPTER_LOG(FATAL)
            << "Missing the processing of "
            << OperationTypeToString(operation->type)
            << " for applying the contraints to quantization parameters.";
        break;
    }
  }
}

}  // namespace rockchip_npu
}  // namespace nnadapter
