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

#include "driver/mediatek_apu/optimizer/restrict_input_output_quant_params.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace mediatek_apu {

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

static bool RestrictOutputScaleGreaterThanInput0ScaleMultiplyInput1Scale(
    core::Model* model,
    core::Operation* operation,
    core::Operand* input0_operand,
    core::Operand* input1_operand,
    core::Operand* output_operand) {
  if (!IsAsymmPerLayerQuantType(input0_operand->type.precision) ||
      !IsAsymmPerLayerQuantType(input1_operand->type.precision) ||
      !IsAsymmPerLayerQuantType(output_operand->type.precision)) {
    return false;
  }
  auto input0_scale = input0_operand->type.asymm_per_layer_params.scale;
  auto input1_scale = input1_operand->type.asymm_per_layer_params.scale;
  float output_scale = output_operand->type.asymm_per_layer_params.scale;
  // Insert a requantize operation and a operand with output_scale =
  // input0_scale * input1_scale + 1e-6f
  double input0_scale_x_input1_scale =
      static_cast<double>(input0_scale) * static_cast<double>(input1_scale);
  if (output_scale > input0_scale_x_input1_scale) {
    return false;
  }
  return RestrictInputOutputScale(
      model,
      operation,
      output_operand,
      input0_scale_x_input1_scale + 1e-6f,
      output_operand->type.asymm_per_layer_params.zero_point,
      true);
}

static bool RestrictBiasScaleEqualInputScaleMultiplyFilterScale(
    core::Model* model,
    core::Operation* operation,
    core::Operand* input_operand,
    core::Operand* weight_operand,
    core::Operand* bias_operand) {
  if (IsAsymmPerLayerQuantType(input_operand->type.precision) &&
      IsAsymmPerLayerQuantType(weight_operand->type.precision) &&
      IsInt32SymmPerLayerQuantType(bias_operand->type.precision)) {
    auto input_scale = input_operand->type.asymm_per_layer_params.scale;
    auto weight_scale = weight_operand->type.asymm_per_layer_params.scale;
    auto& bias_scale = bias_operand->type.symm_per_layer_params.scale;
    // Requantize the bias data if the following condition is not satisfied
    double input_scale_x_weight_scale =
        static_cast<double>(input_scale) * static_cast<double>(weight_scale);
    if (fabs(input_scale_x_weight_scale - bias_scale) <= 1e-6f) {
      return false;
    }
    NNADAPTER_CHECK_EQ(bias_operand->type.dimensions.count, 1);
    auto channel_size = bias_operand->type.dimensions.data[0];
    auto bias_data = reinterpret_cast<int32_t*>(bias_operand->buffer);
    std::vector<float> dequantized_bias(channel_size);
    NNADAPTER_CHECK(DequantizeData<int32_t>(bias_data,
                                            &channel_size,
                                            1,
                                            &bias_scale,
                                            NULL,
                                            -1,
                                            -2147483647,
                                            2147483647,
                                            dequantized_bias.data()));
    NNADAPTER_VLOG(5) << "Requantize bias operand "
                      << OperandIdToString(bias_operand) << ": scale "
                      << bias_scale << " -> " << input_scale_x_weight_scale;
    bias_scale = input_scale_x_weight_scale;
    NNADAPTER_CHECK(QuantizeData<int32_t>(dequantized_bias.data(),
                                          &channel_size,
                                          1,
                                          &bias_scale,
                                          NULL,
                                          -1,
                                          -2147483647,
                                          2147483647,
                                          bias_data));
    return true;
  } else if (IsAsymmPerLayerQuantType(input_operand->type.precision) &&
             IsSymmPerChannelQuantType(weight_operand->type.precision) &&
             IsInt32SymmPerChannelQuantType(bias_operand->type.precision)) {
    auto input_scale = input_operand->type.asymm_per_layer_params.scale;
    auto weight_scales = weight_operand->type.symm_per_channel_params.scales;
    auto bias_scales = bias_operand->type.symm_per_channel_params.scales;
    NNADAPTER_CHECK_EQ(bias_operand->type.symm_per_channel_params.channel_dim,
                       0);
    NNADAPTER_CHECK_EQ(bias_operand->type.dimensions.count, 1);
    auto channel_size = bias_operand->type.dimensions.data[0];
    NNADAPTER_CHECK_EQ(bias_operand->type.symm_per_channel_params.scale_count,
                       channel_size);
    // Requantize the bias data if the following condition is not satisfied
    bool same = true;
    std::vector<double> input_scale_x_weight_scales(channel_size);
    for (int32_t i = 0; i < channel_size; i++) {
      input_scale_x_weight_scales[i] = static_cast<double>(input_scale) *
                                       static_cast<double>(weight_scales[i]);
      if (fabs(input_scale_x_weight_scales[i] - bias_scales[i]) > 1e-6f) {
        same = false;
      }
    }
    if (same) return false;
    auto bias_data = reinterpret_cast<int32_t*>(bias_operand->buffer);
    std::vector<float> dequantized_bias(channel_size);
    NNADAPTER_CHECK(DequantizeData<int32_t>(
        bias_data,
        &channel_size,
        1,
        bias_scales,
        NULL,
        bias_operand->type.symm_per_channel_params.channel_dim,
        -2147483647,
        2147483647,
        dequantized_bias.data()));
    for (int32_t i = 0; i < channel_size; i++) {
      NNADAPTER_VLOG(5) << "Requantize bias operand "
                        << OperandIdToString(bias_operand) << ": scale[" << i
                        << "] " << bias_scales[i] << " -> "
                        << input_scale_x_weight_scales[i];
      bias_scales[i] = input_scale_x_weight_scales[i];
    }
    NNADAPTER_CHECK(QuantizeData<int32_t>(
        dequantized_bias.data(),
        &channel_size,
        1,
        bias_scales,
        NULL,
        bias_operand->type.symm_per_channel_params.channel_dim,
        -2147483647,
        2147483647,
        bias_data));
    return true;
  }
  return false;
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
      case NNADAPTER_CONV_2D:
      case NNADAPTER_FULLY_CONNECTED: {
        RestrictBiasScaleEqualInputScaleMultiplyFilterScale(model,
                                                            operation,
                                                            input_operands[0],
                                                            input_operands[1],
                                                            input_operands[2]);
        // The following condition must be satisfied: output_scale > input_scale
        // * weight_scale
        RestrictOutputScaleGreaterThanInput0ScaleMultiplyInput1Scale(
            model,
            operation,
            input_operands[0],
            input_operands[1],
            output_operands[0]);
      } break;
      case NNADAPTER_CONCAT:
        for (uint32_t i = 0; i < input_count - 1; i++) {
          RestrictInputOutputScale(
              model, operation, input_operands[i], output_operands[0], false);
        }
        break;
      case NNADAPTER_FLATTEN:
      case NNADAPTER_RELU:
      case NNADAPTER_RESHAPE:
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
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_BATCH_NORMALIZATION:
      case NNADAPTER_CONV_2D_TRANSPOSE:
      case NNADAPTER_DIV:
      case NNADAPTER_MAT_MUL:
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH:
      case NNADAPTER_MAX_POOL_2D:
      case NNADAPTER_MUL:
      case NNADAPTER_RELU6:
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

}  // namespace mediatek_apu
}  // namespace nnadapter
