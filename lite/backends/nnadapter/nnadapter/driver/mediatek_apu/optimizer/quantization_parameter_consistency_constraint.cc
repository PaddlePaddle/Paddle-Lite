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

#include "driver/mediatek_apu/optimizer/quantization_parameter_consistency_constraint.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace mediatek_apu {

static void PropagateQuantizationParameters(hal::Operand* reference_operand,
                                            hal::Operand* target_operand) {
  auto& reference_type = reference_operand->type;
  auto& target_type = target_operand->type;
  auto reference_precision = reference_type.precision;
  auto target_precision = target_type.precision;
  if (IsAsymmPerLayerQuantization(reference_precision) &&
      IsAsymmPerLayerQuantization(target_precision)) {
    if (target_type.asymm_per_layer_params.scale !=
        reference_type.asymm_per_layer_params.scale) {
      NNADAPTER_CHECK(target_type.lifetime != NNADAPTER_MODEL_INPUT &&
                      target_type.lifetime != NNADAPTER_MODEL_OUTPUT);
    }
    target_type.asymm_per_layer_params = reference_type.asymm_per_layer_params;
  } else if (IsSymmPerLayerQuantization(reference_precision) &&
             IsSymmPerLayerQuantization(target_precision)) {
    if (target_type.symm_per_layer_params.scale !=
        reference_type.symm_per_layer_params.scale) {
      NNADAPTER_CHECK(target_type.lifetime != NNADAPTER_MODEL_INPUT &&
                      target_type.lifetime != NNADAPTER_MODEL_OUTPUT);
    }
    target_type.symm_per_layer_params = reference_type.symm_per_layer_params;
  } else {
    NNADAPTER_LOG(FATAL) << "Unhandled case: reference_precision="
                         << OperandPrecisionCodeToString(
                                reference_type.precision)
                         << ", target_precision="
                         << OperandPrecisionCodeToString(target_precision);
  }
}

static void UpdateBiasScaleWithInputScaleXWeightScale(
    hal::Operand* input_operand,
    hal::Operand* weight_operand,
    hal::Operand* bias_operand) {
  auto& input_type = input_operand->type;
  auto& weight_type = weight_operand->type;
  auto& bias_type = bias_operand->type;
  bool is_symm_per_layer =
      IsInt8SymmPerLayerQuantization(input_type.precision) &&
      IsInt8SymmPerLayerQuantization(weight_type.precision) &&
      IsInt32SymmPerLayerQuantization(bias_type.precision);
  bool is_symm_per_channel =
      IsInt8SymmPerLayerQuantization(input_type.precision) &&
      IsInt8SymmPerChannelQuantization(weight_type.precision) &&
      IsInt32SymmPerChannelQuantization(bias_type.precision);
  bool is_asymm_per_layer =
      IsUInt8AsymmPerLayerQuantization(input_type.precision) &&
      IsUInt8AsymmPerLayerQuantization(weight_type.precision) &&
      IsInt32SymmPerLayerQuantization(bias_type.precision);
  bool is_asymm_per_channel =
      IsUInt8AsymmPerLayerQuantization(input_type.precision) &&
      IsInt8SymmPerChannelQuantization(weight_type.precision) &&
      IsInt32SymmPerChannelQuantization(bias_type.precision);
  NNADAPTER_CHECK(is_symm_per_layer || is_symm_per_channel ||
                  is_asymm_per_layer || is_asymm_per_channel);
  if (is_symm_per_layer || is_asymm_per_layer) {
    // Per-layer
    float input_scale = is_symm_per_layer
                            ? input_type.symm_per_layer_params.scale
                            : input_type.asymm_per_layer_params.scale;
    float weight_scale = is_symm_per_layer
                             ? weight_type.symm_per_layer_params.scale
                             : weight_type.asymm_per_layer_params.scale;
    auto& old_bias_scale = bias_type.symm_per_layer_params.scale;
    float new_bias_scale = input_scale * weight_scale;
    bool update_bias_scale = std::fabs(new_bias_scale - old_bias_scale) > 1e-7f;
    if (update_bias_scale) {
      NNADAPTER_CHECK_EQ(bias_type.dimension_count, 1);
      auto channel_size = bias_type.dimensions[0];
      auto quant_bias_data = reinterpret_cast<int32_t*>(bias_operand->buffer);
      std::vector<float> float_bias_data(channel_size);
      DequantizeData<int32_t>(quant_bias_data,
                              channel_size,
                              &old_bias_scale,
                              1,
                              &float_bias_data[0]);
      QuantizeData<int32_t>(&float_bias_data[0],
                            channel_size,
                            &new_bias_scale,
                            1,
                            quant_bias_data);
      old_bias_scale = new_bias_scale;
    }
  } else if (is_symm_per_channel || is_asymm_per_channel) {
    // Per-channel
    float input_scale = is_symm_per_channel
                            ? input_type.symm_per_layer_params.scale
                            : input_type.asymm_per_layer_params.scale;
    auto weight_scale = weight_type.symm_per_channel_params.scales;
    auto old_bias_scale = bias_type.symm_per_channel_params.scales;
    auto channel_size = weight_type.symm_per_channel_params.scale_count;
    NNADAPTER_CHECK_EQ(channel_size,
                       bias_type.symm_per_channel_params.scale_count);
    std::vector<float> new_bias_scale(channel_size);
    bool update_bias_scale = false;
    for (size_t i = 0; i < channel_size; i++) {
      new_bias_scale[i] = input_scale * weight_scale[i];
      update_bias_scale |=
          std::fabs(new_bias_scale[i] - old_bias_scale[i]) > 1e-7f;
    }
    if (update_bias_scale) {
      NNADAPTER_CHECK_EQ(bias_type.dimension_count, 1);
      NNADAPTER_CHECK_EQ(bias_type.dimensions[0], channel_size);
      auto quant_bias_data = reinterpret_cast<int32_t*>(bias_operand->buffer);
      std::vector<float> float_bias_data(channel_size);
      DequantizeData<int32_t>(quant_bias_data,
                              channel_size,
                              old_bias_scale,
                              channel_size,
                              &float_bias_data[0]);
      QuantizeData<int32_t>(&float_bias_data[0],
                            channel_size,
                            &new_bias_scale[0],
                            channel_size,
                            quant_bias_data);
      memcpy(old_bias_scale, &new_bias_scale[0], channel_size * sizeof(float));
    }
  } else {
    NNADAPTER_LOG(FATAL) << "Unhandled case: input_precision="
                         << OperandPrecisionCodeToString(input_type.precision)
                         << " weight_precision="
                         << OperandPrecisionCodeToString(weight_type.precision)
                         << " bias_precision="
                         << OperandPrecisionCodeToString(bias_type.precision);
  }
}

void ApplyQuantizationParametersConsistencyConstraint(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    auto& input_operands = operation->input_operands;
    auto& output_operands = operation->output_operands;
    switch (operation->type) {
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_RESHAPE:
      case NNADAPTER_TRANSPOSE:
        PropagateQuantizationParameters(input_operands[0], output_operands[0]);
        break;
      case NNADAPTER_CONV_2D:
      case NNADAPTER_FULLY_CONNECTED:
        UpdateBiasScaleWithInputScaleXWeightScale(
            input_operands[0], input_operands[1], input_operands[2]);
        break;
      case NNADAPTER_TANH:
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH:
      case NNADAPTER_SIGMOID:
      case NNADAPTER_SOFTMAX:
      case NNADAPTER_ADD:
      case NNADAPTER_DIV:
      case NNADAPTER_MUL:
      case NNADAPTER_SUB:
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
