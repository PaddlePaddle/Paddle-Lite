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

#include "optimizer/convert_quantization_bias_per_channel_to_per_layer.h"
#include <algorithm>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

static void ConvertBiasOperandPerChannelToPerLayer(
    core::Operand* filter_operand, core::Operand* bias_operand) {
  auto bias_precision = bias_operand->type.precision;
  auto filter_precision = filter_operand->type.precision;
  if (IsSymmPerChannelQuantType(bias_precision) &&
      IsSymmPerChannelQuantType(filter_precision)) {
    NNADAPTER_VLOG(5) << "I am come in";
    auto bias_scale_count =
        bias_operand->type.symm_per_channel_params.scale_count;
    auto* bias_scales = bias_operand->type.symm_per_channel_params.scales;
    auto bias_data = reinterpret_cast<int32_t*>(bias_operand->buffer);
    auto bias_operand_data_length =
        GetOperandPrecisionDataLength(bias_precision);
    NNADAPTER_CHECK_EQ(bias_scale_count,
                       bias_operand->length / bias_operand_data_length);
    std::vector<float> origin_bias_value(bias_scale_count, 0);
    for (int i = 0; i < bias_scale_count; i++) {
      origin_bias_value[i] = bias_data[i] * bias_scales[i];
    }
    auto filter_scale_count =
        filter_operand->type.symm_per_channel_params.scale_count;
    auto* filter_scales_data =
        filter_operand->type.symm_per_channel_params.scales;
    auto filter_operand_data_length =
        GetOperandPrecisionDataLength(filter_precision);
    NNADAPTER_CHECK_EQ(filter_scale_count, bias_scale_count);
    // int32_t max_abs_filter_scale_index = 0;
    // float max_abs_filter_scale_value = 0;
    // for (int i = 0; i < filter_scale_count; i++) {
    //     auto filter_scale = filter_scales_data[i] < 0 ?
    //     -filter_scales_data[i] : filter_scales_data[i];
    //     if (filter_scale > max_abs_filter_scale_value) {
    //         max_abs_filter_scale_value = filter_scale;
    //         max_abs_filter_scale_index = i;
    //     }
    // }

    int32_t min_abs_filter_scale_index = 0;
    float min_abs_filter_scale_value = 99999999;
    for (int i = 0; i < filter_scale_count; i++) {
      auto filter_scale = filter_scales_data[i] < 0 ? -filter_scales_data[i]
                                                    : filter_scales_data[i];
      if (filter_scale < min_abs_filter_scale_value) {
        min_abs_filter_scale_value = filter_scale;
        min_abs_filter_scale_index = i;
      }
    }
    std::vector<float> new_bias_value(bias_scale_count, 0);
    NNADAPTER_VLOG(5) << "bias_scale_count:" << bias_scale_count;
    for (int i = 0; i < bias_scale_count; i++) {
      // new_bias_value[i] = origin_bias_value[i] /
      // bias_scales[max_abs_filter_scale_index];
      new_bias_value[i] =
          origin_bias_value[i] / bias_scales[min_abs_filter_scale_index];
      NNADAPTER_VLOG(5) << "bias_data[" << i
                        << "]:" << static_cast<int32_t>(bias_data[i]) << '\t'
                        << "bias_scales[" << i << "]:" << bias_scales[i] << '\t'
                        << "origin_bias_value[" << i
                        << "]:" << origin_bias_value[i] << '\t'
                        << "new_bias_value[" << i << "]:" << new_bias_value[i];
    }
    // auto new_bias_scale =  bias_scales[max_abs_filter_scale_index];
    auto new_bias_scale = bias_scales[min_abs_filter_scale_index];
    NNADAPTER_VLOG(5) << "new_bias_scale:" << new_bias_scale;
    // Update bias operand buffer
    bias_operand->type.precision = NNADAPTER_QUANT_INT32_SYMM_PER_LAYER;
    bias_operand->type.symm_per_layer_params.scale = new_bias_scale;
    auto is_constant_copy =
        bias_operand->type.lifetime == NNADAPTER_CONSTANT_COPY;
    auto is_constant_reference =
        bias_operand->type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
    if (is_constant_copy || is_constant_reference) {
      auto transform_buffer = static_cast<int32_t*>(bias_operand->buffer);
      if (is_constant_reference) {
        transform_buffer = static_cast<int32_t*>(malloc(bias_operand->length));
      }
      for (int i = 0; i < bias_scale_count; i++) {
        transform_buffer[i] = static_cast<int32_t>(new_bias_value[i]);
      }
      if (is_constant_reference) {
        bias_operand->buffer = transform_buffer;
        bias_operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
      }
    }
  }
}

NNADAPTER_EXPORT void ConvertQuantizationBiasPerchannelToPerlayer(
    core::Model* model) {
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
      case NNADAPTER_CONV_2D:
        ConvertBiasOperandPerChannelToPerLayer(input_operands[1],
                                               input_operands[2]);
        break;
      default:
        break;
    }
  }
}

}  // namespace nnadapter
