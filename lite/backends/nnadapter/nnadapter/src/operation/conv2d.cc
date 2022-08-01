// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/conv2d.h"
#include "core/types.h"
#include "operation/math/conv2d.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT void UpdateConv2DPadAndDilation(
    int32_t input_size,
    int32_t filter_height_or_width,
    NNAdapterAutoPadCode auto_pad,
    int32_t* pad_top_or_left,
    int32_t* pad_bottom_or_right,
    int32_t stride_height_or_width,
    int32_t* dilation_height_or_width) {
  if (auto_pad == NNADAPTER_AUTO_PAD_SAME) {
    NNADAPTER_CHECK_NE(input_size, NNADAPTER_UNKNOWN);
    auto output_size =
        (input_size + stride_height_or_width - 1) / stride_height_or_width;
    auto pad_size = (output_size - 1) * stride_height_or_width +
                    filter_height_or_width - input_size;
    pad_size = pad_size < 0 ? 0 : pad_size;
    *pad_top_or_left = pad_size / 2;
    *pad_bottom_or_right = pad_size - *pad_top_or_left;
    *dilation_height_or_width = 1;
  } else if (auto_pad == NNADAPTER_AUTO_PAD_VALID) {
    *pad_top_or_left = 0;
    *pad_bottom_or_right = 0;
  }
}

NNADAPTER_EXPORT int32_t
CalcConv2DOutputSize(int32_t input_size,
                     int32_t filter_height_or_width,
                     NNAdapterAutoPadCode auto_pad,
                     int32_t pad_top_or_left,
                     int32_t pad_bottom_or_right,
                     int32_t stride_height_or_width,
                     int32_t dilation_height_or_width) {
  if (input_size == NNADAPTER_UNKNOWN) {
    return NNADAPTER_UNKNOWN;
  }
  UpdateConv2DPadAndDilation(input_size,
                             filter_height_or_width,
                             auto_pad,
                             &pad_top_or_left,
                             &pad_bottom_or_right,
                             stride_height_or_width,
                             &dilation_height_or_width);
  auto dkernel = dilation_height_or_width * (filter_height_or_width - 1) + 1;
  return (input_size + (pad_top_or_left + pad_bottom_or_right) - dkernel) /
             stride_height_or_width +
         1;
}

NNADAPTER_EXPORT void GetConv2DFilterDims(const core::Operand* filter_operand,
                                          int32_t* c_out,
                                          int32_t* c_in,
                                          int32_t* h,
                                          int32_t* w,
                                          bool is_depthwise_mode) {
  auto filter_layout = filter_operand->type.layout;
  switch (filter_layout) {
    case NNADAPTER_NHWC:
      if (is_depthwise_mode) {
        *c_out = filter_operand->type.dimensions.data[3];
        *c_in = filter_operand->type.dimensions.data[0];
      } else {
        *c_out = filter_operand->type.dimensions.data[0];
        *c_in = filter_operand->type.dimensions.data[3];
      }
      *h = filter_operand->type.dimensions.data[1];
      *w = filter_operand->type.dimensions.data[2];
      break;
    case NNADAPTER_HWCN:
      *c_out = filter_operand->type.dimensions.data[3];
      *c_in = filter_operand->type.dimensions.data[2];
      *h = filter_operand->type.dimensions.data[0];
      *w = filter_operand->type.dimensions.data[1];
      break;
    case NNADAPTER_NCHW:
    default:
      *c_out = filter_operand->type.dimensions.data[0];
      *c_in = filter_operand->type.dimensions.data[1];
      *h = filter_operand->type.dimensions.data[2];
      *w = filter_operand->type.dimensions.data[3];
  }
}

NNADAPTER_EXPORT bool ValidateConv2D(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int PrepareConv2D(core::Operation* operation) {
  CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions) {
    output_dimensions[0] = input_dimensions[0];
    output_dimensions[1] = output_channel_size;
    output_dimensions[2] = CalcConv2DOutputSize(input_dimensions[2],
                                                filter_height,
                                                auto_pad,
                                                pad_height_top,
                                                pad_height_bottom,
                                                stride_height,
                                                dilation_height);
    output_dimensions[3] = CalcConv2DOutputSize(input_dimensions[3],
                                                filter_width,
                                                auto_pad,
                                                pad_width_left,
                                                pad_width_right,
                                                stride_width,
                                                dilation_width);
  };
  infer_output_shape(input_operand->type.dimensions.data,
                     output_operand->type.dimensions.data);
  for (uint32_t i = 0; i < input_operand->type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_operand->type.dimensions.dynamic_data[i],
                       output_operand->type.dimensions.dynamic_data[i]);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteConv2D(core::Operation* operation) {
  CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  int status = -1;
  auto& input_type = input_operand->type;
  auto input_shape = std::vector<int32_t>(
      input_type.dimensions.data,
      input_type.dimensions.data + input_type.dimensions.count);
  const auto input_buffer = input_operand->buffer;
  NNADAPTER_CHECK(input_buffer);
  auto& filter_type = filter_operand->type;
  auto filter_shape = std::vector<int32_t>(
      filter_type.dimensions.data,
      filter_type.dimensions.data + filter_type.dimensions.count);
  const auto filter_buffer = filter_operand->buffer;
  NNADAPTER_CHECK(filter_buffer);
  const auto bias_buffer = bias_operand->buffer;
  NNADAPTER_CHECK(bias_buffer);
  auto& output_type = output_operand->type;
  auto output_buffer = AllocateOperand(output_operand);
  NNADAPTER_CHECK_EQ(input_type.precision, output_type.precision);
  if (input_type.precision == NNADAPTER_FLOAT32) {
    const auto input_data = reinterpret_cast<const float*>(input_buffer);
    const auto filter_data = reinterpret_cast<const float*>(filter_buffer);
    const auto bias_data = reinterpret_cast<const float*>(bias_buffer);
    auto output_data = reinterpret_cast<float*>(output_buffer);
    status = math::conv2d<float>(input_data,
                                 input_shape,
                                 filter_data,
                                 filter_shape,
                                 bias_data,
                                 pad_height_top,
                                 pad_height_bottom,
                                 pad_width_left,
                                 pad_width_right,
                                 stride_height,
                                 stride_width,
                                 dilation_height,
                                 dilation_width,
                                 group,
                                 static_cast<math::FuseCode>(fuse_code),
                                 output_data);
  } else if (input_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER &&
             (filter_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER ||
              filter_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL)) {
    const auto input_data = reinterpret_cast<const int8_t*>(input_buffer);
    const auto filter_data = reinterpret_cast<const int8_t*>(filter_buffer);
    const auto bias_data = reinterpret_cast<const int32_t*>(bias_buffer);
    auto output_data = reinterpret_cast<int8_t*>(output_buffer);
    auto filter_scales = std::make_pair(
        std::vector<float>({filter_type.symm_per_layer_params.scale}), -1);
    if (filter_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL) {
      filter_scales.first = std::vector<float>(
          filter_type.symm_per_channel_params.scales,
          filter_type.symm_per_channel_params.scales +
              filter_type.symm_per_channel_params.scale_count);
      filter_scales.second = filter_type.symm_per_channel_params.channel_dim;
    }
    status = math::conv2d(input_data,
                          input_shape,
                          input_type.symm_per_layer_params.scale,
                          filter_data,
                          filter_shape,
                          filter_scales,
                          bias_data,
                          pad_height_top,
                          pad_height_bottom,
                          pad_width_left,
                          pad_width_right,
                          stride_height,
                          stride_width,
                          dilation_height,
                          dilation_width,
                          group,
                          static_cast<math::FuseCode>(fuse_code),
                          output_data,
                          output_type.symm_per_layer_params.scale);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported precision code("
                         << OperandPrecisionCodeToString(input_type.precision)
                         << ") for " << OperationTypeToString(operation->type)
                         << " is found!";
  }
  NNADAPTER_CHECK_EQ(status, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
