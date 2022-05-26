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

#include "operation/pool2d.h"
#include "core/types.h"
#include "operation/math/pool2d.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT void UpdatePool2DPadAndDilation(
    int32_t input_size,
    int32_t kernel_height_or_width,
    NNAdapterAutoPadCode auto_pad,
    int32_t* pad_top_or_left,
    int32_t* pad_bottom_or_right,
    int32_t stride_height_or_width) {
  NNADAPTER_CHECK_NE(input_size, NNADAPTER_UNKNOWN);
  if (auto_pad == NNADAPTER_AUTO_PAD_SAME) {
    auto output_size =
        (input_size + stride_height_or_width - 1) / stride_height_or_width;
    auto pad_size = (std::max)((output_size - 1) * stride_height_or_width +
                                   kernel_height_or_width - input_size,
                               0);
    *pad_top_or_left = pad_size / 2;
    *pad_bottom_or_right = pad_size - *pad_top_or_left;
  } else if (auto_pad == NNADAPTER_AUTO_PAD_VALID) {
    *pad_top_or_left = 0;
    *pad_bottom_or_right = 0;
  }
}

NNADAPTER_EXPORT int32_t CalPoolOutputSize(int32_t input_size,
                                           int32_t kernel_height_or_width,
                                           NNAdapterAutoPadCode auto_pad,
                                           int32_t pad_top_or_left,
                                           int32_t pad_bottom_or_right,
                                           int32_t stride_height_or_width,
                                           bool ceil_mode) {
  if (input_size == NNADAPTER_UNKNOWN) {
    return NNADAPTER_UNKNOWN;
  }
  UpdatePool2DPadAndDilation(input_size,
                             kernel_height_or_width,
                             auto_pad,
                             &pad_top_or_left,
                             &pad_bottom_or_right,
                             stride_height_or_width);
  int32_t output_size;
  if (!ceil_mode) {
    output_size = (input_size - kernel_height_or_width + pad_top_or_left +
                   pad_bottom_or_right) /
                      stride_height_or_width +
                  1;
  } else {
    output_size = (input_size - kernel_height_or_width + pad_top_or_left +
                   pad_bottom_or_right + stride_height_or_width - 1) /
                      stride_height_or_width +
                  1;
  }
  return output_size;
}

NNADAPTER_EXPORT bool ValidatePool2D(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int PreparePool2D(core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);

  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions) {
    if (global_pooling) {
      output_dimensions[2] = 1;
      output_dimensions[3] = 1;
    } else {
      output_dimensions[2] = CalPoolOutputSize(input_dimensions[2],
                                               kernel_height,
                                               auto_pad,
                                               pad_height_top,
                                               pad_height_bottom,
                                               stride_height,
                                               ceil_mode);
      output_dimensions[3] = CalPoolOutputSize(input_dimensions[3],
                                               kernel_width,
                                               auto_pad,
                                               pad_width_left,
                                               pad_width_right,
                                               stride_width,
                                               ceil_mode);
    }
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

NNADAPTER_EXPORT int ExecutePool2D(core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  int status = -1;
  auto& input_type = input_operand->type;
  auto input_shape = std::vector<int32_t>(
      input_type.dimensions.data,
      input_type.dimensions.data + input_type.dimensions.count);
  const auto input_buffer = input_operand->buffer;
  NNADAPTER_CHECK(input_buffer);
  auto& output_type = output_operand->type;
  auto output_buffer = AllocateOperand(output_operand);
  NNADAPTER_CHECK_EQ(input_type.precision, output_type.precision);
  if (input_type.precision == NNADAPTER_FLOAT32) {
    const auto input_data = reinterpret_cast<const float*>(input_buffer);
    auto output_data = reinterpret_cast<float*>(output_buffer);
    if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
      status =
          math::average_pool2d<float>(input_data,
                                      input_shape,
                                      kernel_height,
                                      kernel_width,
                                      pad_height_top,
                                      pad_height_bottom,
                                      pad_width_left,
                                      pad_width_right,
                                      stride_height,
                                      stride_width,
                                      ceil_mode,
                                      flag,
                                      static_cast<math::FuseCode>(fuse_code),
                                      output_data);
    } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
      status =
          math::max_pool2d<float>(input_data,
                                  input_shape,
                                  kernel_height,
                                  kernel_width,
                                  pad_height_top,
                                  pad_height_bottom,
                                  pad_width_left,
                                  pad_width_right,
                                  stride_height,
                                  stride_width,
                                  ceil_mode,
                                  flag,
                                  static_cast<math::DataTypeCode>(indices_type),
                                  static_cast<math::FuseCode>(fuse_code),
                                  output_data);
    } else {
      NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
    }
  } else if (input_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER) {
    const auto input_data = reinterpret_cast<const int8_t*>(input_buffer);
    auto output_data = reinterpret_cast<int8_t*>(output_buffer);
    if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
      status = math::average_pool2d(input_data,
                                    input_shape,
                                    input_type.symm_per_layer_params.scale,
                                    kernel_height,
                                    kernel_width,
                                    pad_height_top,
                                    pad_height_bottom,
                                    pad_width_left,
                                    pad_width_right,
                                    stride_height,
                                    stride_width,
                                    ceil_mode,
                                    flag,
                                    static_cast<math::FuseCode>(fuse_code),
                                    output_data,
                                    output_type.symm_per_layer_params.scale);
    } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
      status = math::max_pool2d(input_data,
                                input_shape,
                                input_type.symm_per_layer_params.scale,
                                kernel_height,
                                kernel_width,
                                pad_height_top,
                                pad_height_bottom,
                                pad_width_left,
                                pad_width_right,
                                stride_height,
                                stride_width,
                                ceil_mode,
                                flag,
                                static_cast<math::DataTypeCode>(indices_type),
                                static_cast<math::FuseCode>(fuse_code),
                                output_data,
                                output_type.symm_per_layer_params.scale);
    } else {
      NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
    }
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
