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

#include "driver/huawei_kirin_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int Program::ConvertPool2D(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 8);
  auto operation_type = operation->type;
  if (operation_type == NNADAPTER_AVERAGE_POOL_2D) {
    NNADAPTER_CHECK_EQ(output_count, 1);
  } else if (operation_type == NNADAPTER_MAX_POOL_2D) {
    NNADAPTER_CHECK_EQ(output_count, 2);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }

  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Auto pad: AscendNPU does not support auto_pad.
  // Pads: Pads are transed according to auto_pad, so pads are used.
  uint32_t pads_size =
      input_operands[2]->length / static_cast<uint32_t>(sizeof(int32_t));
  NNADAPTER_CHECK_EQ(pads_size, 4U);
  auto pads_buffer = reinterpret_cast<int32_t*>(input_operands[2]->buffer);
  auto pad_height_top = pads_buffer[0];
  auto pad_height_bottom = pads_buffer[1];
  auto pad_width_left = pads_buffer[2];
  auto pad_width_right = pads_buffer[3];
  NNADAPTER_VLOG(5) << "paddings = [" << pad_height_top << ", "
                    << pad_height_bottom << ", " << pad_width_left << ", "
                    << pad_width_right << "]";
  // Kernel shape
  uint32_t kernel_shape_size =
      input_operands[3]->length / static_cast<uint32_t>(sizeof(int32_t));
  NNADAPTER_CHECK_EQ(kernel_shape_size, 2U);
  auto kernel_buffer = reinterpret_cast<int32_t*>(input_operands[3]->buffer);
  auto kernel_height = kernel_buffer[0];
  auto kernel_width = kernel_buffer[1];
  NNADAPTER_VLOG(5) << "kernel = [" << kernel_height << ", " << kernel_width
                    << "]";
  bool global_pooling = kernel_height == input_operand->type.dimensions[2] &&
                        kernel_width == input_operand->type.dimensions[3];
  NNADAPTER_VLOG(5) << "global_pooling = " << global_pooling;
  // Strides
  uint32_t strides_size =
      input_operands[4]->length / static_cast<uint32_t>(sizeof(int32_t));
  NNADAPTER_CHECK_EQ(strides_size, 2U);
  auto strides_buffer = reinterpret_cast<int32_t*>(input_operands[4]->buffer);
  auto stride_height = strides_buffer[0];
  auto stride_width = strides_buffer[1];
  NNADAPTER_VLOG(5) << "strides = [" << stride_height << ", " << stride_width
                    << "]";
  // Ceil mode
  bool ceil_mode = *reinterpret_cast<bool*>(input_operands[5]->buffer);
  NNADAPTER_VLOG(5) << "ceil_mode = " << ceil_mode;
  // Count include pad(for avg_pool) or return indices(for max_pool)
  bool flag = *reinterpret_cast<bool*>(input_operands[6]->buffer);
  NNADAPTER_VLOG(5) << "count_include_pad/return_indices = " << flag;
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    NNADAPTER_CHECK(!flag) << "Only support count_include_pad = false.";
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    NNADAPTER_CHECK(!flag) << "Only support return_indices = false.";
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[7]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code = " << fuse_code;

  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  auto pool2d_name = GetOperatorName(output_operand);
  auto pool2d_op = std::make_shared<hiai::op::PoolingD>(pool2d_name);
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    pool2d_op->set_attr_mode(1);
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    pool2d_op->set_attr_mode(0);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  pool2d_op->set_attr_global_pooling(global_pooling);
  pool2d_op->set_attr_window(
      ge::AttrValue::LIST_INT({kernel_height, kernel_width}));
  pool2d_op->set_attr_pad(ge::AttrValue::LIST_INT(
      {pad_height_top, pad_height_bottom, pad_width_left, pad_width_right}));
  pool2d_op->set_attr_stride(
      ge::AttrValue::LIST_INT({stride_height, stride_width}));
  if (ceil_mode) {
    pool2d_op->set_attr_ceil_mode(1);
    pool2d_op->set_attr_data_mode(0);
  }
  SET_INPUT(pool2d_op, x, input_operator);
  MAP_OUTPUT(pool2d_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
