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

namespace nnadapter {
namespace huawei_kirin_npu {

int Program::ConvertPool2D(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 12);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Paddings
  auto padding_width_left =
      *reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  auto padding_width_right =
      *reinterpret_cast<int32_t*>(input_operands[2]->buffer);
  auto padding_height_top =
      *reinterpret_cast<int32_t*>(input_operands[3]->buffer);
  auto padding_height_bottom =
      *reinterpret_cast<int32_t*>(input_operands[4]->buffer);
  NNADAPTER_VLOG(5) << "paddings=[" << padding_width_left << ","
                    << padding_width_right << "," << padding_height_top << ","
                    << padding_height_bottom << "]";
  // Strides
  auto stride_width = *reinterpret_cast<int32_t*>(input_operands[5]->buffer);
  auto stride_height = *reinterpret_cast<int32_t*>(input_operands[6]->buffer);
  NNADAPTER_VLOG(5) << "strides=[" << stride_width << "," << stride_height
                    << "]";
  // Filter
  auto filter_width = *reinterpret_cast<int32_t*>(input_operands[7]->buffer);
  auto filter_height = *reinterpret_cast<int32_t*>(input_operands[8]->buffer);
  NNADAPTER_VLOG(5) << "filter=[" << filter_width << "," << filter_height
                    << "]";
  bool global_pooling = filter_width == input_operand->type.dimensions[3] &&
                        filter_height == input_operand->type.dimensions[2];
  NNADAPTER_VLOG(5) << "global_pooling=" << global_pooling;
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[9]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Ceil mode
  bool ceil_mode = *reinterpret_cast<int8_t*>(input_operands[10]->buffer);
  NNADAPTER_VLOG(5) << "ceil_mode=" << ceil_mode;
  // Count include pad
  bool count_include_pad =
      *reinterpret_cast<int8_t*>(input_operands[11]->buffer);
  NNADAPTER_VLOG(5) << "count_include_pad=" << count_include_pad;
  NNADAPTER_CHECK_EQ(count_include_pad, false)
      << "rknpu_ddk doesn't suppport count_include_pad=true";
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to HiAI operators
  auto input_operator = ConvertOperand(input_operand);
  auto pool2d_operator = AddOperator<ge::op::Pooling>(output_operand);
  pool2d_operator->set_input_x(*input_operator);
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    pool2d_operator->set_attr_mode(1);
    NNADAPTER_CHECK(!count_include_pad) << "Only count_include_pad=false is "
                                           "supported for the pooling type "
                                           "'avg' in HiAI";
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    pool2d_operator->set_attr_mode(0);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  pool2d_operator->set_attr_pad_mode(0);  // NOTSET
  pool2d_operator->set_attr_global_pooling(global_pooling);
  pool2d_operator->set_attr_window(
      ge::AttrValue::LIST_INT({filter_height, filter_width}));
  pool2d_operator->set_attr_pad(ge::AttrValue::LIST_INT({padding_height_bottom,
                                                         padding_height_top,
                                                         padding_width_right,
                                                         padding_width_left}));
  pool2d_operator->set_attr_stride(
      ge::AttrValue::LIST_INT({stride_height, stride_width}));
  if (ceil_mode) {
    pool2d_operator->set_attr_ceil_mode(1);
    pool2d_operator->set_attr_data_mode(0);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
