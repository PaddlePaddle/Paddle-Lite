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
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertPool2D(Converter* converter, core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    auto pool2d_op = converter->AddOperator<ge::op::AvgPoolV2>(output_operand);
    pool2d_op->set_attr_ksize(
        ge::Operator::OpListInt({1, 1, kernel_height, kernel_width}));
    pool2d_op->set_attr_strides(
        ge::Operator::OpListInt({1, 1, stride_height, stride_width}));
    auto GetPoolingPaddingMode = [&](int32_t auto_pad) {
      switch (auto_pad) {
        case NNADAPTER_AUTO_PAD_VALID:
          return "VALID";
        case NNADAPTER_AUTO_PAD_SAME:
          return "SAME";
        case NNADAPTER_AUTO_PAD_NONE:
        default:
          return "CALCULATED";
      }
    };
    pool2d_op->set_attr_padding_mode(
        ge::Operator::OpString(GetPoolingPaddingMode(auto_pad)));
    pool2d_op->set_attr_pads(ge::Operator::OpListInt(
        {pad_height_top, pad_height_bottom, pad_width_left, pad_width_right}));
    pool2d_op->set_attr_global_pooling(global_pooling);
    pool2d_op->set_attr_ceil_mode(ceil_mode);
    if (flag) {
      pool2d_op->set_attr_exclusive(0);
    }
    SET_INPUT(pool2d_op, x, input_operator);
    MAP_OUTPUT(pool2d_op, y, output_operand);
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    auto pool2d_op = converter->AddOperator<ge::op::Pooling>(output_operand);
    pool2d_op->set_attr_mode(0);
    pool2d_op->set_attr_global_pooling(global_pooling);
    pool2d_op->set_attr_window(
        ge::Operator::OpListInt({kernel_height, kernel_width}));
    pool2d_op->set_attr_pad(ge::Operator::OpListInt(
        {pad_height_top, pad_height_bottom, pad_width_left, pad_width_right}));
    pool2d_op->set_attr_stride(
        ge::Operator::OpListInt({stride_height, stride_width}));
    // "0" (ceil mode) or "1" (floor mode). Defaults to "0"
    if (!ceil_mode) {
      pool2d_op->set_attr_ceil_mode(1);
    }
    SET_INPUT(pool2d_op, x, input_operator);
    MAP_OUTPUT(pool2d_op, y, output_operand);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
