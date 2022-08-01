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

#include "operation/pool2d.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertPool2D(Converter* converter, core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto pool2d_op = converter->AddOperator<hiai::op::PoolingD>(output_operand);
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
