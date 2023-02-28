// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "operation/stack.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertStack(Converter* converter, core::Operation* operation) {
  STACK_OPERATION_EXTRACT_INPUTS_OUTPUTS

  ElementType data_type;
  TensorVector op_datas_reshape;
  auto axis_const = converter->AddConstantTensor<int64_t>(axis);
  for (int i = 0; i < input_count - 1; i++) {
    auto input_operand = input_operands[i];
    auto input_tensor = converter->GetMappedTensor(input_operand);
    if (!input_tensor) {
      input_tensor = converter->ConvertOperand(input_operand);
    }
    if (i == 0) {
      data_type = input_tensor->get_element_type();
    }
    NNADAPTER_CHECK(data_type == input_tensor->get_element_type());
    op_datas_reshape.push_back(
        std::make_shared<default_opset::Unsqueeze>(*input_tensor, *axis_const));
  }
  auto concat_op =
      std::make_shared<default_opset::Concat>(op_datas_reshape, axis);
  MAP_OUTPUT(output_operand, concat_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
