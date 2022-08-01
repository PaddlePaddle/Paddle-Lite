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

#include "operation/elementwise.h"
#include "driver/imagination_nna/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace imagination_nna {

int ConvertElementwise(Converter* converter, core::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to imgdnn tensors and operators
  auto input0_tensor = converter->GetMappedTensor(input0_operand);
  if (!input0_tensor) {
    input0_tensor = converter->ConvertOperand(input0_operand);
  }
  auto input1_tensor = converter->GetMappedTensor(input1_operand);
  if (!input1_tensor) {
    input1_tensor = converter->ConvertOperand(input1_operand);
  }
  imgdnn_quant_param output_quant_param;
  output_quant_param.scale = output_operand->type.asymm_per_layer_params.scale;
  output_quant_param.zero_point =
      output_operand->type.asymm_per_layer_params.zero_point;
  imgdnn_tensor output_tensor;
  switch (operation->type) {
#define CONVERT_ELEMENTWISE(type, class_name)                   \
  case NNADAPTER_##type: {                                      \
    output_tensor = ADD_OPERATOR(CreateElementwiseOpsLayer,     \
                                 input0_tensor,                 \
                                 input1_tensor,                 \
                                 IMGDNN_OPERATION_##class_name, \
                                 output_quant_param);           \
    converter->UpdateTensorMap(output_operand, output_tensor);  \
  } break;
    CONVERT_ELEMENTWISE(ADD, ADD);
    CONVERT_ELEMENTWISE(SUB, SUB);
    CONVERT_ELEMENTWISE(MUL, MUL);
    CONVERT_ELEMENTWISE(DIV, DIV);
    CONVERT_ELEMENTWISE(MAX, MAX);
    CONVERT_ELEMENTWISE(MIN, MIN);
#undef CONVERT_ELEMENTWISE
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported element-wise operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }

  // Fuse RELU
  if (fuse_code == NNADAPTER_FUSED_NONE) {
  } else if (fuse_code == NNADAPTER_FUSED_RELU) {
    output_tensor = ADD_OPERATOR(
        CreateReLULayer, output_tensor, true, 0.0, false, 0.0, false);
  } else if (fuse_code == NNADAPTER_FUSED_RELU1) {
    output_tensor = ADD_OPERATOR(
        CreateReLULayer, output_tensor, true, 0.0, true, 1.0, false);
  } else if (fuse_code == NNADAPTER_FUSED_RELU6) {
    output_tensor = ADD_OPERATOR(
        CreateReLULayer, output_tensor, true, 0.0, true, 6.0, false);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                         << ") is found.";
  }
  converter->UpdateTensorMap(output_operand, output_tensor);

  return NNADAPTER_NO_ERROR;
}

}  // namespace imagination_nna
}  // namespace nnadapter
