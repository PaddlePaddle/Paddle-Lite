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

#include <memory>
#include "driver/qualcomm_qnn/converter/converter.h"
#include "operation/unary_activations.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace qualcomm_qnn {

int ConvertSwish(Converter* converter, core::Operation* operation) {
  UNARY_ACTIVATIONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to qnn tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  NNADAPTER_VLOG(5) << "output operand type:" << output_operand->type.lifetime;
  Qnn_Tensor_t sigmoid_output_tensor;
  std::vector<uint32_t> dimensions(output_operand->type.dimensions.data,
                                   output_operand->type.dimensions.data +
                                       output_operand->type.dimensions.count);
  if (IsAsymmetricQuantType(output_operand->type.precision)) {
    float scale = output_operand->type.asymm_per_layer_params.scale /
                  input_operand->type.asymm_per_layer_params.scale;
    int32_t zero_point = 128;
    sigmoid_output_tensor = converter->AddVariableTensor(
        output_operand->type.precision, dimensions, &scale, &zero_point);
  } else {
    sigmoid_output_tensor = converter->AddVariableTensor(
        output_operand->type.precision, dimensions);
  }
  converter->AddNode(QNN_OP_SIGMOID, {input_tensor}, {sigmoid_output_tensor});
  auto output_tensor = converter->GetMappedTensor(output_operand);
  converter->AddNode(QNN_OP_ELEMENT_WISE_MULTIPLY,
                     {input_tensor, sigmoid_output_tensor},
                     {output_tensor});
  return NNADAPTER_NO_ERROR;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
