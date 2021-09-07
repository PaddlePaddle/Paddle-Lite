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

#include "core/operation/softmax.h"
#include "driver/imagination_nna/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace imagination_nna {

int Program::ConvertSoftmax(hal::Operation* operation) {
  SOFTMAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to imgdnn tensors and operators
  auto input_tensor = GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = ConvertOperand(input_operand);
  }
  imgdnn_quant_param output_quant_param;
  output_quant_param.scale = output_operand->type.asymm_per_layer_params.scale;
  output_quant_param.zero_point =
      output_operand->type.asymm_per_layer_params.zero_point;
  auto output_tensor = imgdnn_mgr_.CreateSoftmaxLayer(
      input_tensor, 1.0f, axis, output_quant_param);
  UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace imagination_nna
}  // namespace nnadapter
