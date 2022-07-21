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

#include "operation/resize_linear.h"
#include "driver/qualcomm_qnn/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace qualcomm_qnn {

int ConvertResizeLinear(Converter* converter, core::Operation* operation) {
  RESIZE_LINEAR_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to qnn tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  auto output_tensor = converter->GetMappedTensor(output_operand);
  auto align_corners_param = converter->GetParam(
      QNN_OP_RESIZE_BILINEAR_PARAM_ALIGN_CORNERS, align_corners);
  converter->AddNode(QNN_OP_RESIZE_BILINEAR,
                     {input_tensor},
                     {output_tensor},
                     {align_corners_param});
  return NNADAPTER_NO_ERROR;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
