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

#include "operation/adaptive_pool2d.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertAdaptivePool2D(Converter* converter, core::Operation* operation) {
  ADAPTIVE_POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  if (operation->type == NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D) {
    auto shape_tensor =
        converter->AddConstantTensor<int64_t>({output_height, output_width});
    auto adaptive_avg_pool_op =
        std::make_shared<default_opset::AdaptiveAvgPool>(*input_tensor,
                                                         *shape_tensor);
    MAP_OUTPUT(output_operand, adaptive_avg_pool_op, 0);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported adaptive pool2d operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
