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
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertAdaptivePool2D(Converter* converter, core::Operation* operation) {
  ADAPTIVE_POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  if (operation->type == NNADAPTER_ADAPTIVE_MAX_POOL_2D) {
    NNADAPTER_LOG(FATAL) << "Not support max pool2d.";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  } else {
    auto pool2d_node =
        converter->network()->AddIAdaptiveAvgPool2DNode(input_tensor);
    NNADAPTER_CHECK(pool2d_node) << "Failed to add adaptive_avg_pool2d node.";
    pool2d_node->SetOutputSize(static_cast<int64_t>(output_height),
                               static_cast<int64_t>(output_width));
    magicmind::Layout input_layout =
        ConvertToMagicMindDataLayout(input_operand->type.layout);
    pool2d_node->SetLayout(input_layout, input_layout);
    auto output_tensor = pool2d_node->GetOutput(0);
    converter->UpdateTensorMap(output_operand, output_tensor);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
