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
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertPool2D(Converter* converter, core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  magicmind::INode* pool2d_node = nullptr;
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    auto avg_pool2d_node =
        converter->network()->AddIAvgPool2DNode(input_tensor);
    NNADAPTER_CHECK(avg_pool2d_node) << "Failed to add avg_pool2d node.";

    avg_pool2d_node->SetKernel(static_cast<int64_t>(kernel_height),
                               static_cast<int64_t>(kernel_width));
    avg_pool2d_node->SetPad(static_cast<int64_t>(pad_height_top),
                            static_cast<int64_t>(pad_height_bottom),
                            static_cast<int64_t>(pad_width_left),
                            static_cast<int64_t>(pad_width_right));
    avg_pool2d_node->SetStride(static_cast<int64_t>(stride_height),
                               static_cast<int64_t>(stride_width));
    avg_pool2d_node->SetCeilMode(ceil_mode);
    avg_pool2d_node->SetCountIncludePad(flag);
    magicmind::Layout input_layout =
        ConvertToMagicMindDataLayout(input_operand->type.layout);
    avg_pool2d_node->SetLayout(input_layout, input_layout);
    pool2d_node = avg_pool2d_node;
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    auto max_pool2d_node =
        converter->network()->AddIMaxPool2DNode(input_tensor, flag);
    NNADAPTER_CHECK(max_pool2d_node) << "Failed to add max_pool2d node.";

    max_pool2d_node->SetKernel(static_cast<int64_t>(kernel_height),
                               static_cast<int64_t>(kernel_width));
    max_pool2d_node->SetPad(static_cast<int64_t>(pad_height_top),
                            static_cast<int64_t>(pad_height_bottom),
                            static_cast<int64_t>(pad_width_left),
                            static_cast<int64_t>(pad_width_right));
    max_pool2d_node->SetStride(static_cast<int64_t>(stride_height),
                               static_cast<int64_t>(stride_width));
    max_pool2d_node->SetCeilMode(ceil_mode);
    magicmind::Layout input_layout =
        ConvertToMagicMindDataLayout(input_operand->type.layout);
    max_pool2d_node->SetLayout(input_layout, input_layout);
    pool2d_node = max_pool2d_node;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  auto output_tensor = pool2d_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
