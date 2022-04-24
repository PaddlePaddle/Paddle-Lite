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

#include "operation/reduce.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertReduce(Converter* converter, core::Operation* operation) {
  REDUCE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to trr tensors and node.
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  // Bitmask for axis.
  uint32_t axis_mask = 0;
  NNADAPTER_CHECK_LE(axes_size, 32);
  for (int i = 0; i < axes_size; i++) {
    NNADAPTER_CHECK_GT(axes_data[i], 0);
    axis_mask |= 1 << (axes_data[i] - 1);
  }
  switch (operation->type) {
#define CONVERT_REDUCE(type, reduce_type)                      \
  case NNADAPTER_##type: {                                     \
    auto reduce_layer = converter->network()->addReduce(       \
        *input_tensor, reduce_type, axis_mask, keep_dim);      \
    auto output_tensor = reduce_layer->getOutput(0);           \
    converter->UpdateTensorMap(output_operand, output_tensor); \
  } break;
    CONVERT_REDUCE(REDUCE_SUM, nvinfer1::ReduceOperation::kSUM);
#undef CONVERT_REDUCE
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported reduce operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
