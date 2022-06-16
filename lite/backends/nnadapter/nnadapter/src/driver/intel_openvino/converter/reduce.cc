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
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertReduce(Converter* converter, core::Operation* operation) {
  REDUCE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto axes_tensor = converter->AddConstantTensor(
      std::vector<int32_t>(axes_data, axes_data + axes_size));
  switch (operation->type) {
#define CONVERT_REDUCE(type, reduce_type)                                     \
  case NNADAPTER_##type: {                                                    \
    auto reduce_op =                                                          \
        std::make_shared<reduce_type>(*input_tensor, *axes_tensor, keep_dim); \
    MAP_OUTPUT(output_operand, reduce_op, 0);                                 \
  } break;
    CONVERT_REDUCE(REDUCE_MEAN, default_opset::ReduceMean);
    CONVERT_REDUCE(REDUCE_MAX, default_opset::ReduceMax);
#undef CONVERT_REDUCE
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported reduce operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
