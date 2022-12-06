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

#include "operation/reduce.h"
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertReduce(Converter* converter, core::Operation* operation) {
  REDUCE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to tim-vx tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  std::vector<int32_t> axis;
  for (int i = 0; i < axes_size; i++) {
    axis.push_back(ConvertToTimVXAxis(
        axes_data[i], output_operand->type.dimensions.count) /* WHCN */);
  }
  switch (operation->type) {
#define CONVERT_REDUCE(type, class_name)                               \
  case NNADAPTER_##type: {                                             \
    auto reduce_op =                                                   \
        converter->graph()->CreateOperation<tim::vx::ops::class_name>( \
            axis, keep_dim);                                           \
    reduce_op->BindInputs({input_tensor});                             \
    reduce_op->BindOutputs({output_tensor});                           \
  } break;
    CONVERT_REDUCE(REDUCE_MEAN, ReduceMean);
    CONVERT_REDUCE(REDUCE_SUM, ReduceSum);
    CONVERT_REDUCE(REDUCE_MAX, ReduceMax);
#undef CONVERT_REDUCE
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported reduce operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
