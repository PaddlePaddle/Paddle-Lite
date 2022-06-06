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

#include "operation/elementwise.h"
#include "driver/qualcomm_qnn/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace qualcomm_qnn {

int ConvertElementwise(Converter* converter, core::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(fuse_code, NNADAPTER_FUSED_NONE);
  // Convert to qnn tensors and node
  auto input0_tensor = converter->GetMappedTensor(input0_operand);
  auto input1_tensor = converter->GetMappedTensor(input1_operand);
  auto output_tensor = converter->GetMappedTensor(output_operand);
  std::map<NNAdapterOperationType, const char*> op_type_map{
      {NNADAPTER_ADD, QNN_OP_ELEMENT_WISE_ADD},
  };
  converter->AddNode(op_type_map.at(operation->type),
                     {input0_tensor, input1_tensor},
                     {output_tensor});
  return NNADAPTER_NO_ERROR;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
