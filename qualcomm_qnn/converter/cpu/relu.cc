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

#include "driver/qualcomm_qnn/converter/cpu/relu.h"
#include "driver/qualcomm_qnn/converter/cpu/utility.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace cpu {

Qnn_ErrorHandle_t Relu::Finalize() {
  QNN_CHECK_EQ(InputSize(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);
  QNN_CHECK_EQ(OutputSize(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  auto input = GetInput(0);
  auto output = GetOutput(0);
  QNN_CHECK_EQ(input->dataType,
               output->dataType,
               QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  // Supporting upto 4D input tensor
  const int input_rank = TensorRank(input);
  const int output_rank = TensorRank(output);
  QNN_CHECK_VALUE(input_rank == output_rank,
                  QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);
  QNN_CHECK_VALUE(input_rank >= 1 && input_rank <= 4,
                  QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  SetIsFinalize(true);

  return QNN_SUCCESS;
}

void Relu::ReluKernel(const float* in, const int input_size, float* out) {
  for (int32_t s = 0; s < input_size; ++s) {
    const float f = *in;
    if (f < 0) {
      *out = 0;
    } else {
      *out = f;
    }
    in++;
    out++;
  }
}

Qnn_ErrorHandle_t Relu::Execute() {
  QNN_CHECK_VALUE(GetIsFinalize(), QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED);
  auto input = GetInput(0);
  auto output = GetOutput(0);

  ReluKernel(reinterpret_cast<const float*>(input->data),
             TensorSize(input),
             reinterpret_cast<float*>(output->data));

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t Relu::SetOpNode(QnnCpuOpPackage_Node_t* node) {
  // Add input
  for (uint32_t i = 0; i < node->numOfInputs; i++) {
    AddInput(node->inputs[i]);
  }

  // Add output
  for (uint32_t i = 0; i < node->numOfOutputs; i++) {
    AddOutput(node->outputs[i]);
  }

  return QNN_SUCCESS;
}

}  // namespace cpu
}  // namespace qualcomm_qnn
}  // namespace nnadapter
