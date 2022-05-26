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

#include <cmath>
#include "driver/nvidia_tensorrt/kernel/cuda/special_softmax.h"
#include "driver/nvidia_tensorrt/operation/special_softmax.h"

namespace nnadapter {
namespace nvidia_tensorrt {
namespace cuda {

template <typename T>
__global__ void Softmax(const T* input, T* output, int num) {
  const int idx = threadIdx.x;
  T x_max = input[0];
  for (int i = 1; i < num; i++) {
    x_max = input[i] > x_max ? input[i] : x_max;
  }
  output[idx] = exp(input[idx] - x_max);
  __syncthreads();
  T sum = output[0];
  for (int i = 1; i < num; i++) {
    sum += output[i];
  }
  output[idx] /= sum;
}

int SpecialSoftmaxKernel::Run(
    core::Operation* operation,
    std::map<core::Operand*, std::shared_ptr<Tensor>>* operand_map,
    cudaStream_t stream) {
  NNADAPTER_CHECK_EQ(operation->type, NNADAPTER_SPECIAL_SOFTMAX);
  SPECIAL_SOFTMAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = operand_map->at(operation->input_operands[0]);
  auto output_tensor = operand_map->at(operation->output_operands[0]);
  output_tensor->Resize(input_tensor->Dims());
  int num = input_tensor->Length();
  const float* input = reinterpret_cast<const float*>(input_tensor->Data());
  float* output = reinterpret_cast<float*>(output_tensor->Data());
  Softmax<float><<<1, num, 0, stream>>>(input, output, num);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cuda
}  // namespace nvidia_tensorrt
}  // namespace nnadapter
