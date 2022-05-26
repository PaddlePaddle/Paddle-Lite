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

#include "driver/nvidia_tensorrt/kernel/host/naive_softmax.h"
#include <cmath>
#include "driver/nvidia_tensorrt/operation/naive_softmax.h"

namespace nnadapter {
namespace nvidia_tensorrt {
namespace host {

template <typename T>
void Softmax(const T* input, T* output, const int num) {
  T x_max = input[0];
  for (int i = 1; i < num; i++) {
    x_max = input[i] > x_max ? input[i] : x_max;
  }
  for (int i = 0; i < num; i++) {
    output[i] = exp(input[i] - x_max);
  }
  T sum = output[0];
  for (int i = 1; i < num; i++) {
    sum += output[i];
  }
  for (int i = 0; i < num; i++) {
    output[i] /= sum;
  }
}

int NaiveSoftmaxKernel::Run(
    core::Operation* operation,
    std::map<core::Operand*, std::shared_ptr<Tensor>>* operand_map,
    cudaStream_t stream) {
  NNADAPTER_CHECK_EQ(operation->type, NNADAPTER_NAIVE_SOFTMAX);
  NAIVE_SOFTMAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = operand_map->at(input_operand);
  auto output_tensor = operand_map->at(output_operand);
  output_tensor->Resize(input_tensor->Dims());
  int num = input_tensor->Length();
  const float* input =
      reinterpret_cast<const float*>(input_tensor->Data(false));
  float* output = reinterpret_cast<float*>(output_tensor->Data(false));
  Softmax<float>(input, output, num);
  return NNADAPTER_NO_ERROR;
}

}  // namespace host
}  // namespace nvidia_tensorrt
}  // namespace nnadapter
