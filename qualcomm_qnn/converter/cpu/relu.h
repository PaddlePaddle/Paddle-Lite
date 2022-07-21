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

#pragma once

#include <cmath>

#include "CPU/QnnCpuOpPackage.h"
#include "driver/qualcomm_qnn/converter/cpu/op_base.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace cpu {

class Relu : public OpBase {
 public:
  Relu() {}
  explicit Relu(QnnCpuOpPackage_Node_t* node)
      : OpBase(node->name, node->typeName) {}

  ~Relu() {}

  Qnn_ErrorHandle_t Finalize() override;

  void ReluKernel(const float* in, const int input_size, float* out);

  Qnn_ErrorHandle_t Execute() override;

  Qnn_ErrorHandle_t SetOpNode(QnnCpuOpPackage_Node_t* node) override;
};

}  // namespace cpu
}  // namespace qualcomm_qnn
}  // namespace nnadapter
