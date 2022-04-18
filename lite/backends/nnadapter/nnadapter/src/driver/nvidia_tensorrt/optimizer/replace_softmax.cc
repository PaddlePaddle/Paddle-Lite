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

#include "driver/nvidia_tensorrt/optimizer/replace_softmax.h"
#include <cmath>
#include <vector>
#include "driver/nvidia_tensorrt/operation/type.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {

void ReplaceSoftmaxWithNaiveSoftmax(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    if (operation->type == NNADAPTER_SOFTMAX) {
      operation->type = NNADAPTER_NAIVE_SOFTMAX;
    }
  }
}

void ReplaceSoftmaxWithSpecialSoftmax(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    if (operation->type == NNADAPTER_SOFTMAX) {
      operation->type = NNADAPTER_SPECIAL_SOFTMAX;
    }
  }
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
