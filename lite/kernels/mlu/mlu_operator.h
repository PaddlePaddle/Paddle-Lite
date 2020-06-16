// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "lite/backends/mlu/mlu_utils.h"
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {

struct MLUOperator {
  cnmlBaseOp_t cnml_op = nullptr;
  // compile time tensor
  std::vector<cnmlTensor_t> input_tensors{};
  std::vector<cnmlTensor_t> output_tensors{};
  ~MLUOperator() {
    if (cnml_op != nullptr) {
      CNML_CALL(cnmlDestroyBaseOp(&cnml_op));
      cnml_op = nullptr;
    }
    if (!input_tensors.empty()) {
      std::for_each(input_tensors.begin(),
                    input_tensors.end(),
                    [](cnmlTensor_t t) { CNML_CALL(cnmlDestroyTensor(&t)); });
      input_tensors.clear();
    }
    if (!output_tensors.empty()) {
      std::for_each(output_tensors.begin(),
                    output_tensors.end(),
                    [](cnmlTensor_t t) { CNML_CALL(cnmlDestroyTensor(&t)); });
      output_tensors.clear();
    }
  }
};

}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
