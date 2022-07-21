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

#include "driver/qualcomm_qnn/optimizer/restrict_input_output_quant_params.h"
#include <cmath>
#include <vector>
#include "driver/qualcomm_qnn/utility.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace qualcomm_qnn {

static void RestrictRelu(core::Operation* operation) {
  auto& out_type = operation->output_operands[0]->type;
  if (out_type.precision == NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER) {
    out_type.asymm_per_layer_params.zero_point = 0;
  }
}

void RestrictInputOutputQuantParams(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    switch (operation->type) {
      case NNADAPTER_RELU:
        RestrictRelu(operation);
        break;
      default:
        break;
    }
  }
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
