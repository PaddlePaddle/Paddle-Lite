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

#include "driver/mediatek_apu/optimizer/restrict_same_input_output_quant_params.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace mediatek_apu {

static void RestrictSameInputOutputScale(hal::Model* model,
                                         hal::Operation* operation,
                                         hal::Operand* target_operand,
                                         hal::Operand* reference_operand,
                                         double threshold = 1e-5f) {
  auto& target_type = target_operand->type;
  auto& reference_type = reference_operand->type;
  auto target_precision = target_type.precision;
  auto reference_precision = reference_type.precision;
  if (IsAsymmPerLayerQuantType(target_precision) &&
      IsAsymmPerLayerQuantType(reference_precision)) {
    if (fabs(target_type.asymm_per_layer_params.scale -
             reference_type.asymm_per_layer_params.scale) > threshold) {
      AddRequantOperation(model, operation, target_operand, reference_operand);
    }
  } else if (IsSymmPerLayerQuantType(target_precision) &&
             IsSymmPerLayerQuantType(reference_precision)) {
    if (fabs(target_type.symm_per_layer_params.scale -
             reference_type.symm_per_layer_params.scale) > threshold) {
      AddRequantOperation(model, operation, target_operand, reference_operand);
    }
  } else {
    NNADAPTER_LOG(FATAL) << "Unhandled case: target_precision="
                         << OperandPrecisionCodeToString(target_precision)
                         << ", reference_precision="
                         << OperandPrecisionCodeToString(reference_precision);
  }
}

void RestrictSameInputOutputQuantParams(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  auto operation_count = operations.size();
  // Add a dummy ADD operation that does requantization to make the input and
  // output share the same quantization parameters.
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    auto& input_operands = operation->input_operands;
    auto input_count = input_operands.size();
    auto& output_operands = operation->output_operands;
    auto output_count = output_operands.size();
    switch (operation->type) {
      case NNADAPTER_CONCAT:
        for (uint32_t i = 0; i < input_count - 1; i++) {
          RestrictSameInputOutputScale(
              model, operation, input_operands[i], output_operands[0]);
        }
        break;
      case NNADAPTER_FLATTEN:
      case NNADAPTER_RESHAPE:
      case NNADAPTER_TRANSPOSE:
      case NNADAPTER_UNSQUEEZE:
        RestrictSameInputOutputScale(
            model, operation, input_operands[0], output_operands[0]);
        break;
      case NNADAPTER_SPLIT:
        for (uint32_t i = 0; i < output_count; i++) {
          RestrictSameInputOutputScale(
              model, operation, output_operands[i], input_operands[0]);
        }
        break;
      case NNADAPTER_ADD:
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_CONV_2D:
      case NNADAPTER_CONV_2D_TRANSPOSE:
      case NNADAPTER_DIV:
      case NNADAPTER_FULLY_CONNECTED:
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH:
      case NNADAPTER_MAX_POOL_2D:
      case NNADAPTER_MUL:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_SIGMOID:
      case NNADAPTER_SOFTMAX:
      case NNADAPTER_SUB:
      case NNADAPTER_TANH:
        break;
      default:
        NNADAPTER_LOG(FATAL)
            << "Missing the processing of "
            << OperationTypeToString(operation->type)
            << " for applying the contraints to quantization parameters.";
        break;
    }
  }
}

}  // namespace mediatek_apu
}  // namespace nnadapter
