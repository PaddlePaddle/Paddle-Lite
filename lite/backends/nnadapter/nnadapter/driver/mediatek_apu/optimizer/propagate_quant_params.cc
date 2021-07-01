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

#include "driver/mediatek_apu/optimizer/propagate_quant_params.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace mediatek_apu {

static void MakeQuantParamsSameAs(hal::Operand* reference_operand,
                                  hal::Operand* target_operand) {
  auto& reference_type = reference_operand->type;
  auto& target_type = target_operand->type;
  auto reference_precision = reference_type.precision;
  auto target_precision = target_type.precision;
  if (IsAsymmPerLayerQuantization(reference_precision) &&
      IsAsymmPerLayerQuantization(target_precision)) {
    if (fabs(target_type.asymm_per_layer_params.scale -
             reference_type.asymm_per_layer_params.scale) > 1e-7) {
      NNADAPTER_CHECK(target_type.lifetime != NNADAPTER_MODEL_INPUT &&
                      target_type.lifetime != NNADAPTER_MODEL_OUTPUT);
    }
    target_type.asymm_per_layer_params = reference_type.asymm_per_layer_params;
  } else if (IsSymmPerLayerQuantization(reference_precision) &&
             IsSymmPerLayerQuantization(target_precision)) {
    if (fabs(target_type.symm_per_layer_params.scale -
             reference_type.symm_per_layer_params.scale) > 1e-7) {
      NNADAPTER_CHECK(target_type.lifetime != NNADAPTER_MODEL_INPUT &&
                      target_type.lifetime != NNADAPTER_MODEL_OUTPUT);
    }
    target_type.symm_per_layer_params = reference_type.symm_per_layer_params;
  } else {
    NNADAPTER_LOG(FATAL) << "Unhandled case: reference_precision="
                         << OperandPrecisionCodeToString(
                                reference_type.precision)
                         << ", target_precision="
                         << OperandPrecisionCodeToString(target_precision);
  }
}

void PropagateQuantParams(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  auto operation_count = operations.size();
  // Traverse the operations in reverse order, and propagate the quantization
  // parameters of output operands to the input operands, make sure they share
  // the same quantization parameters
  for (int i = operation_count - 1; i >= 0; i--) {
    auto operation = operations[i];
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    auto& input_operands = operation->input_operands;
    auto& output_operands = operation->output_operands;
    switch (operation->type) {
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_RESHAPE:
      case NNADAPTER_TRANSPOSE:
        MakeQuantParamsSameAs(output_operands[0], input_operands[0]);
        break;
      case NNADAPTER_ADD:
      case NNADAPTER_CONCAT:
      case NNADAPTER_CONV_2D:
      case NNADAPTER_DIV:
      case NNADAPTER_FULLY_CONNECTED:
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH:
      case NNADAPTER_MUL:
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
