// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "core/operation/fully_connected.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertFullyConnected(Converter* converter, hal::Operation* operation) {
  FULLY_CONNECTED_OPERATION_EXTRACT_INPUTS_OUTPUTS
  auto batch_size =
      ProductionOfDimensions(input_operand->type.dimensions.data,
                             input_operand->type.dimensions.count) /
      input_size;
  NNADAPTER_VLOG(5) << "batch_size: " << batch_size;

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  magicmind::ITensor* final_input_tensor = nullptr;
  // Reshape the input operator to 2-D tensor {batch_size, input_size} if the
  // dimensions_count not equal 2
  if (input_operand->type.dimensions.count != 2) {
    auto shape_tensor = converter->AddInt32ConstantTensor(
        std::vector<int32_t>({static_cast<int32_t>(batch_size), input_size})
            .data(),
        {2});
    auto reshape_node =
        converter->network()->AddIReshapeNode(input_tensor, shape_tensor);
    if (reshape_node == nullptr) {
      NNADAPTER_VLOG(5) << "Failed to reshape input operator to 2-D tensor for "
                           "fully_connected node.";
      return NNADAPTER_DEVICE_INTERNAL_ERROR;
    }
    final_input_tensor = reshape_node->GetOutput(0);
  } else {
    final_input_tensor = input_tensor;
  }
  auto weight_tensor = converter->ConvertOperand(weight_operand);
  auto bias_tensor = converter->ConvertOperand(bias_operand);
  auto matmul_node = converter->network()->AddIMatMulNode(
      final_input_tensor, weight_tensor, bias_tensor);
  NNADAPTER_CHECK(matmul_node) << "Failed to add fully_connected node.";
  matmul_node->SetTransA(false);
  matmul_node->SetTransB(true);
  auto output_tensor = matmul_node->GetOutput(0);
  // fuse activations ?
  switch (fuse_code) {
#define CONVERT_ACTIVATION(type, mm_type)                                 \
  case NNADAPTER_FUSED_##type: {                                          \
    auto activation_node =                                                \
        converter->network()->AddIActivationNode(output_tensor, mm_type); \
    auto fuse_out_tensor = activation_node->GetOutput(0);                 \
    converter->UpdateTensorMap(output_operand, fuse_out_tensor);          \
    break;                                                                \
  }
    CONVERT_ACTIVATION(RELU, magicmind::IActivation::RELU);
    CONVERT_ACTIVATION(RELU6, magicmind::IActivation::RELU6);
#undef CONVERT_ACTIVATION
    case NNADAPTER_FUSED_NONE:
      converter->UpdateTensorMap(output_operand, output_tensor);
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                           << ") is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
