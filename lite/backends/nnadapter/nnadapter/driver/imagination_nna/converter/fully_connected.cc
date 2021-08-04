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

#include "driver/imagination_nna/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace imagination_nna {

int Program::ConvertFullyConnected(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Weight
  auto weight_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "weight: " << OperandToString(weight_operand);
  NNADAPTER_CHECK_EQ(weight_operand->type.dimension_count, 2);
  auto num_units = weight_operand->type.dimensions[0];
  auto input_size = weight_operand->type.dimensions[1];
  // Bias
  auto bias_operand = input_operands[2];
  NNADAPTER_VLOG(5) << "bias: " << OperandToString(bias_operand);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimension_count, 1);
  NNADAPTER_CHECK_EQ(num_units, bias_operand->type.dimensions[0]);
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[3]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  NNADAPTER_CHECK_EQ(fuse_code, NNADAPTER_FUSED_NONE)
      << "imgdnn doesn't support fuse_code=" << fuse_code
      << " in fully connected layer";
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to imgdnn tensors and operators
  auto input_tensor = GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = ConvertOperand(input_operand);
  }
  // Transpose weight tensor from (m,k) to (k,m)
  std::vector<uint8_t> transpose_weight_data(num_units * input_size);
  std::vector<int32_t> transpose_weight_dimensions(
      weight_operand->type.dimension_count);
  TransposeData(reinterpret_cast<uint8_t*>(weight_operand->buffer),
                transpose_weight_data.data(),
                {1, 0},
                weight_operand->type.dimensions,
                transpose_weight_dimensions.data());
  NNADAPTER_CHECK(
      IsUInt8AsymmPerLayerQuantization(weight_operand->type.precision));
  auto transpose_weight_tensor = AddQuant8ConstantTensor(
      transpose_weight_data.data(),
      transpose_weight_dimensions.data(),
      transpose_weight_dimensions.size(),
      weight_operand->type.asymm_per_layer_params.scale,
      weight_operand->type.asymm_per_layer_params.zero_point);
  // Expand bias tensor from (c) to (1, c)
  auto bias_tensor =
      ConvertOperand(bias_operand, {1, bias_operand->type.dimensions[0]});
  NNADAPTER_CHECK(
      IsUInt8AsymmPerLayerQuantization(output_operand->type.precision));
  imgdnn_quant_param output_quant_param;
  output_quant_param.scale = output_operand->type.asymm_per_layer_params.scale;
  output_quant_param.zero_point =
      output_operand->type.asymm_per_layer_params.zero_point;
  auto output_tensor = imgdnn_mgr_.CreateFullyConnectedLayer(
      input_tensor, transpose_weight_tensor, bias_tensor, output_quant_param);
  UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace imagination_nna
}  // namespace nnadapter
