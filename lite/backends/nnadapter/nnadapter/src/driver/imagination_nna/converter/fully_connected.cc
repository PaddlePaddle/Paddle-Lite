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

#include "operation/fully_connected.h"
#include "driver/imagination_nna/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace imagination_nna {

int ConvertFullyConnected(Converter* converter, core::Operation* operation) {
  FULLY_CONNECTED_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to imgdnn tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  // Transpose weight tensor from (m,k) to (k,m)
  std::vector<uint8_t> transpose_weight_data(num_units * input_size);
  std::vector<int32_t> transpose_weight_dimensions(
      weight_operand->type.dimensions.count);
  TransposeData(reinterpret_cast<uint8_t*>(weight_operand->buffer),
                transpose_weight_data.data(),
                {1, 0},
                weight_operand->type.dimensions.data,
                transpose_weight_dimensions.data());
  NNADAPTER_CHECK(
      IsUInt8AsymmPerLayerQuantType(weight_operand->type.precision));
  auto transpose_weight_tensor = converter->AddQuant8ConstantTensor(
      transpose_weight_data.data(),
      transpose_weight_dimensions.data(),
      transpose_weight_dimensions.size(),
      weight_operand->type.asymm_per_layer_params.scale,
      weight_operand->type.asymm_per_layer_params.zero_point);
  auto bias_tensor = converter->ConvertOperand(
      bias_operand, {1, bias_operand->type.dimensions.data[0]});
  NNADAPTER_CHECK(
      IsUInt8AsymmPerLayerQuantType(output_operand->type.precision));
  imgdnn_quant_param output_quant_param;
  output_quant_param.scale = output_operand->type.asymm_per_layer_params.scale;
  output_quant_param.zero_point =
      output_operand->type.asymm_per_layer_params.zero_point;
  auto output_tensor = ADD_OPERATOR(CreateFullyConnectedLayer,
                                    input_tensor,
                                    transpose_weight_tensor,
                                    bias_tensor,
                                    output_quant_param);
  // fuse RELU ?
  if (fuse_code == NNADAPTER_FUSED_NONE) {
  } else if (fuse_code == NNADAPTER_FUSED_RELU) {
    output_tensor = ADD_OPERATOR(
        CreateReLULayer, output_tensor, true, 0.0, false, 0.0, false);
  } else if (fuse_code == NNADAPTER_FUSED_RELU1) {
    output_tensor = ADD_OPERATOR(
        CreateReLULayer, output_tensor, true, 0.0, true, 1.0, false);
  } else if (fuse_code == NNADAPTER_FUSED_RELU6) {
    output_tensor = ADD_OPERATOR(
        CreateReLULayer, output_tensor, true, 0.0, true, 6.0, false);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                         << ") is found.";
  }
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace imagination_nna
}  // namespace nnadapter
