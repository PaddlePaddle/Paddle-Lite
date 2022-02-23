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

#include "operation/pool2d.h"
#include "driver/imagination_nna/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace imagination_nna {

int ConvertPool2D(Converter* converter, core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to imgdnn tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  imgdnn_pooling_type pool_type;
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    pool_type = IMGDNN_POOLING_AVERAGE;
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    pool_type = IMGDNN_POOLING_MAX;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  unsigned int ksizes[2] = {static_cast<unsigned int>(kernel_height),
                            static_cast<unsigned int>(kernel_width)};
  unsigned int strides[2] = {static_cast<unsigned int>(stride_height),
                             static_cast<unsigned int>(stride_width)};
  // Top and left
  unsigned int pad_to_begin[2] = {static_cast<unsigned int>(pad_height_top),
                                  static_cast<unsigned int>(pad_width_left)};
  // Bottom and right
  unsigned int pad_to_end[2] = {static_cast<unsigned int>(pad_height_bottom),
                                static_cast<unsigned int>(pad_width_right)};
  NNADAPTER_CHECK(
      IsUInt8AsymmPerLayerQuantType(output_operand->type.precision));
  imgdnn_quant_param output_quant_param;
  output_quant_param.scale = output_operand->type.asymm_per_layer_params.scale;
  output_quant_param.zero_point =
      output_operand->type.asymm_per_layer_params.zero_point;
  auto output_tensor = ADD_OPERATOR(CreatePoolingLayer,
                                    input_tensor,
                                    output_quant_param,
                                    ksizes,
                                    strides,
                                    pad_to_begin,
                                    pad_to_end,
                                    flag,
                                    pool_type);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace imagination_nna
}  // namespace nnadapter
