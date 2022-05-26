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

#include "operation/conv2d.h"
#include "driver/qualcomm_qnn/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace qualcomm_qnn {

int ConvertConv2D(Converter* converter, core::Operation* operation) {
  CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to qnn tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  auto weight_tensor = converter->GetMappedTensor(filter_operand);
  auto bias_tensor = converter->GetMappedTensor(bias_operand);
  auto output_tensor = converter->GetMappedTensor(output_operand);

  auto group_param = converter->GetParam(QNN_OP_CONV_2D_PARAM_GROUP,
                                         static_cast<uint32_t>(group));
  auto pad_param = converter->GetParam(
      QNN_OP_CONV_2D_PARAM_PAD_AMOUNT,
      std::vector<uint32_t>{static_cast<uint32_t>(pad_height_top),
                            static_cast<uint32_t>(pad_height_bottom),
                            static_cast<uint32_t>(pad_width_left),
                            static_cast<uint32_t>(pad_width_right)},
      {2, 2});
  auto stride_param = converter->GetParam(
      QNN_OP_CONV_2D_PARAM_STRIDE,
      std::vector<uint32_t>{static_cast<uint32_t>(stride_height),
                            static_cast<uint32_t>(stride_width)});
  auto dilation_param = converter->GetParam(
      QNN_OP_CONV_2D_PARAM_DILATION,
      std::vector<uint32_t>{static_cast<uint32_t>(dilation_height),
                            static_cast<uint32_t>(dilation_width)});

  converter->AddNode(QNN_OP_CONV_2D,
                     {input_tensor, weight_tensor, bias_tensor},
                     {output_tensor},
                     {group_param, pad_param, stride_param, dilation_param});
  return NNADAPTER_NO_ERROR;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
