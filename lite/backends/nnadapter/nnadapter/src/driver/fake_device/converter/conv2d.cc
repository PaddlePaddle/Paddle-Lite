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

#include "operation/conv2d.h"
#include "converter/converter.h"
#include "converter/validator.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace fake_device {

bool ValidateConv2D(Validator* validator, const core::Operation* operation) {
  return true;
}

int ConvertConv2D(Converter* converter, core::Operation* operation) {
  CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS
  if (auto_pad != NNADAPTER_AUTO_PAD_NONE) {
    operation::UpdateConv2DPadAndDilation(
        input_operand->type.dimensions.data[2],
        filter_height,
        auto_pad,
        &pad_height_top,
        &pad_height_bottom,
        stride_height,
        &dilation_height);
    operation::UpdateConv2DPadAndDilation(
        input_operand->type.dimensions.data[3],
        filter_width,
        auto_pad,
        &pad_width_left,
        &pad_width_right,
        stride_width,
        &dilation_width);
  }

  // Convert to fake device tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto filter_tensor = converter->ConvertOperand(filter_operand);
  auto bias_tensor = converter->ConvertOperand(bias_operand);
  if (!bias_tensor) {
    bias_tensor = converter->ConvertOperand(bias_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  fake_ddk::Conv2DAttr attr;
  attr.pad_type = fake_ddk::PadType::AUTO;
  attr.pad[0] = pad_height_top;
  attr.pad[1] = pad_height_bottom;
  attr.pad[2] = pad_width_left;
  attr.pad[3] = pad_width_right;
  attr.stride[0] = stride_height;
  attr.stride[1] = stride_width;
  attr.dilation[0] = dilation_height;
  attr.dilation[1] = dilation_width;
  attr.group = group;
  attr.fuse_type = ConvertFuseCodeToFakeDeviceFuseType(fuse_code);
  converter->AddOperator(fake_ddk::OperatorType::FAKE_DDK_CONV2D,
                         {input_tensor, filter_tensor, bias_tensor},
                         {output_tensor},
                         &attr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace fake_device
}  // namespace nnadapter
