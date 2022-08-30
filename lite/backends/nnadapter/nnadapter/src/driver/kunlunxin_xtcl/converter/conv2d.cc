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

#include "operation/conv2d.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

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

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  auto filter_expr = converter->ConvertOperand(filter_operand);
  auto bias_expr = converter->ConvertOperand(bias_operand);
  auto conv_attrs = xtcl::make_object<xtcl::network::Conv2DAttrs>();
  conv_attrs->strides = std::move(ConvertToXTCLArray<xtcl::xIndexExpr>(
      std::vector<int>({stride_height, stride_width})));
  conv_attrs->padding = std::move(ConvertToXTCLArray<xtcl::xIndexExpr>(
      std::vector<int>({pad_height_top,
                        pad_width_left,
                        pad_height_bottom,
                        pad_width_right})));
  conv_attrs->dilation = std::move(ConvertToXTCLArray<xtcl::xIndexExpr>(
      std::vector<int>({dilation_height, dilation_width})));
  conv_attrs->groups = group;
  conv_attrs->channels = output_channel_size;
  conv_attrs->kernel_size = std::move(ConvertToXTCLArray<xtcl::xIndexExpr>(
      std::vector<int>({filter_height, filter_width})));
  conv_attrs->data_layout = "NCHW";
  conv_attrs->kernel_layout = "OIHW";
  conv_attrs->out_layout = "";
  auto conv_expr =
      converter->builder()->CreateConv2D(input_expr, filter_expr, conv_attrs);
  auto bias_add_expr =
      converter->builder()->CreateBiasAdd(conv_expr, 1, bias_expr);
  converter->UpdateExprMap(output_operand, bias_add_expr);
  // Fuse activations
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, func)                              \
  case NNADAPTER_FUSED_##type:                                            \
    converter->UpdateExprMap(output_operand, converter->builder()->func); \
    break;
    CONVERT_UNARY_ACTIVATION(RELU, CreateRelu(bias_add_expr));
    CONVERT_UNARY_ACTIVATION(RELU6, CreateRelu6(bias_add_expr));
#undef CONVERT_UNARY_ACTIVATION
    case NNADAPTER_FUSED_NONE:
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                           << ") is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
