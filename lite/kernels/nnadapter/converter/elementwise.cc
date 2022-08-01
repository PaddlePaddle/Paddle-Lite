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

#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertElementwise(Converter* converter, OpInfo* op, Scope* scope) {
  // X operand
  auto x_name = op->Input("X").front();
  auto x_tensor = scope->FindTensor(x_name);
  auto x_persistable = x_tensor->persistable();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input0_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  // Y operand
  auto y_name = op->Input("Y").front();
  auto y_tensor = scope->FindTensor(y_name);
  auto y_persistable = y_tensor->persistable();
  auto y_scale_name = "Y0_scale";
  std::vector<float> y_scales;
  if (op->HasInputScale(y_scale_name, true)) {
    y_scales = op->GetInputScale(y_scale_name, true);
  }
  auto input1_operand = converter->AddInputOperand(scope, y_name, {}, y_scales);

  // Check whether the two dimensions are compatiable(Numpy-style broadcasting
  // https://numpy.org/doc/stable/user/basics.broadcasting.html).
  // Unsqueeze the small rank
  uint32_t x_rank = converter->GetOperandType(input0_operand)->dimensions.count;
  uint32_t y_rank = converter->GetOperandType(input1_operand)->dimensions.count;
  uint32_t max_rank = std::max(x_rank, y_rank);
  int32_t axis = op->GetAttr<int32_t>("axis");
  if (axis < 0) {
    axis = std::abs(static_cast<int>(x_rank) - static_cast<int>(y_rank));
  }
  // Prepare unsqueeze axes
  std::vector<int32_t> axes;
  for (int32_t i = 0; i < axis; i++) {
    axes.push_back(i);
  }
  int32_t remain =
      max_rank - static_cast<int32_t>(std::min(x_rank, y_rank)) - axis;
  for (int32_t i = 0; i < remain; i++) {
    axes.push_back(max_rank - remain + i);
  }
  // If persistable, set matched dims.
  // If not persistable, add unsqueeze to match shape.
  if (x_rank > y_rank) {
    if (y_persistable) {
      std::vector<int64_t> shape = y_tensor->dims().Vectorize();
      shape.insert(shape.begin(), axis, 1);
      shape.insert(shape.end(), max_rank - shape.size(), 1);
      input1_operand = converter->AddConstantOperand(
          *y_tensor, DDim(shape), false, y_scales);
    } else {
      CHECK(!axes.empty());
      input1_operand = converter->AddUnsqueezeOperation(input1_operand, axes);
    }
  } else if (y_rank > x_rank) {
    if (x_persistable) {
      std::vector<int64_t> shape = x_tensor->dims().Vectorize();
      shape.insert(shape.begin(), axis, 1);
      shape.insert(shape.end(), max_rank - shape.size(), 1);
      input0_operand = converter->AddConstantOperand(
          *x_tensor, DDim(shape), false, x_scales);
    } else {
      CHECK(!axes.empty());
      input0_operand = converter->AddUnsqueezeOperation(input0_operand, axes);
    }
  }

  // Fuse code operand
  int32_t fuse_code_value = NNADAPTER_FUSED_NONE;
  auto act_type =
      op->HasAttr("act_type") ? op->GetAttr<std::string>("act_type") : "";
  if (act_type == "relu") {
    fuse_code_value = NNADAPTER_FUSED_RELU;
    act_type = "";
  } else if (act_type == "relu1") {
    fuse_code_value = NNADAPTER_FUSED_RELU1;
    act_type = "";
  } else if (act_type == "relu6") {
    fuse_code_value = NNADAPTER_FUSED_RELU6;
    act_type = "";
  }
  auto fuse_code_operand = converter->AddConstantOperand(fuse_code_value);

  // Output
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);

  // Elementwise operation
  NNAdapterOperationType eltwise_operation_type;
  auto op_type = op->Type();
  if (op_type == "elementwise_add" ||
      op_type == "fusion_elementwise_add_activation") {
    eltwise_operation_type = NNADAPTER_ADD;
  } else if (op_type == "elementwise_sub" ||
             op_type == "fusion_elementwise_sub_activation") {
    eltwise_operation_type = NNADAPTER_SUB;
  } else if (op_type == "elementwise_mul" ||
             op_type == "fusion_elementwise_mul_activation") {
    eltwise_operation_type = NNADAPTER_MUL;
  } else if (op_type == "elementwise_div" ||
             op_type == "fusion_elementwise_div_activation") {
    eltwise_operation_type = NNADAPTER_DIV;
  } else if (op_type == "elementwise_max" ||
             op_type == "fusion_elementwise_max_activation") {
    eltwise_operation_type = NNADAPTER_MAX;
  } else if (op_type == "elementwise_min" ||
             op_type == "fusion_elementwise_min_activation") {
    eltwise_operation_type = NNADAPTER_MIN;
  } else if (op_type == "elementwise_pow" ||
             op_type == "fusion_elementwise_pow_activation") {
    eltwise_operation_type = NNADAPTER_POW;
  } else {
    LOG(WARNING) << "Unsupported elementwise op type: " << op_type;
    return UNSUPPORTED_FEATURE;
  }
  converter->AddOperation(eltwise_operation_type,
                          {input0_operand, input1_operand, fuse_code_operand},
                          {output_operand});

  // Unpack the fused activations
  if (!act_type.empty()) {
    auto fused_act_output_operand =
        converter->AddOutputOperand(out_name, out_scales);
    if (act_type == "leaky_relu") {
      auto alpha = op->GetAttr<float>("leaky_relu_alpha");
      auto alpha_operand = converter->AddConstantOperand(alpha);
      converter->AddOperation(NNADAPTER_LEAKY_RELU,
                              {output_operand, alpha_operand},
                              {fused_act_output_operand});
    } else {
      // Unpack the fused unary activations
      auto unary_act_operation_type =
          ConvertUnaryActTypeToNNOperationType(act_type);
      CHECK(unary_act_operation_type != NNADAPTER_UNKNOWN)
          << "Failed to unpack the fused activation type: " << act_type;
      converter->AddOperation(unary_act_operation_type,
                              {output_operand},
                              {fused_act_output_operand});
    }
  }
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
