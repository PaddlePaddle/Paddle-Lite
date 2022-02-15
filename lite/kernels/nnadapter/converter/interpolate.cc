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

int ConvertInterpolate(Converter* converter, OpInfo* op, Scope* scope) {
  auto op_type = op->Type();
  // Some interp ops' scale has only one data(like bilinear_interp), but some
  // interp ops' scale has more than one data(like bilinear_interp_v2)
  const std::vector<std::string> ops_has_one_scale{"bilinear_interp",
                                                   "nearest_interp"};
  const std::vector<std::string> nearest_interp_ops{"nearest_interp",
                                                    "nearest_interp_v2"};
  const std::vector<std::string> linear_interp_ops{"bilinear_interp",
                                                   "bilinear_interp_v2"};

  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  // Shape operand
  // Priority: SizeTensor(tensor) > OutSize(tensor) > out_d/out_h/out_w(attr)
  NNAdapterOperand* shape_operand = nullptr;
  if (HasInput(op, scope, "SizeTensor")) {
    // Generate shape_operand by concat
    auto in_names = op->Input("SizeTensor");
    std::vector<NNAdapterOperand*> input_operands;
    for (auto in_name : in_names) {
      auto in_operand = converter->AddInputOperand(scope, in_name);
      input_operands.push_back(in_operand);
    }
    auto axis_operand = converter->AddConstantOperand<int>(0);
    input_operands.push_back(axis_operand);
    shape_operand = converter->AddOutputOperand();
    converter->AddOperation(NNADAPTER_CONCAT, input_operands, {shape_operand});
  } else if (HasInput(op, scope, "OutSize")) {
    auto shape_name = op->Input("OutSize").front();
    shape_operand = converter->AddInputOperand(scope, shape_name);
  } else if (op->HasAttr("out_h") && op->HasAttr("out_w")) {
    int out_h = op->GetAttr<int>("out_h");
    int out_w = op->GetAttr<int>("out_w");
    if (out_h > 0 && out_w > 0) {
      shape_operand =
          converter->AddConstantOperand(std::vector<int>{out_h, out_w});
    }
  } else {
    VLOG(5) << op_type
            << " doesn't have 'SizeTensor', 'OutSize' or 'out_h/out_w'.";
  }

  // Scales operand
  NNAdapterOperand* scales_operand = nullptr;
  // Priority: Scale(tensor) > scale/scales(attr)
  if (HasInput(op, scope, "Scale")) {
    auto scales_name = op->Input("Scale").front();
    auto scales_tensor = scope->FindTensor(scales_name);
    if (scales_tensor->persistable()) {
      int scales_size = scales_tensor->numel();
      auto scales_value = scales_tensor->data<float>();
      if (scales_size == 1) {
        scales_operand = converter->AddConstantOperand(
            std::vector<float>{scales_value[0], scales_value[0]});
      } else if (scales_size == 2) {
        scales_operand = converter->AddConstantOperand(
            std::vector<float>{scales_value[0], scales_value[1]});
      } else {
        LOG(FATAL) << "Should only have 1 or 2 scales.";
        return PARAMETER_ERROR;
      }
    } else {
      if (std::find(ops_has_one_scale.begin(),
                    ops_has_one_scale.end(),
                    op_type) != ops_has_one_scale.end()) {
        // Generate scales_operand by concat
        auto scale_operand = converter->GetMappedOperand(scales_name);
        auto axis_operand = converter->AddConstantOperand<int>(0);
        scales_operand = converter->AddOutputOperand();
        converter->AddOperation(NNADAPTER_CONCAT,
                                {scale_operand, scale_operand, axis_operand},
                                {scales_operand});
      } else {
        scales_operand = converter->GetMappedOperand(scales_name);
      }
    }
  } else if (op->HasAttr("scale")) {
    if (std::find(ops_has_one_scale.begin(),
                  ops_has_one_scale.end(),
                  op_type) != ops_has_one_scale.end()) {
      float scale = op->GetAttr<float>("scale");
      scales_operand =
          converter->AddConstantOperand(std::vector<float>{scale, scale});
    } else {
      std::vector<float> scales = op->GetAttr<std::vector<float>>("scale");
      if (!scales.empty())
        scales_operand = converter->AddConstantOperand(scales);
    }
  } else {
    VLOG(5) << op_type << " doesn't have 'Scale'(tensor) or 'scale'(attr).";
  }

  CHECK(shape_operand != nullptr || scales_operand != nullptr);

  // Align_corners operand
  bool align_corners = op->GetAttr<bool>("align_corners");
  auto align_corners_operand = converter->AddConstantOperand(align_corners);

  // Align_mode operand(only for linear_interp)
  std::vector<NNAdapterOperand*> input_operands{
      input_operand, shape_operand, scales_operand, align_corners_operand};
  if (std::find(linear_interp_ops.begin(), linear_interp_ops.end(), op_type) !=
      linear_interp_ops.end()) {
    int align_mode =
        op->HasAttr("align_mode") ? op->GetAttr<int>("align_mode") : 1;
    if (align_mode == 0 && align_corners) {
      align_mode = 1;
    }
    auto align_mode_operand = converter->AddConstantOperand(align_mode);
    input_operands.push_back(align_mode_operand);
  }

  // Output operand
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);
  // Resize operation
  NNAdapterOperationType resize_operation_type;
  if (std::find(nearest_interp_ops.begin(),
                nearest_interp_ops.end(),
                op_type) != nearest_interp_ops.end()) {
    resize_operation_type = NNADAPTER_RESIZE_NEAREST;
  } else if (std::find(linear_interp_ops.begin(),
                       linear_interp_ops.end(),
                       op_type) != linear_interp_ops.end()) {
    resize_operation_type = NNADAPTER_RESIZE_LINEAR;
  } else {
    LOG(FATAL) << "Unsupported op_type: " << op_type;
    return NO_ERROR;
  }
  converter->AddOperation(
      resize_operation_type, input_operands, {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
