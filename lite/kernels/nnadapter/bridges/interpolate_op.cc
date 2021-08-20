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

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int InterpolateConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Some interp ops' scale has only one data(like bilinear_interp), but some
  // interp ops' scale has more than one data(like bilinear_interp_v2)
  const std::vector<std::string> ops_has_one_scale{"bilinear_interp",
                                                   "nearest_interp"};
  const std::vector<std::string> nearest_interp_ops{"nearest_interp",
                                                    "nearest_interp_v2"};
  const std::vector<std::string> linear_interp_ops{"bilinear_interp",
                                                   "bilinear_interp_v2"};

  // Input operand
  auto x_name = op_info->Input("X").front();
  NNAdapterOperand* input_operand = nullptr;
  if (converter->HasOperand(x_name)) {
    input_operand = converter->GetOperand(x_name);
  } else {
    input_operand = converter->AddFloat32VariableOperand(
        scope->FindTensor(x_name)->dims(), x_name);
  }

  // Shape operand
  NNAdapterOperand* shape_operand = nullptr;
  // Priority: SizeTensor(tensor) > OutSize(tensor) > out_d/out_h/out_w(attr)
  if (op_info->HasInput("SizeTensor") &&
      !op_info->Input("SizeTensor").empty()) {
    // Use concat to generate shape_operand
    // Concat inputs
    auto shape_names = op_info->Input("SizeTensor");
    std::vector<NNAdapterOperand*> shape_operands;
    for (size_t i = 0; i < shape_names.size(); i++) {
      auto one_shape_name = shape_names[i];
      NNAdapterOperand* one_shape_operand = nullptr;
      if (converter->HasOperand(one_shape_name)) {
        one_shape_operand = converter->GetOperand(one_shape_name);
      } else {
        auto* one_shape_tensor = scope->FindMutableTensor(one_shape_name);
        if (one_shape_tensor->persistable()) {
          one_shape_operand = converter->AddInt32ConstantOperand(
              one_shape_tensor->mutable_data<int32_t>(), DDim({1}));
        } else {
          one_shape_operand =
              converter->AddInt32VariableOperand(DDim({1}), one_shape_name);
        }
      }
      shape_operands.push_back(one_shape_operand);
    }

    // Concat axis
    auto* axis_operand = converter->AddInt32ConstantOperand(0);
    shape_operands.push_back(axis_operand);

    // Concat output
    std::string shape_name = x_name + "_shape";
    shape_operand = converter->AddInt32VariableOperand(
        DDim({static_cast<int64_t>(shape_names.size())}), shape_name);

    // Concat operation
    std::vector<NNAdapterOperand*> output_operands = {shape_operand};
    converter->AddOperation(
        NNADAPTER_CONCAT, &shape_operands, &output_operands);
  } else if (op_info->HasInput("OutSize") &&
             !op_info->Input("OutSize").empty()) {
    auto shape_name = op_info->Input("OutSize").front();
    if (converter->HasOperand(shape_name)) {
      shape_operand = converter->GetOperand(shape_name);
    } else {
      auto* outsize_tensor = scope->FindMutableTensor(shape_name);
      auto outsize_dims = outsize_tensor->dims();
      if (outsize_tensor->persistable()) {
        shape_operand = converter->AddInt32ConstantOperand(
            outsize_tensor->mutable_data<int32_t>(), outsize_dims);
      } else {
        shape_operand =
            converter->AddInt32VariableOperand(outsize_dims, shape_name);
      }
    }
  } else if (op_info->HasAttr("out_h") && op_info->HasAttr("out_w")) {
    int out_h = op_info->GetAttr<int>("out_h");
    int out_w = op_info->GetAttr<int>("out_w");
    if (out_h > 0 && out_w > 0) {
      std::vector<int> shape_vec{out_h, out_w};
      shape_operand =
          converter->AddInt32ConstantOperand(&(shape_vec.at(0)), DDim({2}));
    }
  } else {
    VLOG(5) << op_type
            << " doesn't have 'SizeTensor', 'OutSize' or 'out_h/out_w'.";
  }

  // Scales operand
  NNAdapterOperand* scales_operand = nullptr;
  // Priority: Scale(tensor) > scale/scales(attr)
  if (op_info->HasInput("Scale") && !op_info->Input("Scale").empty()) {
    auto scales_name = op_info->Input("Scale").front();
    if (std::find(ops_has_one_scale.begin(),
                  ops_has_one_scale.end(),
                  op_type) != ops_has_one_scale.end()) {
      // Use concat to generate scales_operand
      // TODO(zhupengyang): use tile to replace later.
      NNAdapterOperand* scale_operand = nullptr;
      if (converter->HasOperand(scales_name)) {
        scale_operand = converter->GetOperand(scales_name);
      } else {
        auto* scale_tensor = scope->FindMutableTensor(scales_name);
        if (scale_tensor->persistable()) {
          scale_operand = converter->AddFloat32ConstantOperand(
              scale_tensor->mutable_data<float>(), DDim({1}));
        } else {
          scale_operand =
              converter->AddFloat32VariableOperand(DDim({1}), scales_name);
        }
      }
      auto axis_operand = converter->AddInt32ConstantOperand(0);
      std::vector<NNAdapterOperand*> input_operands{
          scales_operand, scales_operand, axis_operand};
      scales_operand =
          converter->AddFloat32VariableOperand(DDim({2}), scales_name + "_all");
      std::vector<NNAdapterOperand*> output_operands{scales_operand};
      converter->AddOperation(
          NNADAPTER_CONCAT, &input_operands, &output_operands);
    } else {
      if (converter->HasOperand(scales_name)) {
        scales_operand = converter->GetOperand(scales_name);
      } else {
        auto* scales_tensor = scope->FindMutableTensor(scales_name);
        auto scales_dims = scales_tensor->dims();
        if (scales_tensor->persistable()) {
          scales_operand = converter->AddFloat32ConstantOperand(
              scales_tensor->mutable_data<float>(), scales_dims);
        } else {
          scales_operand =
              converter->AddFloat32VariableOperand(scales_dims, scales_name);
        }
      }
    }
  } else if (op_info->HasAttr("scale")) {
    if (std::find(ops_has_one_scale.begin(),
                  ops_has_one_scale.end(),
                  op_type) != ops_has_one_scale.end()) {
      auto scale = op_info->GetAttr<float>("scale");
      std::vector<float> scales{scale, scale};
      scales_operand = converter->AddFloat32ConstantOperand(
          &(scales.at(0)), DDim({static_cast<int64_t>(scales.size())}));
    } else {
      std::vector<float> scales = op_info->GetAttr<std::vector<float>>("scale");
      scales_operand = converter->AddFloat32ConstantOperand(
          &(scales.at(0)), DDim({static_cast<int64_t>(scales.size())}));
    }
  } else {
    VLOG(5) << op_type << " doesn't have 'Scale' or 'scale'.";
  }

  if (shape_operand == nullptr && scales_operand == nullptr) {
    LOG(WARNING) << "either shape_operand or scales_operand should be set.";
    return FAILED;
  }

  // Align_corners operand
  bool align_corners = op_info->GetAttr<bool>("align_corners");
  auto align_corners_operand =
      converter->AddBool8ConstantOperand(align_corners);

  // Align_mode operand(only linear_interp has align_mode)
  std::vector<NNAdapterOperand*> input_operands{
      input_operand, shape_operand, scales_operand, align_corners_operand};
  if (std::find(linear_interp_ops.begin(), linear_interp_ops.end(), op_type) !=
      linear_interp_ops.end()) {
    NNAdapterOperand* align_mode_operand = nullptr;
    if (op_info->HasAttr("align_mode")) {
      int align_mode = op_info->GetAttr<int>("align_mode");
      align_mode_operand = converter->AddInt32ConstantOperand(align_mode);
    }
    input_operands.push_back(align_mode_operand);
  }

  // Output operand
  auto out_name = op_info->Output("Out").front();
  auto* output_operand = converter->AddFloat32VariableOperand(
      scope->FindTensor(out_name)->dims(), out_name);

  // Resize_nearest operation
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
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
    LOG(WARNING) << "unsupported op_type: " << op_type;
    return FAILED;
  }
  converter->AddOperation(
      resize_operation_type, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    nearest_interp,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::InterpolateConverter);
REGISTER_SUBGRAPH_BRIDGE(
    nearest_interp_v2,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::InterpolateConverter);
REGISTER_SUBGRAPH_BRIDGE(
    bilinear_interp,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::InterpolateConverter);
REGISTER_SUBGRAPH_BRIDGE(
    bilinear_interp_v2,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::InterpolateConverter);
