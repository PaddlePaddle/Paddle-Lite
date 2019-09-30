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

#include "ai_ddk_lib/include/graph/buffer.h"
#include "ai_ddk_lib/include/graph/graph.h"
#include "ai_ddk_lib/include/graph/model.h"
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "ai_ddk_lib/include/graph/operator.h"
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/utils.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

node_map_type InterpolateConverter(
    const std::shared_ptr<lite::OpLite> interpolate_op,
    const node_map_type& inputs_map) {
  auto scope = interpolate_op->scope();
  auto op_info = interpolate_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = UniqueName(op_type);
  LOG(INFO) << "Converting " + op_type + "...";

  // get input, output and attributes from lite op
  auto x_var_name = op_info->Input("X").front();
  CHECK(inputs_map.count(x_var_name));
  OpList::Global().add(inputs_map.at(x_var_name));

  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  auto x_h = x_dims[2];
  auto x_w = x_dims[3];
  CHECK_EQ(x_dims.size(), 4);
  auto scale = op_info->GetAttr<float>("scale");
  auto out_w = op_info->GetAttr<int>("out_w");
  auto out_h = op_info->GetAttr<int>("out_h");
  auto align_corners = op_info->GetAttr<bool>("align_corners");
  int align_mode = op_info->GetAttr<int>("align_mode");
  CHECK(!(align_mode == 0 && !align_corners))
      << "align_mode = 0 && align_corners = false isn't supported in NPU DDK";

  // priority: OutSize > scale > out_h/out_w
  if (scale > 0) {
    out_h = static_cast<int>(x_h * scale);
    out_w = static_cast<int>(x_w * scale);
    out_h = out_h > 0 ? out_h : -1;
    out_w = out_w > 0 ? out_w : -1;
  }

  // update out_h and out_w if has OutSize
  bool inputs_map_has_w = false;
  if (HasInputArg(op_info, scope, "OutSize")) {
    auto out_size_var_name = op_info->Input("OutSize").front();
    if (inputs_map.count(out_size_var_name)) {
      inputs_map_has_w = true;
    } else {
      auto out_size =
          scope->FindVar(out_size_var_name)->GetMutable<lite::Tensor>();
      CHECK_EQ(out_size->numel(), 2);
      auto out_size_data = out_size->mutable_data<int>();
      // update out_h and out_w if has OutSize
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
  }

  node_map_type outputs_map;
  auto interp_method = op_info->GetAttr<std::string>("interp_method");
  if (interp_method == "bilinear") {
    auto interp_node = std::make_shared<ge::op::ResizeBilinear>(unique_op_type);
    OpList::Global().add(interp_node);
    interp_node->set_input_x(*inputs_map.at(x_var_name));
    if (inputs_map_has_w) {
      auto out_size_var_name = op_info->Input("OutSize").front();
      interp_node->set_input_w(*inputs_map.at(out_size_var_name));
      OpList::Global().add(inputs_map.at(out_size_var_name));
    } else {
      const float largest_multiple = 7.0f;
      float multiple = static_cast<float>(x_h * x_w) / (out_h * out_w);
      CHECK_LT(multiple, largest_multiple)
          << "multiple=(ih*iw)/(oh*ow)=" << multiple
          << " is too large, should not exceed " << largest_multiple
          << " in NPU DDK";
      auto w_const_node =
          std::make_shared<ge::op::Const>(unique_op_type + "/w");
      w_const_node->set_attr_value(
          CreateTensorAndFillData(std::vector<int>({out_h, out_w})));
      interp_node->set_input_w(*w_const_node);
      OpList::Global().add(w_const_node);
    }
    interp_node->set_attr_output_dim_mode(
        2);  // 0: zoom_factor, 1: shrink_factor, 2: height/width
    interp_node->set_attr_align_corners(align_corners);
    outputs_map[op_info->Output("Out").front()] = interp_node;
  } else if (interp_method == "nearest") {
    auto interp_node =
        std::make_shared<ge::op::ResizeNearestNeighbor>(unique_op_type);
    OpList::Global().add(interp_node);
    interp_node->set_input_image(*inputs_map.at(x_var_name));
    if (inputs_map_has_w) {
      auto out_size_var_name = op_info->Input("OutSize").front();
      interp_node->set_input_size(*inputs_map.at(out_size_var_name));
      OpList::Global().add(inputs_map.at(out_size_var_name));
    } else {
      auto w_const_node =
          std::make_shared<ge::op::Const>(unique_op_type + "/w");
      w_const_node->set_attr_value(
          CreateTensorAndFillData(std::vector<int>({out_h, out_w})));
      interp_node->set_input_size(*w_const_node);
      OpList::Global().add(w_const_node);
    }
    interp_node->set_attr_align_corners(align_corners);
    outputs_map[op_info->Output("Out").front()] = interp_node;
  } else {
    LOG(FATAL) << "unsupported interpolate method: " << interp_method;
  }

  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(bilinear_interp,
                    paddle::lite::npu::bridge::InterpolateConverter);
REGISTER_NPU_BRIDGE(nearest_interp,
                    paddle::lite::npu::bridge::InterpolateConverter);
