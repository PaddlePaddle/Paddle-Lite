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

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int InterpolateConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto x_h = x_dims[2];
  auto x_w = x_dims[3];
  CHECK_EQ(x_dims.size(), 4);
  auto out_name = op_info->Output("Out").front();
  auto scale = op_info->GetAttr<float>("scale");
  auto out_w = op_info->GetAttr<int>("out_w");
  auto out_h = op_info->GetAttr<int>("out_h");
  auto align_corners = op_info->GetAttr<bool>("align_corners");
  int align_mode =
      op_info->HasAttr("align_mode") ? op_info->GetAttr<int>("align_mode") : 1;
  auto interp_method = op_info->GetAttr<std::string>("interp_method");
  if (align_mode == 0 && !align_corners) {
    LOG(WARNING) << "[NPU] align_mode = 0 && "
                    "align_corners = false isn't "
                    "supported in HiAI DDK";
    return FAILED;
  }

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Priority: OutSize > scale > out_h/out_w
  if (scale > 0) {
    out_h = static_cast<int>(x_h * scale);
    out_w = static_cast<int>(x_w * scale);
    out_h = out_h > 0 ? out_h : -1;
    out_w = out_w > 0 ? out_w : -1;
  }

  // Update out_h and out_w and create out_size node if has OutSize
  std::shared_ptr<Node> out_size_node = nullptr;
  if (HasInputArg(op_info, scope, "OutSize")) {
    auto out_size_name = op_info->Input("OutSize").front();
    if (graph->Has(out_size_name)) {
      out_size_node = graph->Get(out_size_name);
    } else {
      auto out_size = scope->FindMutableTensor(out_size_name);
      CHECK_EQ(out_size->numel(), 2);
      CHECK(out_size->persistable());
      auto out_size_data = out_size->mutable_data<int>();
      // Update out_h and out_w if has OutSize
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
  }
  if (out_size_node == nullptr) {
    if (interp_method == "bilinear") {
      const float largest_multiple = 7.0f;
      float multiple = static_cast<float>(x_h * x_w) / (out_h * out_w);
      if (multiple >= largest_multiple) {
        LOG(WARNING) << "[NPU] multiple=(ih*iw)/(oh*ow)=" << multiple
                     << " is too large, should not exceed " << largest_multiple
                     << " in HiAI DDK";
        return FAILED;
      }
    }
    out_size_node =
        graph->Add(out_name + "/out_size", std::vector<int>({out_h, out_w}));
  }

  if (interp_method == "bilinear") {
    auto bilinear_interp_node = graph->Add<ge::op::ResizeBilinear>(out_name);
    auto bilinear_interp_op =
        bilinear_interp_node->data<ge::op::ResizeBilinear>();
    bilinear_interp_op->set_input_x(*x_node->data());
    bilinear_interp_op->set_input_size(*out_size_node->data());
    bilinear_interp_op->set_attr_align_corners(align_corners);
  } else if (interp_method == "nearest") {
    auto nearest_interp_node =
        graph->Add<ge::op::ResizeNearestNeighbor>(out_name);
    auto nearest_interp_op =
        nearest_interp_node->data<ge::op::ResizeNearestNeighbor>();
    nearest_interp_op->set_input_image(*x_node->data());
    nearest_interp_op->set_input_size(*out_size_node->data());
    nearest_interp_op->set_attr_align_corners(align_corners);
  } else {
    LOG(WARNING) << "[NPU] Unsupported interpolate method: " << interp_method;
    return FAILED;
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(bilinear_interp,
                         kNPU,
                         paddle::lite::subgraph::npu::InterpolateConverter);
REGISTER_SUBGRAPH_BRIDGE(nearest_interp,
                         kNPU,
                         paddle::lite::subgraph::npu::InterpolateConverter);
