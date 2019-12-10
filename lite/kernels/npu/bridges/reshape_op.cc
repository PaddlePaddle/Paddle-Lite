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

#include "lite/operators/reshape_op.h"
#include "lite/core/mir/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/npu/bridges/context.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int ReshapeConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph_ctx = static_cast<Context*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
  auto x_dims = x->dims();

  // Create reshape node and set input node from inputs_map
  auto reshape_node = graph_ctx->AddNode<ge::op::Reshape>(out_var_name);
  reshape_node->set_input_tensor(*graph_ctx->GetNode(x_var_name));

  // Read shape from "ShapeTensor"(input), or "Shape"(input), or "shape"(attr)
  if (HasInputArg(op_info, scope, "ShapeTensor")) {
    LOG(WARNING) << "[NPU] not support \"Shape\" from more than one Tensor.";
    return FAILED;
  } else if (HasInputArg(op_info, scope, "Shape")) {
    auto actual_shape_var_name = op_info->Input("Shape").front();
    if (!graph_ctx->HasNode(actual_shape_var_name)) {
      auto actual_shape =
          scope->FindVar(actual_shape_var_name)->GetMutable<Tensor>();
      auto actual_shape_dims = actual_shape->dims();
      auto actual_shape_data = actual_shape->mutable_data<int>();
      auto shape =
          std::vector<int>(actual_shape_data,
                           actual_shape_data + actual_shape_dims.production());
      auto out_dims = lite::operators::ValidateShape(shape, x_dims);
      auto out_shape = out_dims.Vectorize();
      if (out_shape.size() > 4) {
        LOG(WARNING) << "[NPU] HiAI DDK only supports less than 4 dimensions, "
                        "but Shape has "
                     << out_shape.size();
      }
      auto actual_shape_const_node = graph_ctx->AddNode(
          actual_shape_var_name,
          std::vector<int>(out_shape.begin(), out_shape.end()));
      reshape_node->set_input_w(*actual_shape_const_node);
    } else {
      reshape_node->set_input_w(*graph_ctx->GetNode(actual_shape_var_name));
    }
  } else {
    auto shape = op_info->GetAttr<std::vector<int>>("shape");
    auto out_dims = lite::operators::ValidateShape(shape, x_dims);
    auto out_shape = out_dims.Vectorize();
    if (out_shape.size() > 4) {
      LOG(WARNING) << "[NPU] HiAI DDK only supports less than 4 dimensions, "
                      "but shape has "
                   << out_shape.size();
    }
    reshape_node->set_attr_shape(
        ge::AttrValue::LIST_INT(out_shape.begin(), out_shape.end()));
  }

  if (op_type == "reshape2") {
    // Append an extra reshape node to calc XShape
    std::vector<int64_t> xshape_dims(x_dims.size() + 1, 1);
    for (size_t i = 0; i < x_dims.size(); i++) {
      xshape_dims[i + 1] = x_dims[i];
    }
    if (xshape_dims.size() > 4) {
      LOG(WARNING) << "[NPU] HiAI DDK only supports less than 4 dimensions, "
                      "but XShape has "
                   << xshape_dims.size();
    }
    auto xshape_var_name = op_info->Output("XShape").front();
    auto xshape_node = graph_ctx->AddNode<ge::op::Reshape>(xshape_var_name);
    xshape_node->set_input_tensor(*graph_ctx->GetNode(x_var_name));
    xshape_node->set_attr_shape(
        ge::AttrValue::LIST_INT(xshape_dims.begin(), xshape_dims.end()));
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU,
                         reshape,
                         paddle::lite::subgraph::npu::ReshapeConverter);
REGISTER_SUBGRAPH_BRIDGE(NPU,
                         reshape2,
                         paddle::lite::subgraph::npu::ReshapeConverter);
