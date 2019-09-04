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

node_map_type ReshapeConverter(const std::shared_ptr<lite::OpLite> reshape_op,
                               const node_map_type& inputs_map) {
  auto scope = reshape_op->scope();
  auto op_info = reshape_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = UniqueName(op_type);
  LOG(INFO) << "Converting " + op_type + "...";

  // get input, output and op attributes
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();

  // create reshape node and set input node from inputs_map
  auto reshape_node = std::make_shared<ge::op::Reshape>(unique_op_type);
  CHECK(inputs_map.count(x_var_name));
  reshape_node->set_input_tensor(*inputs_map.at(x_var_name));
  OpList::Global().add(inputs_map.at(x_var_name));

  // read shape from actual shape tensor as input "w" if 'Shape' is found
  if (HasInputArg(op_info, scope, "Shape")) {
    auto actual_shape_var_name = op_info->Input("Shape").front();
    if (!inputs_map.count(actual_shape_var_name)) {
      auto actual_shape =
          scope->FindVar(actual_shape_var_name)->GetMutable<lite::Tensor>();
      auto actual_shape_dims = actual_shape->dims();
      auto actual_shape_data = actual_shape->mutable_data<int>();
      auto shape =
          std::vector<int>(actual_shape_data,
                           actual_shape_data + actual_shape_dims.production());
      auto out_dims = operators::ValidateShape(shape, x_dims);
      auto out_shape = out_dims.Vectorize();
      if (out_shape.size() > 4) {
        LOG(WARNING)
            << "NPU DDK only supports less than 4 dimensions, but Shape has "
            << out_shape.size();
      }
      auto actual_shape_const_node =
          std::make_shared<ge::op::Const>(actual_shape_var_name);
      actual_shape_const_node->set_attr_value(CreateTensorAndFillData(
          std::vector<int>(out_shape.begin(), out_shape.end())));
      reshape_node->set_input_w(*actual_shape_const_node);
      OpList::Global().add(actual_shape_const_node);
    } else {
      reshape_node->set_input_w(*inputs_map.at(actual_shape_var_name));
      OpList::Global().add(inputs_map.at(actual_shape_var_name));
    }
  } else {
    auto shape = op_info->GetAttr<std::vector<int>>("shape");
    auto out_dims = operators::ValidateShape(shape, x_dims);
    auto out_shape = out_dims.Vectorize();
    if (out_shape.size() > 4) {
      LOG(WARNING)
          << "NPU DDK only supports less than 4 dimensions, but shape has "
          << out_shape.size();
    }
    reshape_node->set_attr_shape(
        ge::AttrValue::LIST_INT(out_shape.begin(), out_shape.end()));
  }
  OpList::Global().add(reshape_node);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = reshape_node;
  if (op_type == "reshape2") {
    // append an extra reshape node to calc XShape
    std::vector<int64_t> xshape_dims(x_dims.size() + 1, 1);
    for (size_t i = 0; i < x_dims.size(); i++) {
      xshape_dims[i + 1] = x_dims[i];
    }
    if (xshape_dims.size() > 4) {
      LOG(WARNING)
          << "NPU DDK only supports less than 4 dimensions, but XShape has "
          << xshape_dims.size();
    }
    auto xshape_node =
        std::make_shared<ge::op::Reshape>(unique_op_type + "/xshape");
    xshape_node->set_input_tensor(*inputs_map.at(x_var_name));
    xshape_node->set_attr_shape(
        ge::AttrValue::LIST_INT(xshape_dims.begin(), xshape_dims.end()));
    OpList::Global().add(xshape_node);
    outputs_map[op_info->Output("XShape").front()] = xshape_node;
  }
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(reshape, paddle::lite::npu::bridge::ReshapeConverter);
REGISTER_NPU_BRIDGE(reshape2, paddle::lite::npu::bridge::ReshapeConverter);
