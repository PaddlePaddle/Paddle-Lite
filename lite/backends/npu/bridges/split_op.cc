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

#include "lite/backends/npu/bridges/registry.h"
#include "lite/backends/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridges {

int SplitConverter(cvt_ctx_type* ctx, lite::OpLite* op) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto op_name = ctx->UniqueName(op_type);
  VLOG(3) << "[NPU] Converting " << op_type << " ... ";

  auto x_var_name = op_info->Input("X").front();
  auto out_var_names = op_info->Output("Out");
  auto axis = op_info->GetAttr<int>("axis");
  auto num = op_info->GetAttr<int>("num");
  auto sections = op_info->GetAttr<std::vector<int>>("sections");
  int64_t sections_num = static_cast<int64_t>(sections.size());

  auto split_node = ctx->AddNode<ge::op::Split>(op_name);
  CHECK(ctx->HasNode(x_var_name));
  split_node->set_input_x(*ctx->GetNode(x_var_name));
  split_node->set_attr_axis(static_cast<int64_t>(axis));
  if (num > 0) {
    split_node->set_attr_output_num(static_cast<int64_t>(num));
  } else {
    split_node->set_attr_output_num(sections_num);
    auto size_split = ge::AttrValue::LIST_INT(sections.begin(), sections.end());
    split_node->set_attr_size_split(size_split);
  }

  split_node->create_dynamic_output_y(out_var_names.size());
  int idx = 1;
  for (auto& out_var_name : out_var_names) {
    auto zero_const_node =
        ctx->AddNode<ge::op::Const>(op_name + "/zero" + std::to_string(idx));
    zero_const_node->set_attr_value(CreateTensorAndFillData(0));
    auto add_node = ctx->AddNode<ge::op::Add>(out_var_name);
    add_node->set_input_x1(*split_node, "y" + std::to_string(idx));
    add_node->set_input_x2(*zero_const_node);
    idx++;
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace bridges
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(split, paddle::lite::npu::bridges::SplitConverter);
