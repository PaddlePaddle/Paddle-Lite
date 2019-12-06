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

int ConcatConverter(cvt_ctx_type* ctx, lite::OpLite* op) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  VLOG(3) << "[NPU] Converting " << op_type << " ... ";

  auto x_var_names = op_info->Input("X");
  auto out_var_name = op_info->Output("Out").front();
  auto axis = op_info->GetAttr<int>("axis");
  auto num = x_var_names.size();
  auto concat_node = ctx->AddNode<ge::op::Concat>(out_var_name);
  concat_node->set_attr_axis(axis);
  concat_node->set_attr_N(num);
  concat_node->create_dynamic_input_x(num);
  int idx = 1;
  for (auto& x_var_name : x_var_names) {
    if (ctx->HasNode(x_var_name)) {
      concat_node->set_dynamic_input_x(idx, *ctx->GetNode(x_var_name));
    } else {
      auto x_const_node = ctx->AddNode<ge::op::Const>(x_var_name);
      auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
      x_const_node->set_attr_value(CvtTensor(x));
      concat_node->set_dynamic_input_x(idx, *x_const_node);
    }
    idx++;
  }
  return SUCCESS;
}

}  // namespace bridges
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(concat, paddle::lite::npu::bridges::ConcatConverter);
