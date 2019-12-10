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

#include "lite/core/mir/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/xpu/bridges/context.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int MulConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph_ctx = static_cast<Context*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input, and attributes
  auto x_var_name = op_info->Input("X").front();
  auto y_var_name = op_info->Input("Y").front();
  auto out_var_name = op_info->Output("Out").front();
  auto y = scope->FindMutableTensor(y_var_name);
  auto y_dims = y->dims();
  CHECK_EQ(y_dims.size(), 2) << "xpu now only support y_dims.size() == 2";

  auto x_num_col_dims = op_info->GetAttr<int>("x_num_col_dims");
  CHECK_EQ(x_num_col_dims, 1) << "xpu now only support x_num_col_dims == 1";
  auto y_num_col_dims = op_info->GetAttr<int>("x_num_col_dims");
  CHECK_EQ(y_num_col_dims, 1) << "xpu now only support y_num_col_dims == 1";

  // Flatten x node
  auto x_node = graph_ctx->AddNode(
      x_var_name + "/flatten",
      graph_ctx->builder_.CreateBatchFlatten(*graph_ctx->GetNode(x_var_name)));

  // Transpose y data and create y node
  Tensor transpose_y;
  DDim transpose_y_dims(std::vector<int64_t>{y_dims[1], y_dims[0]});
  transpose_y.Resize(transpose_y_dims);
  auto transpose_y_data = transpose_y.mutable_data<float>();
  auto y_data = y->mutable_data<float>();
  for (int i = 0; i < transpose_y_dims[0]; i++) {
    for (int j = 0; j < transpose_y_dims[1]; j++) {
      transpose_y_data[i * transpose_y_dims[1] + j] =
          y_data[j * transpose_y_dims[0] + i];
    }
  }
  auto y_const_node =
      graph_ctx->AddNode(y_var_name + "/transpose", transpose_y);

  // Create mul node and set params from op
  graph_ctx->AddNode(
      out_var_name,
      graph_ctx->builder_.CreateDense(*x_node,
                                      static_cast<int>(y_dims[1]),
                                      ::xtcl::NullValue<::xtcl::DataType>(),
                                      *y_const_node));
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(XPU, mul, paddle::lite::subgraph::xpu::MulConverter);
