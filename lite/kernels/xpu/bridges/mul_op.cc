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

#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int MulConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_type = kernel->GetInputDeclType("X");
  CHECK(x_type->precision() == PRECISION(kFloat));
  CHECK(x_type->layout() == DATALAYOUT(kNCHW));
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto y_name = op_info->Input("Y").front();
  auto y_type = kernel->GetInputDeclType("Y");
  CHECK(y_type->precision() == PRECISION(kFloat));
  CHECK(y_type->layout() == DATALAYOUT(kNCHW));
  auto y = scope->FindMutableTensor(y_name);
  auto y_dims = y->dims();
  CHECK_EQ(y_dims.size(), 2) << "[XPU] Now only support y_dims.size() == 2";
  auto out_name = op_info->Output("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));
  auto x_num_col_dims = op_info->GetAttr<int>("x_num_col_dims");
  CHECK_EQ(x_num_col_dims, 1) << "xpu now only support x_num_col_dims == 1";
  auto y_num_col_dims = op_info->GetAttr<int>("x_num_col_dims");
  CHECK_EQ(y_num_col_dims, 1) << "xpu now only support y_num_col_dims == 1";

  // X node
  std::shared_ptr<xtcl::xExpr> x_node = nullptr;
  if (graph->HasNode(x_name)) {
    x_node = graph->GetNode(x_name);
  } else {
    x_node = graph->AddNode(x_name, x_dims);
  }
  // Flatten x node
  x_node = graph->AddNode(x_name + "/flatten",
                          graph->builder_.CreateBatchFlatten(*x_node));

  // Transpose y data and create Y node
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
  auto y_const_node = graph->AddNode(y_name + "/transpose", transpose_y);

  // Dense node
  graph->AddNode(
      out_name,
      graph->builder_.CreateDense(*x_node,
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
