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

#include <bmcompiler_if.h>
#include <bmcompiler_op_code.h>
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int MatMulConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);

  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_name = lite::subgraph::bm::UniqueName(op_type);
  // input
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  std::vector<int32_t> i_x_shape_data(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int>(x_dims[i]);
  }
  auto y_var_name = op_info->Input("Y").front();
  auto y = scope->FindVar(y_var_name)->GetMutable<lite::Tensor>();
  auto y_dims = y->dims();
  std::vector<int32_t> i_y_shape_data(y_dims.size());
  for (size_t i = 0; i < y_dims.size(); i++) {
    i_y_shape_data[i] = static_cast<int>(y_dims[i]);
  }
  // output
  auto output_var_name = op_info->Output("Out").front();
  auto out = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto out_dims = out->dims();
  std::vector<int32_t> i_out_shape_data(out_dims.size());
  for (size_t i = 0; i < out_dims.size(); i++) {
    i_out_shape_data[i] = static_cast<int>(out_dims[i]);
  }
  bool transpose_x = op_info->GetAttr<bool>("transpose_X");
  bool transpose_y = op_info->GetAttr<bool>("transpose_Y");
  float alpha = op_info->GetAttr<float>("alpha");
  CHECK_EQ(alpha, 1.f);
  CHECK_EQ(transpose_x, 0);
  CHECK_EQ(transpose_y, 0);

  const float* y_data = const_cast<const float*>(y->mutable_data<float>());
  const float* x_data = const_cast<const float*>(x->mutable_data<float>());
  add_batch_matmul_layer(graph->GetCompilerHandle(),
                         static_cast<const char*>(x_var_name.c_str()),
                         const_cast<const int*>(&i_x_shape_data[0]),
                         x_dims.size(),
                         0,
                         x_data,
                         static_cast<const char*>(y_var_name.c_str()),
                         const_cast<const int*>(&i_y_shape_data[0]),
                         y_dims.size(),
                         0,
                         y_data,
                         static_cast<const char*>(output_var_name.c_str()));
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(matmul,
                         kBM,
                         paddle::lite::subgraph::bm::MatMulConverter);
