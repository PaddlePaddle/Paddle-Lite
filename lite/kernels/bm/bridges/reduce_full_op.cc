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

int ReduceFullConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);

  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  // input
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  const int64_t* x_shape_data = const_cast<const int64_t*>(&x_dims.data()[0]);
  std::vector<int32_t> i_x_shape_data(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int>(x_shape_data[i]);
  }
  // output
  auto output_var_name = op_info->Output("Out").front();
  auto dim = op_info->GetAttr<std::vector<int32_t>>("dim");
  auto keep_dim = op_info->GetAttr<bool>("keep_dim");
  int op_code = -1;
  if (op_type == "reduce_sum") {
    op_code = REDUCE_SUM;
  } else if (op_type == "reduce_mean") {
    op_code = REDUCE_MEAN;
  } else if (op_type == "reduce_max") {
    op_code = REDUCE_MAX;
  }

  add_reduce_full_layer(graph->GetCompilerHandle(),
                        static_cast<const char*>(x_var_name.c_str()),
                        static_cast<const char*>(output_var_name.c_str()),
                        const_cast<const int*>(&i_x_shape_data[0]),
                        x_dims.size(),
                        const_cast<const int*>(&dim[0]),
                        dim.size(),
                        op_code,
                        static_cast<int>(keep_dim));
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(reduce_sum,
                         kBM,
                         paddle::lite::subgraph::bm::ReduceFullConverter);
REGISTER_SUBGRAPH_BRIDGE(reduce_mean,
                         kBM,
                         paddle::lite::subgraph::bm::ReduceFullConverter);
REGISTER_SUBGRAPH_BRIDGE(reduce_max,
                         kBM,
                         paddle::lite::subgraph::bm::ReduceFullConverter);
