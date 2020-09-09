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
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int SoftmaxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  // input
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  const int64_t* x_shape_data = const_cast<const int64_t*>(&x_dims.data()[0]);
  size_t length = x_dims.size();
  std::vector<int32_t> i_x_shape_data(length);
  for (size_t i = 0; i < length; i++) {
    i_x_shape_data[i] = static_cast<int>(x_shape_data[i]);
  }
  // output
  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();
  const int64_t* output_shape_data =
      const_cast<const int64_t*>(&output_dims.data()[0]);
  length = output_dims.size();
  std::vector<int32_t> i_output_shape_data(length);
  for (size_t i = 0; i < length; i++) {
    i_output_shape_data[i] = static_cast<int>(output_shape_data[i]);
  }
  int32_t axis = -1;
  if (op_info->HasAttr("axis")) {
    axis = op_info->GetAttr<int>("axis");
  }
  if (axis < 0) {
    axis += x_dims.size();
  }
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_dims.size()).production();
  add_softmax_layer(graph->GetCompilerHandle(),
                    const_cast<const int*>(&i_x_shape_data[0]),
                    x_dims.size(),
                    static_cast<const char*>(x_var_name.c_str()),
                    const_cast<const int*>(&i_output_shape_data[0]),
                    output_dims.size(),
                    static_cast<const char*>(output_var_name.c_str()),
                    inner_num,
                    outer_num,
                    x_dims[axis]);
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(softmax,
                         kBM,
                         paddle::lite::subgraph::bm::SoftmaxConverter);
