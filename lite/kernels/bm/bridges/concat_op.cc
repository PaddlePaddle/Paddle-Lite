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
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int ConcatConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  // input
  auto x_names = op_info->Input("X");
  // output
  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();
  const int64_t* output_shape_data =
      const_cast<const int64_t*>(&output_dims.data()[0]);
  std::vector<int32_t> i_output_shape_data(output_dims.size());
  for (size_t i = 0; i < output_dims.size(); i++) {
    i_output_shape_data[i] = static_cast<int>(output_shape_data[i]);
  }
  const int32_t input_num = x_names.size();
  int32_t** shape = new int32_t*[input_num];
  int32_t* dim = new int32_t[input_num];
  const char** name = new const char*[input_num];
  for (size_t i = 0; i < x_names.size(); i++) {
    auto x = scope->FindMutableTensor(x_names[i]);
    name[i] = x_names[i].c_str();
    auto x_dims = x->dims();
    dim[i] = x_dims.size();
    const int64_t* x_shape_data = const_cast<const int64_t*>(&x_dims.data()[0]);
    shape[i] = new int32_t[x_dims.size()];
    for (size_t j = 0; j < x_dims.size(); j++) {
      shape[i][j] = static_cast<int32_t>(x_shape_data[j]);
    }
  }
  auto axis = op_info->GetAttr<int>("axis");
  add_concat_layer(graph->GetCompilerHandle(),
                   input_num,
                   shape,
                   dim,
                   name,
                   const_cast<const int*>(&i_output_shape_data[0]),
                   output_dims.size(),
                   static_cast<const char*>(output_var_name.c_str()),
                   axis);
  for (size_t i = 0; i < x_names.size(); i++) {
    delete[] shape[i];
  }
  delete[] shape;
  delete[] name;
  delete[] dim;
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
REGISTER_SUBGRAPH_BRIDGE(concat,
                         kBM,
                         paddle::lite::subgraph::bm::ConcatConverter);
