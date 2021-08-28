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

int SplitConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  auto output_names = op_info->Output("Out");
  auto axis = op_info->GetAttr<int>("axis");
  auto num = op_info->GetAttr<int>("num");
  auto sections = op_info->GetAttr<std::vector<int>>("sections");
  if (0 == num) {
    num = sections.size();
  }
  if (0 == sections.size()) {
    for (size_t i = 0; i < num; i++) {
      sections.push_back(x_dims[axis] / num);
    }
  }

  int** shape = new int*[num];
  int* dim = new int[num];
  const char** name = new const char*[num];

  for (size_t i = 0; i < num; i++) {
    auto out = scope->FindVar(output_names[i])->GetMutable<lite::Tensor>();
    name[i] = static_cast<const char*>(output_names[i].c_str());
    auto out_dims = out->dims();
    shape[i] = new int[out_dims.size()];
    for (size_t j = 0; j < out_dims.size(); j++) {
      shape[i][j] = out_dims[j];
    }
    dim[i] = out_dims.size();
  }
  add_tf_split_layer(graph->GetCompilerHandle(),
                     const_cast<const int*>(&i_x_shape_data[0]),
                     x_dims.size(),
                     static_cast<const char*>(x_var_name.c_str()),
                     num,
                     shape,
                     dim,
                     name,
                     x_dims.size(),
                     axis,
                     const_cast<const int*>(&sections[0]),
                     num);
  for (size_t i = 0; i < num; i++) {
    graph->AddNode(output_names[i]);
    delete[] shape[i];
  }
  delete[] shape;
  delete[] name;
  delete[] dim;
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(split,
                         kBM,
                         paddle::lite::subgraph::bm::SplitConverter);
