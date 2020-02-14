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
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int MulConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_name = lite::subgraph::bm::UniqueName(op_type);
  // only support y is const
  // input
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  const int64_t* x_shape_data = const_cast<const int64_t*>(&x_dims.data()[0]);
  std::vector<int> i_x_shape_data(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int>(x_shape_data[i]);
  }
  // add reshape layer
  int i_x_reshape_shape_data[2];
  i_x_reshape_shape_data[0] = static_cast<int>(x_shape_data[0]);
  i_x_reshape_shape_data[1] = 1;
  for (size_t i = 1; i < x_dims.size(); i++) {
    i_x_reshape_shape_data[1] *= static_cast<int>(x_shape_data[i]);
  }
  int reshape_param[] = {0, -1};
  auto unique_op_reshape_name =
      lite::subgraph::bm::UniqueName(op_type + "_reshape");
  add_reshape_layer(graph->GetCompilerHandle(),
                    const_cast<const int*>(&i_x_shape_data[0]),
                    x_dims.size(),
                    static_cast<const char*>(x_var_name.c_str()),
                    const_cast<const int*>(&i_x_reshape_shape_data[0]),
                    2,
                    static_cast<const char*>(unique_op_reshape_name.c_str()),
                    const_cast<const int*>(reshape_param));

  auto y_var_name = op_info->Input("Y").front();
  auto y = scope->FindVar(y_var_name)->GetMutable<lite::Tensor>();
  auto y_dims = y->dims();
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
  add_fc_layer(graph->GetCompilerHandle(),
               const_cast<const int*>(&i_x_reshape_shape_data[0]),
               2,
               static_cast<const char*>(unique_op_reshape_name.c_str()),
               const_cast<const int*>(&i_output_shape_data[0]),
               output_dims.size(),
               static_cast<const char*>(output_var_name.c_str()),
               static_cast<const char*>(unique_op_name.c_str()),
               i_x_reshape_shape_data[1],
               i_output_shape_data[1],
               static_cast<const float*>(y->mutable_data<float>()),
               nullptr,
               0,
               0);
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(mul, kBM, paddle::lite::subgraph::bm::MulConverter);
