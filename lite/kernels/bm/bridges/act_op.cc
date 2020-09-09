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
#include <bmcompiler_if_lite.h>
#include <bmcompiler_op_code.h>
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int ActConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();
  bool x_is_const = !graph->HasNode(x_var_name);
  std::vector<int32_t> i_x_shape_data(x_dims.size());
  std::vector<int32_t> i_output_shape_data(output_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = x_dims[i];
  }
  for (size_t i = 0; i < output_dims.size(); i++) {
    i_output_shape_data[i] = output_dims[i];
  }
  float alpha = 0.f;
  int active_type_id = 0;
  if (op_type == "relu") {
  } else if (op_type == "leaky_relu") {
    alpha = op_info->GetAttr<float>("alpha");
  } else if (op_type == "sqrt") {
    active_type_id = ACTIVE_SQRT;
  } else if (op_type == "square") {
    active_type_id = ACTIVE_SQUARE;
  } else if (op_type == "sigmoid") {
    active_type_id = ACTIVE_SIGMOID;
  } else {
    LOG(FATAL) << "[BM] unsupport act type";
    return FAILED;
  }
  const float* x_data = const_cast<const float*>(x->mutable_data<float>());
  if (x_is_const) {
    bm_add_const_tensor(graph->GetCompilerHandle(),
                        static_cast<const char*>(x_var_name.c_str()),
                        const_cast<const int*>(&i_x_shape_data[0]),
                        x_dims.size(),
                        static_cast<bm_data_type_t>(DTYPE_FP32),
                        static_cast<const void*>(x_data));
  }
  if (op_type == "relu" || op_type == "leaky_relu") {
    add_relu_layer(graph->GetCompilerHandle(),
                   const_cast<const int*>(&i_x_shape_data[0]),
                   x_dims.size(),
                   static_cast<const char*>(x_var_name.c_str()),
                   const_cast<const int*>(&i_output_shape_data[0]),
                   output_dims.size(),
                   static_cast<const char*>(output_var_name.c_str()),
                   alpha,
                   -1.f);
  } else {
    add_active_layer(graph->GetCompilerHandle(),
                     const_cast<const int*>(&i_x_shape_data[0]),
                     x_dims.size(),
                     static_cast<const char*>(x_var_name.c_str()),
                     const_cast<const int*>(&i_output_shape_data[0]),
                     output_dims.size(),
                     static_cast<const char*>(output_var_name.c_str()),
                     active_type_id);
  }
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(relu, kBM, paddle::lite::subgraph::bm::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(leaky_relu,
                         kBM,
                         paddle::lite::subgraph::bm::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(sqrt, kBM, paddle::lite::subgraph::bm::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(square, kBM, paddle::lite::subgraph::bm::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(sigmoid,
                         kBM,
                         paddle::lite::subgraph::bm::ActConverter);
