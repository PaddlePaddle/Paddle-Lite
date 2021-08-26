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
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int ActEnhanceConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  std::vector<int> i_x_shape_data(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int>(x_shape_data[i]);
  }
  // output
  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();
  std::vector<int32_t> i_output_shape_data(output_dims.size());
  for (size_t i = 0; i < output_dims.size(); i++) {
    i_output_shape_data[i] = output_dims[i];
  }
  auto unique_sigmoid_name =
      lite::subgraph::bm::UniqueName(op_type + "_sigmoid");
  if (op_type == "swish") {
    auto beta = op_info->GetAttr<float>("beta");
    CHECK_EQ(beta, 1.f);
    add_active_layer(graph->GetCompilerHandle(),
                     const_cast<const int*>(&i_x_shape_data[0]),
                     x_dims.size(),
                     static_cast<const char*>(x_var_name.c_str()),
                     const_cast<const int*>(&i_output_shape_data[0]),
                     output_dims.size(),
                     static_cast<const char*>(unique_sigmoid_name.c_str()),
                     ACTIVE_SIGMOID);
    bm_add_binary_layer(graph->GetCompilerHandle(),
                        static_cast<const char*>(x_var_name.c_str()),
                        static_cast<const char*>(unique_sigmoid_name.c_str()),
                        static_cast<const char*>(output_var_name.c_str()),
                        BINARY_MUL);
  } else if (op_type == "hard_swish") {
    auto threshold = op_info->GetAttr<float>("threshold");
    auto scale = op_info->GetAttr<float>("scale");
    auto offset = op_info->GetAttr<float>("offset");

    auto unique_add_offset_name =
        lite::subgraph::bm::UniqueName(op_type + "_add_offset");
    add_const_binary_layer(
        graph->GetCompilerHandle(),
        static_cast<const char*>(x_var_name.c_str()),
        const_cast<const int*>(&i_x_shape_data[0]),
        x_dims.size(),
        offset,
        static_cast<const char*>(unique_add_offset_name.c_str()),
        BINARY_ADD,
        0);

    auto unique_max_name = lite::subgraph::bm::UniqueName(op_type + "_max");
    add_const_binary_layer(
        graph->GetCompilerHandle(),
        static_cast<const char*>(unique_add_offset_name.c_str()),
        const_cast<const int*>(&i_x_shape_data[0]),
        x_dims.size(),
        0,
        static_cast<const char*>(unique_max_name.c_str()),
        BINARY_MAX,
        0);

    auto unique_min_name = lite::subgraph::bm::UniqueName(op_type + "_min");
    add_const_binary_layer(graph->GetCompilerHandle(),
                           static_cast<const char*>(unique_max_name.c_str()),
                           const_cast<const int*>(&i_x_shape_data[0]),
                           x_dims.size(),
                           threshold,
                           static_cast<const char*>(unique_min_name.c_str()),
                           BINARY_MIN,
                           0);

    auto unique_mul_name = lite::subgraph::bm::UniqueName(op_type + "_mul");
    bm_add_binary_layer(graph->GetCompilerHandle(),
                        static_cast<const char*>(x_var_name.c_str()),
                        static_cast<const char*>(unique_min_name.c_str()),
                        static_cast<const char*>(unique_mul_name.c_str()),
                        BINARY_MUL);
    add_const_binary_layer(graph->GetCompilerHandle(),
                           static_cast<const char*>(unique_mul_name.c_str()),
                           const_cast<const int*>(&i_x_shape_data[0]),
                           x_dims.size(),
                           scale,
                           static_cast<const char*>(output_var_name.c_str()),
                           BINARY_DIV,
                           0);
  } else if (op_type == "hard_sigmoid") {
    auto slope = op_info->GetAttr<float>("slope");
    auto offset = op_info->GetAttr<float>("offset");
    auto unique_mul_slope_name =
        lite::subgraph::bm::UniqueName(op_type + "_mul_slope");
    add_const_binary_layer(
        graph->GetCompilerHandle(),
        static_cast<const char*>(x_var_name.c_str()),
        const_cast<const int*>(&i_x_shape_data[0]),
        x_dims.size(),
        slope,
        static_cast<const char*>(unique_mul_slope_name.c_str()),
        BINARY_MUL,
        0);
    auto unique_add_offset_name =
        lite::subgraph::bm::UniqueName(op_type + "_add_offset");
    add_const_binary_layer(
        graph->GetCompilerHandle(),
        static_cast<const char*>(unique_mul_slope_name.c_str()),
        const_cast<const int*>(&i_x_shape_data[0]),
        x_dims.size(),
        offset,
        static_cast<const char*>(unique_add_offset_name.c_str()),
        BINARY_ADD,
        0);
    auto unique_min_name = lite::subgraph::bm::UniqueName(op_type + "_min");
    add_const_binary_layer(
        graph->GetCompilerHandle(),
        static_cast<const char*>(unique_add_offset_name.c_str()),
        const_cast<const int*>(&i_x_shape_data[0]),
        x_dims.size(),
        1,
        static_cast<const char*>(unique_min_name.c_str()),
        BINARY_MIN,
        0);
    add_const_binary_layer(graph->GetCompilerHandle(),
                           static_cast<const char*>(unique_min_name.c_str()),
                           const_cast<const int*>(&i_x_shape_data[0]),
                           x_dims.size(),
                           0,
                           static_cast<const char*>(output_var_name.c_str()),
                           BINARY_MAX,
                           0);
  }
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(swish,
                         kBM,
                         paddle::lite::subgraph::bm::ActEnhanceConverter);
REGISTER_SUBGRAPH_BRIDGE(hard_swish,
                         kBM,
                         paddle::lite::subgraph::bm::ActEnhanceConverter);
REGISTER_SUBGRAPH_BRIDGE(hard_sigmoid,
                         kBM,
                         paddle::lite::subgraph::bm::ActEnhanceConverter);
