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

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int InterpolateConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  std::vector<int32_t> i_x_shape_data(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int32_t>(x_dims[i]);
  }
  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();
  std::vector<int32_t> i_output_shape_data(output_dims.size());
  for (size_t i = 0; i < output_dims.size(); i++) {
    i_output_shape_data[i] = static_cast<int32_t>(output_dims[i]);
  }
  auto scale = op_info->GetAttr<float>("scale");
  int32_t i_scale = static_cast<int32_t>(scale);
  bool is_int = false;
  if ((scale - i_scale) < 0.000001f) {
    is_int = true;
  }
  int32_t type = 0;
  if (op_type == "nearest_interp") {
    type = 2;
  } else {
    type = 0;
  }
  is_int = false;
  if (type == 2 && is_int) {
    add_upsample_layer(graph->GetCompilerHandle(),
                       const_cast<const int*>(&i_x_shape_data[0]),
                       x_dims.size(),
                       static_cast<const char*>(x_var_name.c_str()),
                       const_cast<const int*>(&i_output_shape_data[0]),
                       output_dims.size(),
                       static_cast<const char*>(output_var_name.c_str()),
                       i_scale);
  } else {
    add_interp_layer_v2(graph->GetCompilerHandle(),
                        const_cast<const int*>(&i_x_shape_data[0]),
                        x_dims.size(),
                        static_cast<const char*>(x_var_name.c_str()),
                        1,
                        const_cast<const int*>(&i_output_shape_data[0]),
                        output_dims.size(),
                        nullptr,
                        static_cast<const char*>(output_var_name.c_str()),
                        0,
                        0,
                        0,
                        0,
                        type);
  }
  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(nearest_interp,
                         kBM,
                         paddle::lite::subgraph::bm::InterpolateConverter);
REGISTER_SUBGRAPH_BRIDGE(bilinear_interp,
                         kBM,
                         paddle::lite::subgraph::bm::InterpolateConverter);
