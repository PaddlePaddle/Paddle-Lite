// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <bmcompiler_defs.h>
#include <bmcompiler_if.h>
#include <bmcompiler_op_code.h>
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int AbsConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  std::cout << "active abs" << std::endl;
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindTensor(x_name);
  auto x_dims = x->dims();
  const int64_t* input_shape_data =
      const_cast<const int64_t*>(&x_dims.data()[0]);
  std::vector<int> i_x_shape_data(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int>(input_shape_data[i]);
  }

  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindTensor(out_name);
  auto out_shape = out->dims().Vectorize();

  std::vector<int> ouput_shape_data(out->dims().size());
  for (size_t i = 0; i < out->dims().size(); i++) {
    ouput_shape_data[i] = static_cast<int>(out_shape[i]);
  }

  add_active_layer(graph->GetCompilerHandle(),
                   const_cast<const int*>(&i_x_shape_data[0]),
                   x_dims.size(),
                   static_cast<const char*>(x_name.c_str()),
                   const_cast<const int*>(&ouput_shape_data[0]),
                   out->dims().size(),
                   out_name.c_str(),
                   ACTIVE_ABSVAL);
  graph->AddNode(out_name);

  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(abs, kBM, paddle::lite::subgraph::bm::AbsConverter);
