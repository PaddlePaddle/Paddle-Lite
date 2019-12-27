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

#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int ConcatConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " << op_type << " ... ";

  // Get input and output vars and op attributes
  auto x_names = op_info->Input("X");
  auto x_type = kernel->GetInputDeclType("X");
  CHECK(x_type->precision() == PRECISION(kFloat));
  CHECK(x_type->layout() == DATALAYOUT(kNCHW));
  auto out_name = op_info->Output("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));
  auto axis = op_info->GetAttr<int>("axis");
  auto num = x_names.size();

  // Traverse all of input nodes which are added into the new created concat
  // node
  auto concat_node = graph->AddNode<ge::op::Concat>(out_name);
  concat_node->set_attr_axis(axis);
  concat_node->set_attr_N(num);
  concat_node->create_dynamic_input_x(num);
  int idx = 1;
  for (auto& x_name : x_names) {
    auto x = scope->FindMutableTensor(x_name);
    auto x_dims = x->dims();
    std::shared_ptr<ge::Operator> x_node = nullptr;
    if (graph->HasNode(x_name)) {
      x_node = graph->GetNode(x_name);
    } else {
      x_node = graph->AddNode(x_name, x_dims);
    }
    concat_node->set_dynamic_input_x(idx, *x_node);
    idx++;
  }
  return SUCCESS;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU,
                         concat,
                         paddle::lite::subgraph::npu::ConcatConverter);
