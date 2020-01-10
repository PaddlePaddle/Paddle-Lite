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

int DropoutConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_type = kernel->GetInputDeclType("X");
  CHECK(x_type->precision() == PRECISION(kFloat));
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto x_rank = x_dims.size();
  CHECK_GE(x_rank, 2);

  auto out_name = op_info->Output("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));

  auto dropout_implementation =
      op_info->GetAttr<std::string>("dropout_implementation");
  auto scale = 1 - op_info->GetAttr<float>("dropout_prob");
  if (dropout_implementation == "upscale_in_train") {
    scale = 1.f;
  }
  // HiAI only support [n, c, 1, 1] for the shape of scale
  std::vector<int64_t> scale_shape = {
      1, x_rank < 3 ? 1 : x_dims[x_rank - 3], 1, 1};

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x, CvtShape(x_dims));
  }

  // Scale node
  auto scale_node = graph->Add<ge::op::Scale>(out_name);
  auto scale_op = scale_node->data<ge::op::Scale>();
  scale_op->set_input_x(*x_node->data());
  scale_op->set_attr_axis(1);

  // Add filter node(fill with scale)
  auto filter_node = graph->Add(out_name + "/filter", scale, scale_shape);
  scale_op->set_input_filter(*filter_node->data());

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(dropout,
                         kNPU,
                         paddle::lite::subgraph::npu::DropoutConverter);
