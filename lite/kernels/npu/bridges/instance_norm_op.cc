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

int InstanceNormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_type = kernel->GetInputDeclType("X");
  CHECK(x_type->precision() == PRECISION(kFloat));
  CHECK(x_type->layout() == DATALAYOUT(kNCHW));
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  CHECK_EQ(x_dims.size(), 4L);
  auto bs = x_dims[0];
  auto ic = x_dims[1];
  auto out_name = op_info->Output("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));
  float epsilon = op_info->GetAttr<float>("epsilon");

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Scale node
  std::shared_ptr<Node> scale_node = nullptr;
  if (HasInputArg(op_info, scope, "Scale")) {
    auto scale_name = op_info->Input("Scale").front();
    auto scale_type = kernel->GetInputDeclType("Scale");
    CHECK(scale_type->precision() == PRECISION(kFloat));
    CHECK(scale_type->layout() == DATALAYOUT(kNCHW));
    auto scale = scope->FindMutableTensor(scale_name);
    auto scale_dims = scale->dims();
    CHECK_EQ(ic, scale_dims.production());
    scale_node = graph->Add(scale_name, *scale, {1, ic, 1, 1});
  } else {
    scale_node = graph->Add(out_name + "/scale", 1, {1, ic, 1, 1});
  }

  // Bias node
  std::shared_ptr<Node> bias_node = nullptr;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    auto bias_type = kernel->GetInputDeclType("Bias");
    CHECK(bias_type->precision() == PRECISION(kFloat));
    CHECK(bias_type->layout() == DATALAYOUT(kNCHW));
    auto bias = scope->FindMutableTensor(bias_name);
    auto bias_dims = bias->dims();
    CHECK_EQ(ic, bias_dims.production());
    bias_node = graph->Add(bias_name, *bias, {1, ic, 1, 1});
  } else {
    bias_node = graph->Add(out_name + "/bias", 0, {1, ic, 1, 1});
  }

  // InstanceNorm node
  auto instance_norm_node = graph->Add<ge::op::InstanceNorm>(out_name);
  auto instance_norm_op = instance_norm_node->data<ge::op::InstanceNorm>();
  instance_norm_op->set_input_x(*x_node->data());
  instance_norm_op->set_input_scale(*scale_node->data());
  instance_norm_op->set_input_bias(*bias_node->data());
  instance_norm_op->set_attr_reduction_indices(ge::AttrValue::LIST_INT({}));
  instance_norm_op->set_attr_epsilon(epsilon);
  return SUCCESS;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(instance_norm,
                         kNPU,
                         paddle::lite::subgraph::npu::InstanceNormConverter);
