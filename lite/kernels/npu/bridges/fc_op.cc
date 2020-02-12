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

int FCConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  auto input_name = op_info->Input("Input").front();
  auto input_type = kernel->GetInputDeclType("Input");
  CHECK(input_type->precision() == PRECISION(kFloat));
  CHECK(input_type->layout() == DATALAYOUT(kNCHW));
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();
  CHECK_GE(input_dims.size(), 2UL);
  auto w_name = op_info->Input("W").front();
  auto w_type = kernel->GetInputDeclType("W");
  CHECK(w_type->precision() == PRECISION(kFloat));
  CHECK(w_type->layout() == DATALAYOUT(kNCHW));
  auto w = scope->FindMutableTensor(w_name);
  auto w_dims = w->dims();
  CHECK_EQ(w_dims.size(), 2UL);
  auto out_name = op_info->Output("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));
  int in_num_col_dims = op_info->GetAttr<int>("in_num_col_dims");
  int m = input_dims.Slice(0, in_num_col_dims).production();
  int k = input_dims.Slice(in_num_col_dims, input_dims.size()).production();
  int n = w_dims[1];
  CHECK_EQ(k * n, w_dims.production());
  VLOG(3) << "[NPU] input dims: " << input_dims << " w dims: " << w_dims
          << " m: " << m << " k: " << k << " n: " << n;

  // Create input node and reshape it to (m, k, 1, 1)
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    input_node = graph->Add(input_name, *input);
  }
  auto reshaped_input_node =
      graph->Add<ge::op::Reshape>(input_name + "/reshape");
  auto reshaped_input_op = reshaped_input_node->data<ge::op::Reshape>();
  reshaped_input_op->set_input_tensor(*input_node->data());
  reshaped_input_op->set_attr_shape({m, k, 1, 1});
  reshaped_input_op->set_attr_axis(0);

  // Create w const node, set its shape to (n, k, 1, 1) and fill with
  // the transposed w tensor
  Tensor transpose_w;
  transpose_w.Resize({n, k, 1, 1});
  transpose_w.set_persistable(true);
  auto transpose_w_data = transpose_w.mutable_data<float>();
  auto w_data = w->mutable_data<float>();
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      transpose_w_data[j * k + i] = w_data[i * n + j];
    }
  }
  auto trans_w_node = graph->Add(w_name, transpose_w);

  // FC node
  auto fc_node = graph->Add<ge::op::FullConnection>(out_name + "/fc");
  auto fc_op = fc_node->data<ge::op::FullConnection>();
  fc_op->set_input_x(*reshaped_input_node->data());
  fc_op->set_input_w(*trans_w_node->data());
  // Add bias node if bias tensor exists
  if (HasInputArg(op_info, scope, "Bias")) {
    std::shared_ptr<Node> bias_node = nullptr;
    auto bias_name = op_info->Input("Bias").front();
    if (graph->Has(bias_name)) {
      bias_node = graph->Get(bias_name);
    } else {
      auto bias_type = kernel->GetInputDeclType("Bias");
      CHECK(bias_type->precision() == PRECISION(kFloat));
      CHECK(bias_type->layout() == DATALAYOUT(kNCHW));
      auto bias = scope->FindMutableTensor(bias_name);
      auto bias_dims = bias->dims();
      CHECK_EQ(bias_dims.production(), n);
      bias_node = graph->Add(bias_name, *bias, {1, n, 1, 1});
    }
    fc_op->set_input_b(*bias_node->data());
  }
  // Reshape output of FC node from (m, n, 1, 1) to (m, n)
  auto reshaped_fc_node = graph->Add<ge::op::Reshape>(out_name);
  auto reshaped_fc_op = reshaped_fc_node->data<ge::op::Reshape>();
  reshaped_fc_op->set_input_tensor(*fc_node->data());
  reshaped_fc_op->set_attr_shape({m, n});
  reshaped_fc_op->set_attr_axis(0);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(fc, kNPU, paddle::lite::subgraph::npu::FCConverter);
