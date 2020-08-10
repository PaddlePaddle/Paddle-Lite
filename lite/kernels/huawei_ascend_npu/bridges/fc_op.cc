// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int FCConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input data nodes
  auto input_name = op_info->Input("Input").front();
  auto input = scope->FindTensor(input_name);
  auto input_dims = input->dims();

  auto w_name = op_info->Input("W").front();
  auto w = scope->FindTensor(w_name);
  auto w_dims = w->dims();
  CHECK_EQ(w_dims.size(), 2UL);

  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindTensor(out_name);
  auto out_dims = out->dims();

  int in_num_col_dims = op_info->GetAttr<int>("in_num_col_dims");
  int m = input_dims.Slice(0, in_num_col_dims).production();
  int k = input_dims.Slice(in_num_col_dims, input_dims.size()).production();
  int n = w_dims[1];
  CHECK_EQ(k * n, w_dims.production());

  VLOG(3) << "[HUAWEI_ASCEND_NPU] input_dims = " << input_dims.repr()
          << ", w_dims = " << w_dims.repr()
          << ", in_num_col_dims = " << in_num_col_dims << ", m = " << m
          << ", k = " << k << ", n = " << n;

  // Create input node
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    input_node = graph->Add(input_name, *input);
  }

  // w_node, the ascend ddk will transpose the tensor of w
  // if we set transpose attr to be true
  auto w_node = graph->Add(w_name, *w);

  // fc node
  auto fc_node = graph->Add<ge::op::FullyConnection>(out_name);
  auto fc_op = fc_node->data<ge::op::FullyConnection>();
  fc_op->set_input_x(*input_node->data());
  fc_op->set_input_w(*w_node->data());
  fc_op->set_attr_num_output(n);
  fc_op->set_attr_transpose(true);

  INPUT_UPDATE(fc_op, x, input_node);
  INPUT_UPDATE(fc_op, w, w_node);
  OUTPUT_UPDATE(fc_op, y, fc_node);

  if (HasInputArg(op_info, scope, "Bias")) {
    std::shared_ptr<Node> bias_node = nullptr;
    auto bias_name = op_info->Input("Bias").front();
    if (graph->Has(bias_name)) {
      bias_node = graph->Get(bias_name);
    } else {
      auto bias = scope->FindTensor(bias_name);
      auto bias_dims = bias->dims();
      CHECK_EQ(bias_dims.production(), n);

      VLOG(3) << "[HUAWEI_ASCEND_NPU] bias_dims = " << bias_dims.repr();
      bias_node = graph->Add(bias_name, *bias, {1, n, 1, 1});
    }
    fc_op->set_input_b(*bias_node->data());
    INPUT_UPDATE(fc_op, b, bias_node);
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    fc,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::FCConverter);
