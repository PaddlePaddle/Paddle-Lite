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

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/npu/bridges/graph.h"
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
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  CHECK_EQ(x_dims.size(), 4L);
  auto batch_size = x_dims[0];
  auto channel_size = x_dims[1];
  auto spatial_size = x_dims[2] * x_dims[3];
  DDim scale_bias_dims({1, channel_size, 1, 1});
  auto y_name = op_info->Output("Y").front();
  float epsilon = op_info->GetAttr<float>("epsilon");

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Bias node
  std::shared_ptr<Node> bias_node = nullptr;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    auto bias = scope->FindMutableTensor(bias_name);
    auto bias_dims = bias->dims();
    CHECK_EQ(channel_size, bias_dims.production());
    if (spatial_size <= 1) {
      // Bug exists in HiAI DDK when h=1 and w=1
      auto bias_data = bias->mutable_data<float>();
      Tensor y;
      y.Resize(x_dims);
      y.set_persistable(true);
      auto y_data = y.mutable_data<float>();
      for (int i = 0; i < batch_size; i++) {
        std::memcpy(y_data, bias_data, sizeof(float) * channel_size);
        y_data += channel_size;
      }
      graph->Add(y_name, y);
      return SUCCESS;
    } else {
      if (!bias->persistable()) {
        LOG(WARNING) << "[NPU] Only supporting persistable bias tensor.";
        return FAILED;
      }
      bias_node = graph->Add(bias_name, *bias, scale_bias_dims);
    }
  } else {
    if (spatial_size <= 1) {
      // Bug exists in HiAI DDK when h=1 and w=1
      graph->Add(y_name, 0.0f, x_dims);
      return SUCCESS;
    } else {
      bias_node = graph->Add(y_name + "/bias", 0.0f, scale_bias_dims);
    }
  }

  // Scale node
  std::shared_ptr<Node> scale_node = nullptr;
  if (HasInputArg(op_info, scope, "Scale")) {
    auto scale_name = op_info->Input("Scale").front();
    auto scale = scope->FindMutableTensor(scale_name);
    auto scale_dims = scale->dims();
    CHECK_EQ(channel_size, scale_dims.production());
    if (!scale->persistable()) {
      LOG(WARNING) << "[NPU] Only supporting persistable scale tensor.";
      return FAILED;
    }
    scale_node = graph->Add(scale_name, *scale, scale_bias_dims);
  } else {
    scale_node = graph->Add(y_name + "/scale", 1.0f, scale_bias_dims);
  }

  // InstanceNorm node
  auto instance_norm_node = graph->Add<ge::op::InstanceNorm>(y_name);
  auto instance_norm_op = instance_norm_node->data<ge::op::InstanceNorm>();
  instance_norm_op->set_input_x(*x_node->data());
  instance_norm_op->set_input_scale(*scale_node->data());
  instance_norm_op->set_input_bias(*bias_node->data());
  instance_norm_op->set_attr_reduction_indices(ge::AttrValue::LIST_INT({2}));
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
