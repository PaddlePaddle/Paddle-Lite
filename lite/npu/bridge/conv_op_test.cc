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

#include "lite/operators/conv_op.h"
#include <gtest/gtest.h>
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "lite/core/op_registry.h"
#include "lite/npu/bridge/registry.h"

namespace paddle {
namespace lite {

TEST(NPUBridges, CONV) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("conv2d"));
  CHECK(bridges.HasType("depthwise_conv2d"));

  // prepare variables
  Scope scope;
  auto* input = scope.Var("input")->GetMutable<Tensor>();
  auto* filter = scope.Var("filter")->GetMutable<Tensor>();
  auto* bias = scope.Var("bias")->GetMutable<Tensor>();
  auto* output = scope.Var("output")->GetMutable<Tensor>();
  input->Resize({1, 4, 16, 16});
  filter->Resize({5, 4, 3, 3});
  bias->Resize({1, 5, 1, 1});

  // set data
  for (int i = 0; i < input->dims().production(); i++) {
    input->mutable_data<float>()[i] = static_cast<float>(i % 255);
  }
  for (int i = 0; i < filter->dims().production(); i++) {
    filter->mutable_data<float>()[i] = static_cast<float>(i % 16) * 0.14f;
  }
  for (int i = 0; i < bias->dims().production(); i++) {
    bias->mutable_data<float>()[i] = static_cast<float>(i % 32) * 0.031f;
  }

  // prepare op desc
  cpp::OpDesc op_desc;
  op_desc.SetType("conv2d");
  op_desc.SetInput("Input", {"input"});
  op_desc.SetInput("Filter", {"filter"});
  op_desc.SetInput("Bias", {"bias"});
  op_desc.SetOutput("Output", {"output"});
  op_desc.SetAttr("dilations", std::vector<int32_t>({1, 1}));
  op_desc.SetAttr("strides", std::vector<int32_t>({1, 1}));
  op_desc.SetAttr("paddings", std::vector<int32_t>({1, 1}));
  op_desc.SetAttr("groups", 1);
  op_desc.SetAttr("fuse_relu", false);

  std::shared_ptr<operators::ConvOpLite> conv_op =
      std::make_shared<operators::ConvOpLite>("conv2d");

  conv_op->SetValidPlaces({Place{TARGET(kX86), PRECISION(kFloat)},
                           Place{TARGET(kARM), PRECISION(kFloat)}});
  conv_op->Attach(op_desc, &scope);
  conv_op->CheckShape();
  conv_op->InferShape();

  // convert op
  ge::TensorDesc input_desc(
      ge::Shape(input->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  std::shared_ptr<ge::op::Data> input_node =
      std::make_shared<ge::op::Data>("data");
  input_node->update_input_desc_x(input_desc);
  std::vector<std::shared_ptr<ge::Operator>> input_nodes{input_node};
  auto output_nodes =
      supported_lists.at(conv_op->op_info()->Type())(conv_op, input_nodes);
  CHECK_GT(output_nodes.size(), 0);
}

}  // namespace lite
}  // namespace paddle

USE_LITE_OP(conv2d);
USE_NPU_BRIDGE(conv2d);
USE_NPU_BRIDGE(depthwise_conv2d);
