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

#include "lite/operators/softmax_op.h"
#include <gtest/gtest.h>
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "lite/core/op_registry.h"
#include "lite/npu/bridge/registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

TEST(NPUBridges, softmax) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("softmax"));

  Scope scope;
  auto* x = scope.Var("X")->GetMutable<Tensor>();
  auto* output = scope.Var("Out")->GetMutable<Tensor>();
  x->Resize(DDim(std::vector<int64_t>({1, 10, 20, 20})));

  for (int i = 0; i < x->dims().production(); i++) {
    x->mutable_data<float>()[i] = i;
  }

  cpp::OpDesc op_desc;
  op_desc.SetType("softmax");
  op_desc.SetInput("X", {"X"});
  op_desc.SetOutput("Out", {"Out"});
  op_desc.SetAttr("axis", -1);

  std::shared_ptr<operators::SoftmaxOp> softmax_op =
      std::make_shared<operators::SoftmaxOp>("softmax");

  softmax_op->SetValidPlaces({Place{TARGET(kX86), PRECISION(kFloat)},
                              Place{TARGET(kARM), PRECISION(kFloat)}});
  softmax_op->Attach(op_desc, &scope);
  softmax_op->CheckShape();
  softmax_op->InferShape();

  ge::TensorDesc input_desc(
      ge::Shape({1, 10, 20, 20}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  std::shared_ptr<ge::op::Data> input_node =
      std::make_shared<ge::op::Data>("data");
  input_node->update_input_desc_x(input_desc);
  std::vector<std::shared_ptr<ge::Operator>> input_nodes{input_node};
  auto output_nodes = supported_lists.at(softmax_op->op_info()->Type())(
      softmax_op, input_nodes);
  CHECK_GT(output_nodes.size(), 0);
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(softmax);
USE_NPU_BRIDGE(softmax);
