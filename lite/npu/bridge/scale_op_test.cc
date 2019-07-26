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

#include "lite/operators/scale_op.h"
#include <gtest/gtest.h>
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "lite/core/op_registry.h"
#include "lite/npu/bridge/registry.h"

namespace paddle {
namespace lite {

TEST(NPUBridges, SCALE) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("scale"));

  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* out = scope.Var("out")->GetMutable<Tensor>();
  auto* out_ref = scope.Var("out_ref")->GetMutable<Tensor>();
  x->Resize({10, 20});

  // set data
  for (int i = 0; i < x->dims().production(); i++) {
    x->mutable_data<float>()[i] = static_cast<float>(i % 255);
  }

  // prepare op desc
  cpp::OpDesc op_desc;
  op_desc.SetType("scale");
  op_desc.SetInput("X", {"x"});
  op_desc.SetOutput("Out", {"out"});
  op_desc.SetAttr("bias_after_scale", false);
  op_desc.SetAttr("scale", 0.5f);
  op_desc.SetAttr("bias", 0.125f);

  std::shared_ptr<operators::ScaleOp> scale_op =
      std::make_shared<operators::ScaleOp>("scale");

  scale_op->SetValidPlaces({Place{TARGET(kX86), PRECISION(kFloat)},
                            Place{TARGET(kARM), PRECISION(kFloat)}});
  scale_op->Attach(op_desc, &scope);
  scale_op->CheckShape();
  scale_op->InferShape();

  ge::TensorDesc input_desc(
      ge::Shape(x->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  std::shared_ptr<ge::op::Data> input_node =
      std::make_shared<ge::op::Data>("data");
  input_node->update_input_desc_x(input_desc);
  std::vector<std::shared_ptr<ge::Operator>> input_nodes{input_node};
  auto output_nodes =
      supported_lists.at(scale_op->op_info()->Type())(scale_op, input_nodes);
  CHECK_GT(output_nodes.size(), 0);
}

}  // namespace lite
}  // namespace paddle

USE_LITE_OP(scale);
USE_NPU_BRIDGE(scale);
