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

#include "lite/operators/mul_op.h"
#include <gtest/gtest.h>
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "lite/core/op_registry.h"
#include "lite/npu/bridge/registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

TEST(NPUBridges, mul) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("mul"));

  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* y = scope.Var("y")->GetMutable<Tensor>();
  auto* out = scope.Var("out")->GetMutable<Tensor>();
  auto* out_ref = scope.Var("out_ref")->GetMutable<Tensor>();
  x->Resize({2, 3, 4});
  y->Resize({3, 4, 5});

  // set data
  for (int i = 0; i < x->dims().production(); i++) {
    x->mutable_data<float>()[i] = static_cast<float>(i % 255);
  }
  for (int i = 0; i < y->dims().production(); i++) {
    y->mutable_data<float>()[i] = static_cast<float>(i % 16) * 0.14f;
  }

  // prepare op desc
  cpp::OpDesc op_desc;
  op_desc.SetType("mul");
  op_desc.SetInput("X", {"x"});
  op_desc.SetInput("Y", {"y"});
  op_desc.SetOutput("Out", {"out"});
  op_desc.SetAttr("x_num_col_dims", 1);
  op_desc.SetAttr("y_num_col_dims", 1);

  std::shared_ptr<operators::MulOpLite> mul_op =
      std::make_shared<operators::MulOpLite>("mul");

  mul_op->SetValidPlaces({Place{TARGET(kX86), PRECISION(kFloat)},
                          Place{TARGET(kARM), PRECISION(kFloat)}});
  mul_op->Attach(op_desc, &scope);
  mul_op->CheckShape();
  mul_op->InferShape();

  ge::TensorDesc input_x_desc(
      ge::Shape(x->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorDesc input_y_desc(
      ge::Shape(y->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  std::shared_ptr<ge::op::Data> input_x_node =
      std::make_shared<ge::op::Data>("x");
  std::shared_ptr<ge::op::Data> input_y_node =
      std::make_shared<ge::op::Data>("y");
  input_x_node->update_input_desc_x(input_x_desc);
  input_y_node->update_input_desc_x(input_y_desc);
  std::vector<std::shared_ptr<ge::Operator>> input_nodes{input_x_node,
                                                         input_y_node};
  auto output_nodes =
      supported_lists.at(mul_op->op_info()->Type())(mul_op, input_nodes);
  CHECK_GT(output_nodes.size(), 0);
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(mul);
USE_NPU_BRIDGE(mul);
