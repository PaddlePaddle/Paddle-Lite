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

#include "lite/operators/batch_norm_op.h"
#include <gtest/gtest.h>
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "lite/core/op_registry.h"
#include "lite/npu/bridge/registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

TEST(NPUBridges, batch_norm) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("batch_norm"));

  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* scale = scope.Var("scale")->GetMutable<Tensor>();
  auto* bias = scope.Var("bias")->GetMutable<Tensor>();
  auto* mean = scope.Var("mean")->GetMutable<Tensor>();
  auto* variance = scope.Var("variance")->GetMutable<Tensor>();
  auto* y = scope.Var("y")->GetMutable<Tensor>();
  auto* y_ref = scope.Var("y_ref")->GetMutable<Tensor>();
  x->Resize({1, 32, 10, 20});
  auto x_dims = x->dims();
  const int64_t channel_size = x_dims[1];  // NCHW
  scale->Resize({channel_size});
  bias->Resize({channel_size});
  mean->Resize({channel_size});
  variance->Resize({channel_size});

  // set data
  for (int i = 0; i < x_dims.production(); i++) {
    x->mutable_data<float>()[i] = 1.0f;
  }
  for (int i = 0; i < channel_size; i++) {
    scale->mutable_data<float>()[i] = 1.f;
    bias->mutable_data<float>()[i] = 1.f;
    mean->mutable_data<float>()[i] = 1.f;
    variance->mutable_data<float>()[i] = 1.f;
  }

  // prepare op desc
  cpp::OpDesc op_desc;
  op_desc.SetType("batch_norm");
  op_desc.SetInput("X", {"x"});
  op_desc.SetInput("Scale", {"scale"});
  op_desc.SetInput("Bias", {"bias"});
  op_desc.SetInput("Mean", {"mean"});
  op_desc.SetInput("Variance", {"variance"});
  op_desc.SetOutput("Y", {"y"});
  op_desc.SetAttr("is_test", static_cast<int>(1));
  op_desc.SetAttr("use_global_stats", false);
  op_desc.SetAttr("epsilon", 1e-5f);
  op_desc.SetAttr("momentum", 0.9f);
  op_desc.SetAttr("data_layout", std::string("NCHW"));

  std::shared_ptr<operators::BatchNormOp> batch_norm_op =
      std::make_shared<operators::BatchNormOp>("batch_norm");

  batch_norm_op->SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)}});
  batch_norm_op->Attach(op_desc, &scope);
  batch_norm_op->CheckShape();
  batch_norm_op->InferShape();

  // convert op
  ge::TensorDesc input_desc(
      ge::Shape(x_dims.Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  std::shared_ptr<ge::op::Data> input_node =
      std::make_shared<ge::op::Data>("data");
  input_node->update_input_desc_x(input_desc);
  std::vector<std::shared_ptr<ge::Operator>> input_nodes{input_node};
  auto output_nodes = supported_lists.at(batch_norm_op->op_info()->Type())(
      batch_norm_op, input_nodes);
  CHECK_GT(output_nodes.size(), 0);
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(batch_norm);
USE_NPU_BRIDGE(batch_norm);
