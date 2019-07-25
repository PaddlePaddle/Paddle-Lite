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

#include "lite/operators/pool_op.h"
#include <gtest/gtest.h>
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "lite/core/op_registry.h"
#include "lite/npu/bridge/registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

TEST(npu_bridge_pool_op, test) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK_EQ(bridges.HasType("pool2d"), true);

  Scope scope;
  auto* x = scope.Var("X")->GetMutable<Tensor>();
  auto* output = scope.Var("Out")->GetMutable<Tensor>();
  x->Resize(DDim(std::vector<int64_t>({1, 10, 20, 20})));

  for (int i = 0; i < x->dims().production(); i++) {
    x->mutable_data<float>()[i] = i;
  }

  // prepare op desc
  cpp::OpDesc op_desc;
  op_desc.SetType("pool2d");
  op_desc.SetInput("X", {"X"});
  op_desc.SetOutput("Out", {"Out"});
  op_desc.SetAttr("pooling_type", std::string("max"));
  op_desc.SetAttr("ksize", std::vector<int>({3, 3}));
  op_desc.SetAttr("global_pooling", false);
  op_desc.SetAttr("strides", std::vector<int>({1, 1}));
  op_desc.SetAttr("paddings", std::vector<int>({1, 1}));

  std::shared_ptr<operators::PoolOpLite> pool_op =
      std::make_shared<operators::PoolOpLite>("pool");

  pool_op->SetValidPlaces({Place{TARGET(kX86), PRECISION(kFloat)},
                           Place{TARGET(kARM), PRECISION(kFloat)}});
  pool_op->Attach(op_desc, &scope);
  pool_op->CheckShape();
  pool_op->InferShape();

  ge::TensorDesc input_desc(
      ge::Shape({1, 10, 20, 20}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  std::shared_ptr<ge::op::Data> input_node =
      std::make_shared<ge::op::Data>("data");
  input_node->update_input_desc_x(input_desc);
  std::vector<std::shared_ptr<ge::Operator>> input_nodes{input_node};
  auto output_nodes =
      supported_lists.at(pool_op->op_info()->Type())(pool_op, input_nodes);
  CHECK_GT(output_nodes.size(), 0);
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(pool2d);
USE_NPU_BRIDGE(pool2d);
