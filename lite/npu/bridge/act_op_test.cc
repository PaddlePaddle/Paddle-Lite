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

#include <gtest/gtest.h>
#include <random>
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "lite/core/op_registry.h"
#include "lite/npu/bridge/registry.h"
#include "lite/operators/graph_op.h"
#include "lite/operators/relu_op.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

void relu_ref(const std::shared_ptr<operators::ReluOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  auto x_data = x->data<float>();
  auto out_data = out->mutable_data<float>();
  DDim x_dims = x->dims();
  DDim out_dims = out->dims();
  CHECK_EQ(x_dims.production(), out_dims.production());
  for (int i = 0; i < out_dims.production(); i++) {
    out_data[i] = std::max(0.f, x_data[i]);
  }
}

void test_relu(std::vector<int64_t> x_shape) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("relu"));

  // prepare input&output variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* out = scope.Var("out")->GetMutable<Tensor>();
  auto* out_ref = scope.Var("out_ref")->GetMutable<Tensor>();
  x->Resize(x_shape);

  // initialize input&output data
  std::default_random_engine rand_eng;
  std::uniform_real_distribution<float> rand_dist(-5.0f, 5.0f);
  for (int i = 0; i < x->dims().production(); i++) {
    x->mutable_data<float>()[i] = rand_dist(rand_eng);
  }

  // create act op
  cpp::OpDesc act_op_desc;
  act_op_desc.SetType("relu");
  act_op_desc.SetInput("X", {"x"});
  act_op_desc.SetOutput("Out", {"out"});

  std::shared_ptr<operators::ReluOp> act_op =
      std::make_shared<operators::ReluOp>("relu");
  act_op->SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)},
                          Place{TARGET(kARM), PRECISION(kFloat)}});
  act_op->Attach(act_op_desc, &scope);
  act_op->CheckShape();
  act_op->InferShape();

  // convert act op and build IR graph
  ge::TensorDesc input_desc(
      ge::Shape(x->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  std::shared_ptr<ge::op::Data> input_node =
      std::make_shared<ge::op::Data>("data");
  input_node->update_input_desc_x(input_desc);
  std::vector<std::shared_ptr<ge::Operator>> input_nodes{input_node};
  auto output_nodes =
      supported_lists.at(act_op->op_info()->Type())(act_op, input_nodes);
  CHECK_GT(output_nodes.size(), 0);

  // compile IR graph to om model
  std::vector<ge::Operator> graph_inputs{*input_nodes[0]};
  std::vector<ge::Operator> graph_outputs{*output_nodes[0]};
  std::string graph_name("test_relu");
  // TODO(hong19860320) uncomment the following lines if test device is ready
  // CHECK(npu::BuildNPUClient(graph_inputs, graph_outputs, graph_name));
  // CHECK(npu::BuildNPUClient("/data/local/tmp/test_relu.om", graph_name));

  // create graph op
  cpp::OpDesc graph_op_desc;
  graph_op_desc.SetType("graph_op");
  graph_op_desc.SetInput("Input", {"x"});
  graph_op_desc.SetOutput("Output", {"out"});
  graph_op_desc.SetAttr("graph_name", graph_name);

  std::shared_ptr<operators::GraphOpLite> graph_op =
      std::make_shared<operators::GraphOpLite>("graph_op");
  graph_op->SetValidPlaces({Place{TARGET(kNPU), PRECISION(kFloat)}});
  graph_op->Attach(graph_op_desc, &scope);
  graph_op->CheckShape();
  graph_op->InferShape();

  // create graph op kernel
  auto graph_kernels =
      graph_op->CreateKernels({Place{TARGET(kNPU), PRECISION(kFloat)}});
  CHECK(!graph_kernels.empty());
  auto graph_kernel =
      std::move(graph_kernels.front());  // use the first kernel by default
  auto graph_ctx = ContextScheduler::Global().NewContext(TARGET(kNPU));
  graph_kernel->SetContext(std::move(graph_ctx));

  // TODO(hong19860320) uncomment the following lines if test device is ready
  // perform graph op kernel and copy output tensor('out') to 'out_ref'
  // graph_kernel->Launch();
  out_ref->CopyDataFrom(*out);

  // execute reference implementation and save to output tensor('out')
  relu_ref(act_op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    // TODO(hong19860320) uncomment the following lines if test device is ready
    // EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }
}

TEST(NPUBridges, relu) {
  test_relu({1, 4, 5, 9});
  test_relu({3, 7, 2, 4});
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(relu);
USE_NPU_BRIDGE(relu);

USE_LITE_OP(graph_op);
USE_LITE_KERNEL(graph_op, kNPU, kFloat, kNCHW, def);
