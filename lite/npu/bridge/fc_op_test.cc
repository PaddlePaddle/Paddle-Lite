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

#include "lite/operators/fc_op.h"
#include <gtest/gtest.h>
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "lite/core/op_registry.h"
#include "lite/npu/bridge/registry.h"

namespace paddle {
namespace lite {

TEST(NPUBridges, FC) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("fc"));

  // prepare variables
  Scope scope;
  auto* input = scope.Var("input")->GetMutable<Tensor>();
  auto* w = scope.Var("w")->GetMutable<Tensor>();
  auto* bias = scope.Var("bias")->GetMutable<Tensor>();
  auto* out = scope.Var("out")->GetMutable<Tensor>();
  auto* out_ref = scope.Var("out_ref")->GetMutable<Tensor>();
  input->Resize({1, 20});
  w->Resize({20, 30});
  bias->Resize({30});

  // set data
  for (int i = 0; i < input->dims().production(); i++) {
    input->mutable_data<float>()[i] = 1.0f;
  }
  for (int i = 0; i < w->dims().production(); i++) {
    w->mutable_data<float>()[i] = 0.5f;
  }
  for (int i = 0; i < bias->dims().production(); i++) {
    bias->mutable_data<float>()[i] = 0.125f;
  }

  // prepare op desc
  cpp::OpDesc op_desc;
  op_desc.SetType("fc");
  op_desc.SetInput("Input", {"input"});
  op_desc.SetInput("W", {"w"});
  op_desc.SetInput("Bias", {"bias"});
  op_desc.SetOutput("Out", {"out"});
  op_desc.SetAttr("in_num_col_dims", static_cast<int>(1));

  std::shared_ptr<operators::FcOpLite> fc_op =
      std::make_shared<operators::FcOpLite>("fc");

  fc_op->SetValidPlaces({Place{TARGET(kX86), PRECISION(kFloat)},
                         Place{TARGET(kARM), PRECISION(kFloat)}});
  fc_op->Attach(op_desc, &scope);
  fc_op->CheckShape();
  fc_op->InferShape();

  // convert op
  ge::TensorDesc input_desc(
      ge::Shape(input->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  std::shared_ptr<ge::op::Data> input_node =
      std::make_shared<ge::op::Data>("data");
  input_node->update_input_desc_x(input_desc);
  std::vector<std::shared_ptr<ge::Operator>> input_nodes{input_node};
  auto output_nodes =
      supported_lists.at(fc_op->op_info()->Type())(fc_op, input_nodes);
  CHECK_GT(output_nodes.size(), 0);
#if 0
  // build ir graph and generate model
  ge::Graph graph("graph");
  std::vector<ge::Operator> graph_inputs{*(input_nodes[0])};
  std::vector<ge::Operator> graph_outputs{*(output_nodes[0])};
  graph.SetInputs(graph_inputs).SetOutputs(graph_outputs);
  ge::Model model("model", "version");
  model.SetGraph(graph);
  domi::HiaiIrBuild ir_build;
  domi::ModelBufferData model_buf;
  ir_build.CreateModelBuff(model, model_buf);
  CHECK(ir_build.BuildIRModel(model, model_buf));
  // load model
  std::shared_ptr<hiai::AiModelMngerClient> model_client =
  std::make_shared<hiai::AiModelMngerClient>();
  CHECK_EQ(model_client->Init(nullptr), 0);
  std::shared_ptr<hiai::AiModelBuilder> model_builder =
  std::make_shared<hiai::AiModelBuilder>(model_client);
  std::shared_ptr<hiai::AiModelDescription> model_desc =
  std::make_shared<hiai::AiModelDescription>("hiai.om", 3, 0, 0, 0);
  model_desc->SetModelBuffer(model_buf.data, model_buf.length);
  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_descs;
  model_descs.push_back(model_desc);
  model_client->Load(model_descs);
  // prepare input&output tensors
  std::vector<hiai::TensorDimension> input_dims;
  std::vector<hiai::TensorDimension> output_dims;
  CHECK_EQ(model_client->GetModelIOTensorDim(std::string("hiai.om"), input_dims,
  output_dims), 0);
  std::shared_ptr<hiai::AiTensor> input_tensor =
  std::make_shared<hiai::AiTensor>();
  std::shared_ptr<hiai::AiTensor> output_tensor =
  std::make_shared<hiai::AiTensor>();
  std::vector<std::shared_ptr<hiai::AiTensor>> input_tensors;
  std::vector<std::shared_ptr<hiai::AiTensor>> output_tensors;
  input_tensor->Init(&input_dims[0]);
  CHECK(input_tensor->GetSize() > input->dims().production() *
  sizeof(float));
  memcpy(input_tensor->GetBuffer(), input->mutable_data<float>(),
  input->dims().production() * sizeof(float));
  input_tensors.push_back(input_tensor);
  output_tensor->Init(&output_dims[0]);
  output_tensors.push_back(output_tensor);
  // run model and get output buffer
  hiai::AiContext model_ctx;
  std::string model_key = "model";
  std::string model_value = "hiai.om";
  model_ctx.AddPara(model_key, model_value);
  int istamp;
  CHECK_EQ(model_client->Process(model_ctx, input_tensors, output_tensors, 1000,
  istamp), 0);
  memcpy(output_tensor->GetBuffer(), output->mutable_data<float>(),
  output->dims().production() * sizeof(float));
#endif
}

}  // namespace lite
}  // namespace paddle

USE_LITE_OP(fc);
USE_NPU_BRIDGE(fc);
