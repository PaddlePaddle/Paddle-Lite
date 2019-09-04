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

#include "lite/core/mir/subgraph/subgraph_program_pass.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/ssa_graph.h"
#include "lite/core/program.h"
#include "lite/model_parser/cpp/program_desc.h"
#include "lite/model_parser/model_parser.h"

DEFINE_string(model_dir, "", "model_dir");

namespace paddle {
namespace lite {

TEST(SubgraphTest, models) {
  cpp::ProgramDesc program_desc;
  auto scope = std::make_shared<Scope>();
  // LoadModelPb(FLAGS_model_dir,
  //             FLAGS_model_dir + "/model",
  //             FLAGS_model_dir + "/params",
  //             scope.get(),
  //             &program_desc,
  //             true);
  LoadModelPb(FLAGS_model_dir, "", "", scope.get(), &program_desc);
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
#ifdef LITE_WITH_ARM
      Place{TARGET(kARM), PRECISION(kFloat)},
#endif
#ifdef LITE_WITH_NPU
      Place{TARGET(kNPU), PRECISION(kFloat)},
#endif
  });
  lite::Program program(program_desc, scope, valid_places);
  auto graph = std::unique_ptr<mir::SSAGraph>(new mir::SSAGraph());
  graph->Build(program, valid_places);

  std::vector<std::string> supported_op_types{"concat",
                                              "conv2d",
                                              "depthwise_conv2d",
                                              "batch_norm",
                                              "scale",
                                              "pool2d",
                                              "mul",
                                              "elementwise_add",
                                              "softmax",
                                              "split",
                                              "relu",
                                              "reshape2",
                                              "transpose2"};
  auto* pass = new mir::subgraph::SubgraphProgramPass;
  ASSERT_EQ(pass->FuseSubgraph(graph, supported_op_types), 1);
  LOG(INFO) << "After NPU Pass \n" << Visualize(graph.get());
}

// return output_var_names
std::vector<std::string> AddFCDesc(
    cpp::BlockDesc* block_desc,
    const std::shared_ptr<Scope>& scope,
    const std::vector<std::string>& input_var_names,
    const std::vector<int64_t>& wshape) {
  CHECK_EQ(input_var_names.size(), 1);
  CHECK_EQ(wshape.size(), 2);
  static int id = 0;
  std::string prefix = "fc_" + std::to_string(id);
  auto* op_desc = block_desc->AddOp<cpp::OpDesc>();
  auto* wgt = block_desc->AddVar<cpp::VarDesc>();
  auto* bias = block_desc->AddVar<cpp::VarDesc>();
  auto* out = block_desc->AddVar<cpp::VarDesc>();

  wgt->SetName(prefix + "_W");
  bias->SetName(prefix + "_Bias");
  out->SetName(prefix + "_Out");
  std::vector<std::string> out_var_names{prefix + "_Out"};

  auto* wtensor = scope->Var(prefix + "_W")->GetMutable<lite::Tensor>();
  wtensor->Resize(wshape);
  wtensor->mutable_data<float>();

  auto* btensor = scope->Var(prefix + "_Bias")->GetMutable<lite::Tensor>();
  btensor->Resize({wshape[1]});
  btensor->mutable_data<float>();

  scope->Var(prefix + "_Out")->GetMutable<lite::Tensor>();

  op_desc->SetType("fc");
  op_desc->SetInput("Input", input_var_names);
  op_desc->SetInput("W", {prefix + "_W"});
  op_desc->SetInput("Bias", {prefix + "_Bias"});
  op_desc->SetAttr<int>("in_num_col_dims", 1);
  op_desc->SetOutput("Out", out_var_names);
  id++;
  return out_var_names;
}

std::vector<std::string> AddElementwiseAddDesc(
    cpp::BlockDesc* block_desc,
    const std::shared_ptr<Scope>& scope,
    const std::vector<std::string>& input_X_names,
    const std::vector<std::string>& input_Y_names) {
  // CHECK_EQ(input_var_names.size(), 2);
  static int id = 0;
  std::string prefix = "elementwise_add_" + std::to_string(id);
  auto* op_desc = block_desc->AddOp<cpp::OpDesc>();
  auto* out = block_desc->AddVar<cpp::VarDesc>();

  out->SetName(prefix + "_Out");
  std::vector<std::string> out_var_names{prefix + "_Out"};

  scope->Var(prefix + "_Out")->GetMutable<lite::Tensor>();

  op_desc->SetType("elementwise_add");
  op_desc->SetInput("X", input_X_names);
  op_desc->SetInput("Y", input_Y_names);
  op_desc->SetOutput("Out", out_var_names);
  op_desc->SetAttr("axis", -1);
  id++;
  return out_var_names;
}

std::vector<std::string> AddFeedDesc(
    cpp::BlockDesc* block_desc,
    const std::shared_ptr<Scope>& scope,
    const std::vector<std::string>& input_X_names) {
  // CHECK_EQ(input_var_names.size(), 1);
  static int id = 0;
  std::string prefix = "feed_" + std::to_string(id);
  auto* op_desc = block_desc->AddOp<cpp::OpDesc>();
  auto* out = block_desc->AddVar<cpp::VarDesc>();

  out->SetName(prefix + "_Out");
  std::vector<std::string> out_var_names{prefix + "_Out"};

  scope->Var(prefix + "_Out")->GetMutable<lite::Tensor>();

  op_desc->SetType("feed");
  op_desc->SetInput("X", input_X_names);
  op_desc->SetOutput("Out", out_var_names);
  op_desc->SetAttr("col", 1);
  id++;
  return out_var_names;
}

std::vector<std::string> AddFetchDesc(
    cpp::BlockDesc* block_desc,
    const std::shared_ptr<Scope>& scope,
    const std::vector<std::string>& input_X_names) {
  // CHECK_EQ(input_var_names.size(), 1);
  static int id = 0;
  std::string prefix = "fetch_" + std::to_string(id);
  auto* op_desc = block_desc->AddOp<cpp::OpDesc>();
  auto* out = block_desc->AddVar<cpp::VarDesc>();

  out->SetName(prefix + "_Out");
  std::vector<std::string> out_var_names{prefix + "_Out"};

  scope->Var(prefix + "_Out")->GetMutable<lite::Tensor>();

  op_desc->SetType("fetch");
  op_desc->SetInput("X", input_X_names);
  op_desc->SetOutput("Out", out_var_names);
  op_desc->SetAttr("col", 1);
  id++;
  return out_var_names;
}

std::unique_ptr<mir::SSAGraph> BuildSimpleNet(
    cpp::ProgramDesc* program_desc,
    const std::shared_ptr<Scope>& scope,
    const std::vector<Place>& valid_places) {
  program_desc->ClearBlocks();
  auto* block_desc = program_desc->AddBlock<cpp::BlockDesc>();
  block_desc->ClearOps();
  block_desc->ClearVars();

  auto* var_desc = block_desc->AddVar<cpp::VarDesc>();
  var_desc->SetName("feed_var");
  auto* feed_var = scope->Var("feed_var")->GetMutable<lite::Tensor>();
  feed_var->Resize({1, 4});
  auto fc1_out = AddFCDesc(block_desc, scope, {"feed_var"}, {4, 5});
  auto fc2_out = AddFCDesc(block_desc, scope, fc1_out, {5, 2});

  lite::Program program(*program_desc, scope, valid_places);
  auto graph = std::unique_ptr<mir::SSAGraph>(new mir::SSAGraph());
  graph->Build(program, valid_places);

  return graph;
}

TEST(SubGraphTest, SimpleNet) {
  cpp::ProgramDesc program_desc;
  std::vector<Place> places{{TARGET(kHost), PRECISION(kFloat)}};
  auto scope = std::make_shared<Scope>();
  auto graph = BuildSimpleNet(&program_desc, scope, places);

  std::vector<std::string> supported_op_types{"fc"};
  auto* pass = new mir::subgraph::SubgraphProgramPass;
  ASSERT_EQ(pass->FuseSubgraph(graph, supported_op_types), 1);

  const int num_nodes = graph->nodes().size();
  ASSERT_EQ(graph->nodes().size(), 9);
  // LOG(INFO) << "After NPU Pass \n" << Visualize(graph.get());
}

}  // namespace lite
}  // namespace paddle
