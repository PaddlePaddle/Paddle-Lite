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

#include "lite/core/mir/subgraph/subgraph_detector.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/core/mir/ssa_graph.h"
#include "lite/core/program.h"
#include "lite/model_parser/cpp_desc.h"
#include "lite/model_parser/model_parser.h"

DEFINE_string(model_dir, "", "model_dir");
DEFINE_string(model_file, "", "model file path of combined protobuf model");
DEFINE_string(params_file, "", "params file path of combined protobuf model");

namespace paddle {
namespace lite {

// The helper functions for building model manually
std::vector<std::string> AddFCDesc(
    cpp::BlockDesc* block_desc,
    const std::shared_ptr<Scope>& scope,
    const std::vector<std::string>& input_var_names,
    const std::vector<int64_t>& wshape) {
  CHECK_EQ(input_var_names.size(), 1u);
  CHECK_EQ(wshape.size(), 2u);
  static int id = 0;
  std::string prefix = "fc_" + paddle::lite::to_string(id);
  auto* op_desc = block_desc->AddOp<cpp::OpDesc>();

  auto* wgt = block_desc->AddVar<cpp::VarDesc>();
  wgt->SetName(prefix + "_W");
  auto* wtensor = scope->Var(prefix + "_W")->GetMutable<Tensor>();
  wtensor->Resize(wshape);
  wtensor->mutable_data<float>();

  auto* bias = block_desc->AddVar<cpp::VarDesc>();
  bias->SetName(prefix + "_Bias");
  auto* btensor = scope->Var(prefix + "_Bias")->GetMutable<Tensor>();
  btensor->Resize({wshape[1]});
  btensor->mutable_data<float>();

  auto* out = block_desc->AddVar<cpp::VarDesc>();
  out->SetName(prefix + "_Out");
  std::vector<std::string> out_var_names{prefix + "_Out"};
  scope->Var(prefix + "_Out")->GetMutable<Tensor>();

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
  std::string prefix = "elementwise_add_" + paddle::lite::to_string(id);
  auto* op_desc = block_desc->AddOp<cpp::OpDesc>();
  auto* out = block_desc->AddVar<cpp::VarDesc>();

  out->SetName(prefix + "_Out");
  std::vector<std::string> out_var_names{prefix + "_Out"};

  scope->Var(prefix + "_Out")->GetMutable<Tensor>();

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
  std::string prefix = "feed_" + paddle::lite::to_string(id);
  auto* op_desc = block_desc->AddOp<cpp::OpDesc>();
  auto* out = block_desc->AddVar<cpp::VarDesc>();

  out->SetName(prefix + "_Out");
  std::vector<std::string> out_var_names{prefix + "_Out"};

  scope->Var(prefix + "_Out")->GetMutable<Tensor>();

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
  std::string prefix = "fetch_" + paddle::lite::to_string(id);
  auto* op_desc = block_desc->AddOp<cpp::OpDesc>();
  auto* out = block_desc->AddVar<cpp::VarDesc>();

  out->SetName(prefix + "_Out");
  std::vector<std::string> out_var_names{prefix + "_Out"};

  scope->Var(prefix + "_Out")->GetMutable<Tensor>();

  op_desc->SetType("fetch");
  op_desc->SetInput("X", input_X_names);
  op_desc->SetOutput("Out", out_var_names);
  op_desc->SetAttr("col", 1);
  id++;
  return out_var_names;
}

TEST(Subgraph, detect_simple_model) {
  auto program_desc = std::make_shared<cpp::ProgramDesc>();
  std::vector<Place> valid_places{{TARGET(kHost), PRECISION(kFloat)}};
  auto scope = std::make_shared<Scope>();
  // Build a simple network
  auto* block_desc = program_desc->AddBlock<cpp::BlockDesc>();
  block_desc->ClearOps();
  block_desc->ClearVars();
  auto* var_desc = block_desc->AddVar<cpp::VarDesc>();
  var_desc->SetName("feed_var");
  auto* feed_var = scope->Var("feed_var")->GetMutable<Tensor>();
  feed_var->Resize({1, 4});
  auto fc1_out = AddFCDesc(block_desc, scope, {"feed_var"}, {4, 5});
  auto fc2_out = AddFCDesc(block_desc, scope, fc1_out, {5, 2});
  Program program(program_desc, scope, valid_places);
  auto graph = std::unique_ptr<mir::SSAGraph>(new mir::SSAGraph());
  graph->Build(program, valid_places);
  // Apply subgraph detector and check results
  auto teller = [](mir::Node* node) {
    if (!node->IsStmt()) return false;
    auto& stmt = node->AsStmt();
    auto op_type = stmt.op_type();
    const std::vector<std::string> supported_types = {"fc"};
    return std::find(supported_types.begin(), supported_types.end(), op_type) !=
           supported_types.end();
  };
  std::vector<std::vector<mir::Node*>> subgraphs =
      mir::SubgraphDetector(graph.get(), teller)();
  ASSERT_EQ(subgraphs.size(), 1u);
  ASSERT_EQ(graph->nodes().size(), 9u);
  mir::SubgraphVisualizer(graph.get(), subgraphs)();
}

TEST(Subgraph, detect_custom_model) {
  if (FLAGS_model_dir.empty() && FLAGS_model_file.empty() &&
      FLAGS_params_file.empty()) {
    LOG(INFO) << "Using --model_dir, or --model_file and --params_file to set "
                 "the path of model files.";
    return;
  }
  auto program_desc = std::make_shared<cpp::ProgramDesc>();
  auto scope = std::make_shared<Scope>();
  LoadModelPb(FLAGS_model_dir,
              FLAGS_model_file,
              FLAGS_params_file,
              scope.get(),
              program_desc.get(),
              !FLAGS_model_file.empty() && !FLAGS_params_file.empty());
  std::vector<Place> valid_places({
#ifdef LITE_WITH_ARM
      Place{TARGET(kARM), PRECISION(kFloat)},
#endif
#ifdef LITE_WITH_X86
      Place{TARGET(kX86), PRECISION(kFloat)},
#endif
#ifdef LITE_WITH_NPU
      Place{TARGET(kNPU), PRECISION(kFloat)},
#endif
#ifdef LITE_WITH_HUAWEI_ASCEND_NPU
      Place{TARGET(kHuaweiAscendNPU), PRECISION(kFloat)},
#endif
#ifdef LITE_WITH_XTCL
      Place{TARGET(kXPU), PRECISION(kFloat)},
#endif
  });
  Program program(program_desc, scope, valid_places);
  auto graph = std::unique_ptr<mir::SSAGraph>(new mir::SSAGraph());
  graph->Build(program, valid_places);
  // Apply subgraph detector and check results
  auto teller = [](mir::Node* node) {
    if (!node->IsStmt()) return false;
    auto& stmt = node->AsStmt();
    auto op_type = stmt.op_type();
    const std::vector<std::string> unsupported_types = {
        "feed", "fetch", "subgraph"};
    return std::find(unsupported_types.begin(),
                     unsupported_types.end(),
                     op_type) == unsupported_types.end();
  };
  std::vector<std::vector<mir::Node*>> subgraphs =
      mir::SubgraphDetector(graph.get(), teller)();
  mir::SubgraphVisualizer(graph.get(), subgraphs)();
  ASSERT_EQ(subgraphs.size(), 1u);
}

}  // namespace lite
}  // namespace paddle
