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

#include <memory>
#include <vector>
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/xpu_pattern_matcher_high_api.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {

namespace fusion {

class XPUEmbeddingWithEltwiseAddFuser : public FuseBase {
 public:
  explicit XPUEmbeddingWithEltwiseAddFuser(int n_embedding)
      : n_embedding_(n_embedding) {}

  void BuildPattern() override {
    auto* ids0 =
        VarNode("ids0")->assert_is_op_input("lookup_table", "Ids")->AsInput();
    auto* table0 =
        VarNode("table0")->assert_is_op_input("lookup_table", "W")->AsInput();
    auto* embedding0 = OpNode("embedding0", "lookup_table");
    auto* embedding_out0 = VarNode("embedding_out0")
                               ->assert_is_op_output("lookup_table", "Out")
                               ->assert_is_op_input("elementwise_add", "X")
                               ->AsIntermediate();

    auto* ids1 =
        VarNode("ids1")->assert_is_op_input("lookup_table", "Ids")->AsInput();
    auto* table1 =
        VarNode("table1")->assert_is_op_input("lookup_table", "W")->AsInput();
    auto* embedding1 = OpNode("embedding1", "lookup_table")->AsIntermediate();
    auto* embedding_out1 = VarNode("embedding_out1")
                               ->assert_is_op_output("lookup_table", "Out")
                               ->assert_is_op_input("elementwise_add", "Y")
                               ->AsIntermediate();

    auto* ewadd01 = OpNode("ewadd01", "elementwise_add")->AsIntermediate();
    auto* ewadd01_out = VarNode("ewadd01_out")
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->AsIntermediate();

    embedding0->LinksFrom({ids0, table0});
    embedding0->LinksTo({embedding_out0});
    embedding1->LinksFrom({ids1, table1});
    embedding1->LinksTo({embedding_out1});
    ewadd01->LinksFrom({embedding_out0, embedding_out1});
    ewadd01->LinksTo({ewadd01_out});

    auto* last_ewadd_out = ewadd01_out;
    for (int i = 2; i < n_embedding_; ++i) {
      auto ids_name = paddle::lite::string_format("ids%d", i);
      auto table_name = paddle::lite::string_format("table%d", i);
      auto embedding_name = paddle::lite::string_format("embedding%d", i);
      auto embedding_out_name =
          paddle::lite::string_format("embedding_out%d", i);

      auto* new_ids = VarNode(ids_name)
                          ->assert_is_op_input("lookup_table", "Ids")
                          ->AsInput();
      auto* new_table = VarNode(table_name)
                            ->assert_is_op_input("lookup_table", "W")
                            ->AsInput();
      auto* new_embedding =
          OpNode(embedding_name, "lookup_table")->AsIntermediate();
      auto* new_embedding_out = VarNode(embedding_out_name)
                                    ->assert_is_op_output("lookup_table", "Out")
                                    ->assert_is_op_input("elementwise_add", "Y")
                                    ->AsIntermediate();

      new_embedding->LinksFrom({new_ids, new_table});
      new_embedding->LinksTo({new_embedding_out});

      auto ewadd_name = paddle::lite::string_format("ewadd%d%d", i - 1, i);
      auto ewadd_out_name = ewadd_name + "_out";

      auto* new_ewadd = OpNode(ewadd_name, "elementwise_add")->AsIntermediate();
      auto* new_ewadd_out = VarNode(ewadd_out_name)
                                ->assert_is_op_output("elementwise_add", "Out")
                                ->AsIntermediate();

      new_ewadd->LinksFrom({last_ewadd_out, new_embedding_out});
      new_ewadd->LinksTo({new_ewadd_out});
      last_ewadd_out = new_ewadd_out;
    }
    last_ewadd_out->AsOutput();
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__embedding_with_eltwise_add");
    std::vector<std::string> ids_names;
    std::vector<std::string> table_names;
    for (int i = 0; i < n_embedding_; ++i) {
      auto ids_name = paddle::lite::string_format("ids%d", i);
      ids_names.push_back(matched.at(ids_name)->arg()->name);
      auto table_name = paddle::lite::string_format("table%d", i);
      table_names.push_back(matched.at(table_name)->arg()->name);
    }
    op_desc.SetInput("Ids", ids_names);
    op_desc.SetInput("Tables", table_names);
    auto output_name = paddle::lite::string_format(
        "ewadd%d%d_out", n_embedding_ - 2, n_embedding_ - 1);
    op_desc.SetOutput("Output", {matched.at(output_name)->arg()->name});
    op_desc.SetAttr<int>("n_embedding", n_embedding_);
    auto* embedding0_op_info = matched.at("embedding0")->stmt()->op_info();
    op_desc.SetAttr<int64_t>(
        "padding_idx", embedding0_op_info->GetAttr<int64_t>("padding_idx"));

    auto* new_stmt = matched.at("embedding0")->stmt();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, new_stmt->op()->scope());
    new_op->SetValidPlaces(new_stmt->op()->valid_places());
    auto kernels = new_op->CreateKernels(new_op->valid_places());
    new_stmt->SetOp(new_op);
    new_stmt->SetKernels(std::move(kernels));

    for (int i = 0; i < n_embedding_; ++i) {
      auto ids_name = paddle::lite::string_format("ids%d", i);
      auto table_name = paddle::lite::string_format("table%d", i);
      DirectedLink(matched.at(ids_name), matched.at("embedding0"));
      DirectedLink(matched.at(table_name), matched.at("embedding0"));
    }
    IR_OP_VAR_LINK(matched.at("embedding0"), matched.at(output_name));
  }

 private:
  int n_embedding_;
};

}  // namespace fusion

class XPUEmbeddingWithEltwiseAddFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    for (int n_embedding : {4, 3}) {
      fusion::XPUEmbeddingWithEltwiseAddFuser fuser(n_embedding);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__embedding_with_eltwise_add_fuse_pass,
                  paddle::lite::mir::XPUEmbeddingWithEltwiseAddFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__embedding_with_eltwise_add");
