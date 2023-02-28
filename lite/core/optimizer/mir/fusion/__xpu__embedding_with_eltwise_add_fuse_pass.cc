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
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/xpu_pattern_matcher_high_api.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {

namespace fusion {

class XPUEmbeddingWithEltwiseAddFuser : public FuseBase {
 public:
  explicit XPUEmbeddingWithEltwiseAddFuser(
      int n_embedding,
      const std::string& op_type = "lookup_table",
      const std::string& pre_op_type = "")
      : n_embedding_(n_embedding),
        op_type_(op_type),
        pre_op_type_(pre_op_type) {}

  void BuildPattern() override {
    PMNode* x0 = nullptr;
    PMNode* preproces0 = nullptr;
    PMNode* preproces_out0 = nullptr;
    PMNode* preproces_xshape0 = nullptr;
    PMNode* x1 = nullptr;
    PMNode* preproces1 = nullptr;
    PMNode* preproces_out1 = nullptr;
    PMNode* preproces_xshape1 = nullptr;
    PMNode* embedding0 = nullptr;
    if (pre_op_type_ == "squeeze2" || pre_op_type_ == "reshape2") {
      x0 = VarNode("x0")->assert_is_op_input(pre_op_type_, "X")->AsInput();
      preproces0 = OpNode("preproces0", pre_op_type_);
      preproces_out0 = VarNode("preproces_out0")
                           ->assert_is_op_output(pre_op_type_, "Out")
                           ->assert_is_op_input(op_type_, "Ids")
                           ->AsIntermediate();
      preproces_xshape0 = VarNode("preproces_xshape0")
                              ->assert_is_op_output(pre_op_type_, "XShape")
                              ->AsIntermediate();
      x1 = VarNode("x1")->assert_is_op_input(pre_op_type_, "X")->AsInput();
      preproces1 = OpNode("preproces1", pre_op_type_)->AsIntermediate();
      preproces_out1 = VarNode("preproces_out1")
                           ->assert_is_op_output(pre_op_type_, "Out")
                           ->assert_is_op_input(op_type_, "Ids")
                           ->AsIntermediate();
      preproces_xshape1 = VarNode("preproces_xshape1")
                              ->assert_is_op_output(pre_op_type_, "XShape")
                              ->AsIntermediate();
      embedding0 = OpNode("embedding0", op_type_)->AsIntermediate();
    } else {
      x0 = VarNode("x0")->assert_is_op_input(op_type_, "Ids")->AsInput();
      x1 = VarNode("x1")->assert_is_op_input(op_type_, "Ids")->AsInput();
      embedding0 = OpNode("embedding0", op_type_);
    }
    auto* table0 =
        VarNode("table0")->assert_is_op_input(op_type_, "W")->AsInput();
    auto* embedding_out0 = VarNode("embedding_out0")
                               ->assert_is_op_output(op_type_, "Out")
                               ->assert_is_op_input("elementwise_add", "X")
                               ->AsIntermediate();
    auto* table1 =
        VarNode("table1")->assert_is_op_input(op_type_, "W")->AsInput();
    auto* embedding1 = OpNode("embedding1", op_type_)->AsIntermediate();
    auto* embedding_out1 = VarNode("embedding_out1")
                               ->assert_is_op_output(op_type_, "Out")
                               ->assert_is_op_input("elementwise_add", "Y")
                               ->AsIntermediate();

    auto* ewadd01 = OpNode("ewadd01", "elementwise_add")->AsIntermediate();
    auto* ewadd01_out = VarNode("ewadd01_out")
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->AsIntermediate();
    if (pre_op_type_ == "squeeze2" || pre_op_type_ == "reshape2") {
      preproces0->LinksFrom({x0});
      preproces0->LinksTo({preproces_out0, preproces_xshape0});
      embedding0->LinksFrom({preproces_out0, table0});
      preproces1->LinksFrom({x1});
      preproces1->LinksTo({preproces_out1, preproces_xshape1});
      embedding1->LinksFrom({preproces_out1, table1});
    } else {
      embedding0->LinksFrom({x0, table0});
      embedding1->LinksFrom({x1, table1});
    }
    embedding0->LinksTo({embedding_out0});
    embedding1->LinksTo({embedding_out1});
    ewadd01->LinksFrom({embedding_out0, embedding_out1});
    ewadd01->LinksTo({ewadd01_out});

    auto* last_ewadd_out = ewadd01_out;
    for (int i = 2; i < n_embedding_; ++i) {
      auto x_name = paddle::lite::string_format("x%d", i);
      auto preproces_name = paddle::lite::string_format("preproces%d", i);
      auto preproces_out_name =
          paddle::lite::string_format("preproces_out%d", i);
      auto preproces_xshape_name =
          paddle::lite::string_format("preproces_xshape%d", i);
      auto table_name = paddle::lite::string_format("table%d", i);
      auto embedding_name = paddle::lite::string_format("embedding%d", i);
      auto embedding_out_name =
          paddle::lite::string_format("embedding_out%d", i);

      PMNode* new_x = nullptr;
      auto* new_table =
          VarNode(table_name)->assert_is_op_input(op_type_, "W")->AsInput();
      auto* new_embedding = OpNode(embedding_name, op_type_)->AsIntermediate();
      auto* new_embedding_out = VarNode(embedding_out_name)
                                    ->assert_is_op_output(op_type_, "Out")
                                    ->assert_is_op_input("elementwise_add", "Y")
                                    ->AsIntermediate();
      if (pre_op_type_ == "squeeze2" || pre_op_type_ == "reshape2") {
        new_x =
            VarNode(x_name)->assert_is_op_input(pre_op_type_, "X")->AsInput();
        auto* new_preproces =
            OpNode(preproces_name, pre_op_type_)->AsIntermediate();
        auto* new_preproces_out = VarNode(preproces_out_name)
                                      ->assert_is_op_output(pre_op_type_, "Out")
                                      ->assert_is_op_input(op_type_, "Ids")
                                      ->AsIntermediate();
        auto* new_preproces_xshape =
            VarNode(preproces_xshape_name)
                ->assert_is_op_output(pre_op_type_, "XShape")
                ->AsIntermediate();
        new_preproces->LinksFrom({new_x});
        new_preproces->LinksTo({new_preproces_out, new_preproces_xshape});
        new_embedding->LinksFrom({new_preproces_out, new_table});
      } else {
        new_x = VarNode(x_name)->assert_is_op_input(op_type_, "Ids")->AsInput();
        new_embedding->LinksFrom({new_x, new_table});
      }
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
    std::vector<std::string> x_names;
    std::vector<std::string> table_names;
    for (int i = 0; i < n_embedding_; ++i) {
      auto x_name = paddle::lite::string_format("x%d", i);
      x_names.push_back(matched.at(x_name)->arg()->name);
      auto table_name = paddle::lite::string_format("table%d", i);
      table_names.push_back(matched.at(table_name)->arg()->name);
    }
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetInput("Ids", x_names);
    op_desc.SetInput("Tables", table_names);
    auto output_name = paddle::lite::string_format(
        "ewadd%d%d_out", n_embedding_ - 2, n_embedding_ - 1);
    op_desc.SetOutput("Output", {matched.at(output_name)->arg()->name});
    op_desc.SetAttr<int>("n_embedding", n_embedding_);
    auto* embedding0_op_info = matched.at("embedding0")->stmt()->op_info();
    op_desc.SetAttr<int64_t>(
        "padding_idx", embedding0_op_info->GetAttr<int64_t>("padding_idx"));
    auto* new_stmt = matched.at("embedding0")->stmt();
    if (pre_op_type_ == "squeeze2" || pre_op_type_ == "reshape2") {
      new_stmt = matched.at("preproces0")->stmt();
    }
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, new_stmt->op()->scope());
    new_op->SetValidPlaces(new_stmt->op()->valid_places());
    auto kernels = new_op->CreateKernels(new_op->valid_places());
    new_stmt->SetOp(new_op);
    new_stmt->SetKernels(std::move(kernels));

    for (int i = 0; i < n_embedding_; ++i) {
      auto x_name = paddle::lite::string_format("x%d", i);
      auto table_name = paddle::lite::string_format("table%d", i);
      if (pre_op_type_ == "squeeze2" || pre_op_type_ == "reshape2") {
        DirectedLink(matched.at(x_name), matched.at("preproces0"));
        DirectedLink(matched.at(table_name), matched.at("preproces0"));
      } else {
        DirectedLink(matched.at(x_name), matched.at("embedding0"));
        DirectedLink(matched.at(table_name), matched.at("embedding0"));
      }
    }
    if (pre_op_type_ == "squeeze2" || pre_op_type_ == "reshape2") {
      IR_OP_VAR_LINK(matched.at("preproces0"), matched.at(output_name));
    } else {
      IR_OP_VAR_LINK(matched.at("embedding0"), matched.at(output_name));
    }
  }

 private:
  int n_embedding_;
  std::string op_type_;
  std::string pre_op_type_;
};

}  // namespace fusion

class XPUEmbeddingWithEltwiseAddFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    std::vector<std::string> preoptypes{"squeeze2", "reshape2", ""};
    std::vector<std::string> optypes{"lookup_table", "lookup_table_v2"};
    for (auto& pre_op_type : preoptypes) {
      for (int n_embedding : {4, 3, 2}) {
        for (auto& op_type : optypes) {
          fusion::XPUEmbeddingWithEltwiseAddFuser fuser(
              n_embedding, op_type, pre_op_type);
          fuser(graph.get());
        }
      }
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
