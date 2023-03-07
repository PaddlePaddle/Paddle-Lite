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

#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

class Eliminator : public FuseBase {
 public:
  static bool DropoutIsTest(const Node* x) {
    if (x && x->IsStmt()) {
      auto* op_info = x->stmt()->op_info();
      if (op_info->HasAttr("is_test")) {
        auto attr_type = op_info->GetAttrType("is_test");
        if (attr_type == paddle::lite::OpDescAPI::AttrType::INT &&
            op_info->GetAttr<int>("is_test") == 1) {
          return true;
        } else if (attr_type == paddle::lite::OpDescAPI::AttrType::BOOLEAN &&
                   op_info->GetAttr<bool>("is_test")) {
          return true;
        }
      }
    }
    return false;
  }

  void BuildPattern() override {
    // the previous op's output need updat
    auto* pre_op = OpNode("preop")->assert_is_not_op_type("conditional_block");
    // TODO(Superjomn) check has only one output
    auto* x = VarNode("x")->assert_is_op_input("dropout", "X");
    auto* dropout_op = OpNode("dropout", "dropout")
                           ->assert_node_satisfied(Eliminator::DropoutIsTest)
                           ->assert_op_attr<std::string>(
                               "dropout_implementation", "upscale_in_train");
    auto* out = VarNode("out")->assert_is_op_output("dropout", "Out");
    auto* mask = VarNode("mask")->assert_is_op_output("dropout", "Mask");

    *pre_op >> *x >> *dropout_op >> *out;
    *dropout_op >> *mask;

    // The pre_op will be eliminated, and a new output-updated op will insert.
    x->AsIntermediate();  // x is pre_op's output, need to update
    dropout_op->AsIntermediate();
    mask->AsIntermediate();
  }

 private:
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto& pre_op = matched.at("preop")->AsStmt();
    auto op_info = *pre_op.op_info();

    op_info.UpdateAllOutputs(matched.at("x")->AsArg().name,
                             matched.at("out")->AsArg().name);
    pre_op.ResetOp(op_info, graph->valid_places());

    IR_NODE_LINK_TO(matched.at("preop"), matched.at("out"));
  }
};

}  // namespace

class IdentityDropoutEliminatePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    Eliminator eliminator;
    eliminator(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(identity_dropout_eliminate_pass,
                  paddle::lite::mir::IdentityDropoutEliminatePass)
    .BindTargets(
        {TARGET(kARM), TARGET(kX86), TARGET(kXPU), TARGET(kNNAdapter)});
