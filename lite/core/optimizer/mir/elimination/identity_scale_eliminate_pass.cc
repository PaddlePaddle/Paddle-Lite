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

#include <cmath>
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

class Eliminator : public FuseBase {
 public:
  void BuildPattern() override {
    // the previous op's output need updat
    auto* pre_op = OpNode("preop")
                       ->assert_is_not_op_type("conditional_block")
                       ->assert_is_not_op_type("while")
                       ->assert_is_not_op_type("scale");
    // TODO(Superjomn) check has only one output
    auto* x = VarNode("x")->assert_is_op_input("scale", "X");
    auto* scale_op =
        OpNode("scale", "scale")
            ->assert_node_satisfied([](const Node* node) -> bool {
              auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
              bool is_scale_1_bias_0 =
                  op_desc.HasAttr("scale") && op_desc.HasAttr("bias") &&
                  fabs(op_desc.GetAttr<float>("scale") - 1.0f) <= 1e-5f &&
                  fabs(op_desc.GetAttr<float>("bias")) <= 1e-5f;
              bool with_act = (op_desc.HasAttr("with_act") &&
                               op_desc.GetAttr<bool>("with_act")) ||
                              op_desc.HasAttr("fuse_relu");
              return is_scale_1_bias_0 && !with_act;
            });
    auto* out = VarNode("out")->assert_is_op_output("scale", "Out");

    *pre_op >> *x >> *scale_op >> *out;

    // The pre_op will be eliminated, and a new output-updated op will insert.
    x->AsIntermediate();  // x is pre_op's output, need to update
  }

 private:
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto& pre_op = matched.at("preop")->AsStmt();
    auto op_info = *pre_op.op_info();

    op_info.UpdateAllOutputs(matched.at("x")->AsArg().name,
                             matched.at("out")->AsArg().name);
    pre_op.ResetOp(op_info, graph->valid_places());

    GraphSafeRemoveNodes(graph, {matched.at("scale")});

    IR_NODE_LINK_TO(matched.at("preop"), matched.at("out"));
  }
};

}  // namespace

class IdentityScaleEliminatePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    Eliminator eliminator;
    eliminator(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(identity_scale_eliminate_pass,
                  paddle::lite::mir::IdentityScaleEliminatePass)
    .BindTargets({TARGET(kAny)});
