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

class ElementwiseMulConstantEliminator : public FuseBase {
 public:
  void BuildPattern() override {
    auto* pre_op = OpNode("preop");    // the previous op's output need update
    auto* post_op = OpNode("postop");  // the post op's output need update
    // TODO(Superjomn) check has only one output
    auto* x =
        VarNode("x")->assert_is_op_input("elementwise_mul", "X")->AsOutput();
    auto* y = VarNode("Y")->assert_is_op_input("elementwise_mul", "Y");

    // create op nodes
    auto* mul = OpNode("mul", "elementwise_mul")
                    ->assert_is_op("elementwise_mul")
                    ->AsIntermediate();

    auto* fill_constant = OpNode("fill_constant", "fill_constant")
                              ->assert_is_op("fill_constant")
                              ->assert_op_attr<float>("value", 1.)
                              ->AsIntermediate();
    // create output node
    auto* mul_out =
        VarNode("output")->assert_is_op_output("elementwise_mul", "Out");
    // create topology.
    std::vector<PMNode*> add_inputs{x, y};
    *pre_op >> *x;
    *fill_constant >> *y;
    add_inputs >> *mul >> *mul_out;
    *mul_out >> *post_op;

    // The pre_op will be eliminated, and a new output-updated op will insert.
    mul_out->AsIntermediate();  // mul_out is pre_op's output, need to update
    y->AsIntermediate();        // need to update
  }

 private:
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto& post_op = matched.at("postop")->AsStmt();
    auto op_info = *post_op.op_info();

    auto mul_instruct = matched.at("mul")->stmt();
    auto* scope = mul_instruct->op()->scope();
    auto mul_input_x = scope->FindVar(matched.at("x")->arg()->name);
    auto mul_input_x_dims = mul_input_x->Get<lite::Tensor>().dims();
    auto mul_output = scope->FindVar(matched.at("output")->arg()->name);
    auto mul_output_dims = mul_output->Get<lite::Tensor>().dims();
    if (mul_input_x_dims != mul_output_dims) {
      nodes_.erase(nodes_.begin(), nodes_.end());
      LOG(WARNING)
          << "elementwise_mul input x not equal to output, eleminate failed";
      return;
    }

    op_info.UpdateAllInputs(matched.at("output")->AsArg().name,
                            matched.at("x")->AsArg().name);
    post_op.ResetOp(op_info, graph->valid_places());

    IR_NODE_LINK_TO(matched.at("x"), matched.at("postop"));
  }
};

}  // namespace

class ElementwiseMulConstantEliminatePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    ElementwiseMulConstantEliminator eliminator;
    eliminator(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(elementwise_mul_constant_eliminate_pass,
                  paddle::lite::mir::ElementwiseMulConstantEliminatePass)
    .BindTargets({TARGET(kAny)});
