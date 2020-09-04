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
#include <string>
#include "lite/backends/xpu/math.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {
// Special fuse pass for the subgraph block in vis clarity model
// block desc:
//  [["reduce_mean",
//  ["concat"],
//  ["elementwise_sub",
//      ["square", ["reduce_mean", ["sqrt"]]],
//      ["abs", ["pow", ["elementwise_mul", ["reduce_mean", ["abs",
//      ["pow"]]]]]],
//      ["sign"],
//      ["abs", ["pow", ["reduce_mean", ["abs", ["pow"]]]]]]]]

class XPUSfaHeadMomentFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* reduce_mean_input = VarNode("reduce_mean_input")
                                  ->assert_is_op_output("reshape2", "Out")
                                  ->assert_is_op_input("reduce_mean", "X")
                                  ->assert_is_op_input("elementwise_sub", "X")
                                  ->AsInput();
    auto* reduce_mean = OpNode("reduce_mean", "reduce_mean")->AsIntermediate();

    auto* reduce_mean_out = VarNode("reduce_mean_out")
                                ->assert_is_op_output("reduce_mean", "Out")
                                ->assert_is_op_nth_input("concat", "X", 0)
                                ->assert_is_op_input("elementwise_sub", "Y")
                                ->AsIntermediate();

    auto* elementwise_sub =
        OpNode("elementwise_sub", "elementwise_sub")->AsIntermediate();
    auto* elementwise_sub_out =
        VarNode("elementwise_sub_out")
            ->assert_is_op_output("elementwise_sub", "Out")
            ->assert_is_op_input("square", "X")
            ->assert_is_op_input("abs", "X")
            ->assert_is_op_input("sign", "X")
            ->AsIntermediate();

    auto* square = OpNode("square", "square")->AsIntermediate();

    auto* square_out = VarNode("square_out")
                           ->assert_is_op_output("square", "Out")
                           ->assert_is_op_input("reduce_mean", "X")
                           ->AsIntermediate();
    auto* reduce_mean_es =
        OpNode("es_reduce_mean", "reduce_mean")->AsIntermediate();
    auto* reduce_mean_out_es = VarNode("reduce_mean_out_es")
                                   ->assert_is_op_output("reduce_mean", "Out")
                                   ->assert_is_op_input("sqrt", "X")
                                   ->AsIntermediate();
    auto* sqrt = OpNode("sqrt", "sqrt")->AsIntermediate();
    auto* sqrt_out = VarNode("sqrt_out")
                         ->assert_is_op_output("sqrt", "Out")
                         ->assert_is_op_nth_input("concat", "X", 1)
                         ->AsIntermediate();
    auto* concat = OpNode("concat", "concat")->AsIntermediate();
    auto* out =
        VarNode("out")->assert_is_op_output("concat", "Out")->AsOutput();

    auto* abs_e2 = OpNode("e2_abs", "abs")->AsIntermediate();
    auto* abs_e2_out = VarNode("abs_e2_out")
                           ->assert_is_op_input("pow", "X")
                           ->assert_is_op_output("abs", "Out")
                           ->AsIntermediate();

    auto* pow_e2 = OpNode("e2_pow", "pow")->AsIntermediate();
    auto* pow_e2_out = VarNode("pow_e2_out")
                           ->assert_is_op_input("elementwise_mul", "X")
                           ->assert_is_op_output("pow", "Out")
                           ->AsIntermediate();

    auto* sign_e3 = OpNode("e3_sign", "sign")->AsIntermediate();
    auto* sign_e3_out = VarNode("sign_e3_out")
                            ->assert_is_op_input("elementwise_mul", "Y")
                            ->assert_is_op_output("sign", "Out")
                            ->AsIntermediate();

    auto* elementwise_mul_top =
        OpNode("elementwise_mul_top", "elementwise_mul")->AsIntermediate();
    auto* elementwise_mul_top_out =
        VarNode("elementwise_mul_top_out")
            ->assert_is_op_input("reduce_mean", "X")
            ->assert_is_op_output("elementwise_mul", "Out")
            ->AsIntermediate();
    auto* reduce_mean_e2 =
        OpNode("reduce_mean_e2", "reduce_mean")->AsIntermediate();
    auto* reduce_mean_e2_out = VarNode("reduce_mean_e2_out")
                                   ->assert_is_op_input("abs", "X")
                                   ->assert_is_op_input("sign", "X")
                                   ->assert_is_op_output("reduce_mean", "Out")
                                   ->AsIntermediate();
    auto* abs_e2_2 = OpNode("abs_e2_2", "abs")->AsIntermediate();
    auto* abs_e2_2_out = VarNode("abs_e2_2_out")
                             ->assert_is_op_input("pow", "X")
                             ->assert_is_op_output("abs", "Out")
                             ->AsIntermediate();
    auto* pow_e2_2 = OpNode("pow_e2_2", "pow")->AsIntermediate();
    auto* pow_e2_2_out = VarNode("pow_e2_2_out")
                             ->assert_is_op_nth_input("elementwise_mul", "X", 0)
                             ->assert_is_op_output("pow", "Out")
                             ->AsIntermediate();
    auto* sign_e3_2 = OpNode("sign_e3_2", "sign")->AsIntermediate();
    auto* sign_e3_2_out = VarNode("sign_e3_2_out")
                              ->assert_is_op_input("elementwise_mul", "Y")
                              ->assert_is_op_output("sign", "Out")
                              ->AsIntermediate();
    auto* elementwise_mul_bottom =
        OpNode("elementwise_mul_bottom", "elementwise_mul")->AsIntermediate();
    auto* elementwise_mul_bottom_out =
        VarNode("elementwise_mul_bottom_out")
            ->assert_is_op_output("elementwise_mul", "Out")
            ->assert_is_op_nth_input("concat", "X", 2)
            ->AsIntermediate();

    // e4
    auto* abs_e_4 = OpNode("abs_e_4", "abs")->AsIntermediate();
    auto* abs_e_4_out = VarNode("abs_e_4_out")
                            ->assert_is_op_output("abs", "Out")
                            ->assert_is_op_input("pow", "X")
                            ->AsIntermediate();
    auto* pow_e_4 = OpNode("pow_e_4", "pow")->AsIntermediate();
    auto* pow_e_4_out = VarNode("pow_e_4_out")
                            ->assert_is_op_output("pow", "Out")
                            ->assert_is_op_input("reduce_mean", "X")
                            ->AsIntermediate();
    auto* reduce_mean_4 = OpNode("reduce_mean_4")->AsIntermediate();
    auto* reduce_mean_4_out = VarNode("reduce_mean_4_out")
                                  ->assert_is_op_output("reduce_mean", "Out")
                                  ->assert_is_op_input("abs", "X")
                                  ->AsIntermediate();

    auto* abs_e_4_2 = OpNode("abs_e_4_2", "abs")->AsIntermediate();
    auto* abs_e_4_2_out = VarNode("abs_e_4_2_out")
                              ->assert_is_op_output("abs", "Out")
                              ->assert_is_op_input("pow", "X")
                              ->AsIntermediate();

    auto* pow_e_4_2 = OpNode("pow_e_4_2", "pow")->AsIntermediate();
    auto* pow_e_4_2_out = VarNode("pow_e_4_2_out")
                              ->assert_is_op_output("pow", "Out")
                              ->assert_is_op_nth_input("concat", "X", 3)
                              ->AsIntermediate();

    std::vector<PMNode*> elementwise_sub_inputs{reduce_mean_input,
                                                reduce_mean_out};

    *reduce_mean_input >> *reduce_mean >> *reduce_mean_out;
    elementwise_sub_inputs >> *elementwise_sub >> *elementwise_sub_out;
    *elementwise_sub_out >> *square >> *square_out;
    *square_out >> *reduce_mean_es >> *reduce_mean_out_es;
    *reduce_mean_out_es >> *sqrt >> *sqrt_out;

    *elementwise_sub_out >> *sign_e3 >> *sign_e3_out;

    std::vector<PMNode*> elementwise_mul_top_inputs{pow_e2_out, sign_e3_out};
    *elementwise_sub_out >> *abs_e2 >> *abs_e2_out;
    *abs_e2_out >> *pow_e2 >> *pow_e2_out;
    elementwise_mul_top_inputs >> *elementwise_mul_top >>
        *elementwise_mul_top_out;

    *elementwise_mul_top_out >> *reduce_mean_e2 >> *reduce_mean_e2_out;
    *reduce_mean_e2_out >> *abs_e2_2 >> *abs_e2_2_out;
    *abs_e2_2_out >> *pow_e2_2 >> *pow_e2_2_out;

    *reduce_mean_e2_out >> *sign_e3_2 >> *sign_e3_2_out;

    std::vector<PMNode*> elementwise_mul_bottom_inputs{pow_e2_2_out,
                                                       sign_e3_2_out};
    elementwise_mul_bottom_inputs >> *elementwise_mul_bottom >>
        *elementwise_mul_bottom_out;

    *elementwise_sub_out >> *abs_e_4 >> *abs_e_4_out;
    *abs_e_4_out >> *pow_e_4 >> *pow_e_4_out;
    *pow_e_4_out >> *reduce_mean_4 >> *reduce_mean_4_out;
    *reduce_mean_4_out >> *abs_e_4_2 >> *abs_e_4_2_out;
    *abs_e_4_2_out >> *pow_e_4_2 >> *pow_e_4_2_out;

    std::vector<PMNode*> concat_inputs{
        reduce_mean_out, sqrt_out, elementwise_mul_bottom_out, pow_e_4_2_out};
    concat_inputs >> *concat >> *out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto reduce_mean = matched.at("reduce_mean")->stmt()->op();
    auto* scope = reduce_mean->scope();
    auto op_desc = GenOpDesc(matched);
    auto vis_op = LiteOpRegistry::Global().Create("__xpu__sfa_head");
    auto& valid_places = reduce_mean->valid_places();
    vis_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(vis_op, valid_places);

    IR_NODE_LINK_TO(matched.at("reduce_mean_input"), new_op_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("out"));
  }

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) {
    cpp::OpDesc op_desc = *matched.at("reduce_mean")->stmt()->op_info();
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("__xpu__sfa_head");
    op_desc.SetInput("Input", {matched.at("reduce_mean_input")->arg()->name});
    op_desc.SetOutput("Output", {matched.at("out")->arg()->name});
    op_desc.SetAttr("op_type", std::string("moment"));
    return op_desc;
  }
};

}  // namespace fusion

class XPUSfaHeadMomentFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) {
      return;
    }

    fusion::XPUSfaHeadMomentFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__sfa_head_moment_fuse_pass,
                  paddle::lite::mir::XPUSfaHeadMomentFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("reduce_mean");
