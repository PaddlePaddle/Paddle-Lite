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
//      ["concat"],
//      ["elementwise_sub",
//          ["square", ["reduce_sum", ["scale", ["sqrt"]]]]]]]

class XPUSfaHeadMeanstdFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* reduce_mean_input = VarNode("reduce_mean_input")
                                  ->assert_is_op_output("reshape2", "Out")
                                  ->assert_is_op_input("reduce_mean", "X")
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
            ->AsIntermediate();
    auto* square = OpNode("square", "square")->AsIntermediate();
    auto* square_out = VarNode("square_out")
                           ->assert_is_op_output("square", "Out")
                           ->assert_is_op_input("reduce_sum", "X")
                           ->AsIntermediate();
    auto* reduce_sum = OpNode("reduce_sum", "reduce_sum")->AsIntermediate();
    auto* reduce_sum_out = VarNode("reduce_sum_out")
                               ->assert_is_op_output("reduce_sum", "Out")
                               ->assert_is_op_input("elementwise_div", "X")
                               ->AsIntermediate();
    auto* fill_constant =
        OpNode("fill_constant", "fill_constant")->AsIntermediate();
    auto* fill_constant_out = VarNode("fill_constant_out")
                                  ->assert_is_op_output("fill_constant", "Out")
                                  ->AsIntermediate();
    auto* elementwise_div =
        OpNode("elementwise_div", "elementwise_div")->AsIntermediate();
    auto* elementwise_div_out =
        VarNode("elementwise_div_out")
            ->assert_is_op_output("elementwise_div", "Out")
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

    std::vector<PMNode*> elementwise_sub_inputs{reduce_mean_out,
                                                reduce_mean_input};
    std::vector<PMNode*> elementwise_div_inputs{reduce_sum_out,
                                                fill_constant_out};
    std::vector<PMNode*> concat_inputs{reduce_mean_out, sqrt_out};
    *reduce_mean_input >> *reduce_mean >> *reduce_mean_out;
    elementwise_sub_inputs >> *elementwise_sub >> *elementwise_sub_out;
    *elementwise_sub_out >> *square >> *square_out;
    *square_out >> *reduce_sum >> *reduce_sum_out;
    *fill_constant >> *fill_constant_out;
    elementwise_div_inputs >> *elementwise_div >> *elementwise_div_out;
    *elementwise_div_out >> *sqrt >> *sqrt_out;
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
    op_desc.SetAttr("op_type", std::string("meanstd"));
    return op_desc;
  }
};

}  // namespace fusion

class XPUSfaHeadMeanstdFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) {
      return;
    }

    fusion::XPUSfaHeadMeanstdFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__sfa_head_meanstd_fuse_pass,
                  paddle::lite::mir::XPUSfaHeadMeanstdFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("reduce_mean");
