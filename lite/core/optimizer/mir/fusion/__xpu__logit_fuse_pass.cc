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
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

/* fuse logit block: y=log(x/(1-x)),inverse function of sigmoid */
/* For example:                                                 */
/* graph[0]: sub block                                          */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                     clip----                                 */
/*                       |     \                                */
/*                       |      \                               */
/*                       |    fill_any_like                     */
/*                       |      /                               */
/*                       |     /                                */
/*                  elementwise_div                             */
/*                       |                                      */
/*                       |                                      */
/*                      scale                                   */
/*                       |                                      */
/*                       |                                      */
/*                      clip                                    */
/*                       |                                      */
/*                       |                                      */
/*                      log                                     */
/*                       |                                      */
/*                       |                                      */
/*                     scale                                    */
/*                       |                                      */
/*                       |                                      */
/*                     out_Out                                  */
/*                                                              */
/* After the pass is applied:                                   */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                  __xpu__logit                                */
/*                       |                                      */
/*                       |                                      */
/*                 out_Output                                   */
/*                                                              */

class XPULogitFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input = VarNode("input")->assert_is_op_input("clip", "X")->AsInput();
    auto* clip1 = OpNode("clip1", "clip")->AsIntermediate();
    auto* clip1_out = VarNode("clip1_out")
                          ->assert_is_op_output("clip", "Out")
                          ->assert_is_op_input("fill_any_like", "X")
                          ->assert_is_op_input("elementwise_div", "Y")
                          ->AsIntermediate();
    auto* fill_any_like =
        OpNode("fill_any_like", "fill_any_like")->AsIntermediate();
    auto* fill_any_like_out = VarNode("fill_any_like_out")
                                  ->assert_is_op_output("fill_any_like", "Out")
                                  ->assert_is_op_input("elementwise_div", "X")
                                  ->AsIntermediate();
    auto* elementwise_div = OpNode("elementwise_div", "elementwise_div")
                                ->assert_op_attr<int>("axis", -1)
                                ->AsIntermediate();
    auto* elementwise_div_out =
        VarNode("elementwise_div_out")
            ->assert_is_op_output("elementwise_div", "Out")
            ->assert_is_op_input("scale", "X")
            ->AsIntermediate();
    auto* scale1 = OpNode("scale1", "scale")
                       ->assert_op_attr<float>("scale", 1.)
                       ->AsIntermediate();
    auto* scale1_out = VarNode("scale1_out")
                           ->assert_is_op_output("scale", "Out")
                           ->assert_is_op_input("clip", "X")
                           ->AsIntermediate();
    auto* clip2 = OpNode("clip2", "clip")->AsIntermediate();
    auto* clip2_out = VarNode("clip2_out")
                          ->assert_is_op_output("clip", "Out")
                          ->assert_is_op_input("log", "X")
                          ->AsIntermediate();
    auto* log = OpNode("log", "log")->AsIntermediate();
    auto* log_out = VarNode("log_out")
                        ->assert_is_op_output("log", "Out")
                        ->assert_is_op_input("scale", "X")
                        ->AsIntermediate();
    auto* scale2 = OpNode("scale2", "scale")
                       ->assert_op_attr<float>("scale", -1.)
                       ->AsIntermediate();

    auto* out = VarNode("out")->assert_is_op_output("scale", "Out")->AsOutput();

    *input >> *clip1 >> *clip1_out;
    *clip1_out >> *fill_any_like >> *fill_any_like_out >> *elementwise_div;
    *clip1_out >> *elementwise_div;
    *elementwise_div >> *elementwise_div_out >> *scale1 >> *scale1_out;
    *scale1_out >> *clip2 >> *clip2_out >> *log >> *log_out >> *scale2 >> *out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__logit");
    op_desc.SetInput("X", {matched.at("input")->arg()->name});
    op_desc.SetOutput("Out", {matched.at("out")->arg()->name});
    float min = matched.at("clip1")->stmt()->op_info()->GetAttr<float>("min");
    op_desc.SetAttr<float>("eps", min);

    auto log = matched.at("log")->stmt()->op();
    auto* scope = log->scope();
    auto& valid_places = log->valid_places();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(new_op, valid_places);
    CHECK(new_op_node != nullptr) << " GraphCreateInstructNode failed";

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("out"));
  }
};

}  // namespace fusion

class XPULogitFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPULogitFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__logit_fuse_pass, paddle::lite::mir::XPULogitFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__logit");
