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
/* link the previous __xpu__conv2d's OutputMax to   */
/* next __xpu__conv2d as InputMax                   */
/* For example:                                     */
/* graph[1]: sub block                              */
/*                     in_Input                     */
/*        in_Filter      |     in_FilterMax         */
/*                  \    |    /                     */
/*                   \   |   /                      */
/*     in_Bias ------- __xpu__conv2d                */
/*                       |      \                   */
/*                       |       \                  */
/*                out_Output      out_OutputMax     */
/*                       |                          */
/*                       |                          */
/*                    __xpu__conv2d                 */
/*                       |                          */
/*                       |                          */
/*                     out_Output                   */
/*                                                  */
/* After the pass is applied:                       */
/*                     in_Input                     */
/*        in_Filter      |     in_FilterMax         */
/*                  \    |    /                     */
/*                   \   |   /                      */
/*     in_Bias ------- __xpu__conv2d                */
/*                       |      \                   */
/*                       |       \                  */
/*                out_Output      out_OutputMax     */
/*                       |       /                  */
/*                       |      /                   */
/*                    __xpu__conv2d                 */
/*                       |                          */
/*                       |                          */
/*                     out_Output                   */

class XPUConv2dLinkFuser : public FuseBase {
 public:
  explicit XPUConv2dLinkFuser(bool with_branch) : _with_branch(with_branch) {}

  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__conv2d", "Input")
                      ->AsInput();
    auto* filter = VarNode("filter")
                       ->assert_is_op_input("__xpu__conv2d", "Filter")
                       ->AsInput();
    auto* filter_max = VarNode("filter_max")
                           ->assert_is_op_input("__xpu__conv2d", "FilterMax")
                           ->AsInput();
    auto* bias =
        VarNode("bias")->assert_is_op_input("__xpu__conv2d", "Bias")->AsInput();
    auto* xpu_conv = OpNode("xpu_conv", "__xpu__conv2d");
    auto* xpu_conv_out = VarNode("xpu_conv_out")
                             ->assert_is_op_output("__xpu__conv2d", "Output")
                             ->AsOutput();
    auto* xpu_conv_out_max =
        VarNode("xpu_conv_out_max")
            ->assert_is_op_output("__xpu__conv2d", "OutputMax")
            ->AsOutput();

    *input >> *xpu_conv >> *xpu_conv_out;
    *filter >> *xpu_conv;
    *filter_max >> *xpu_conv;
    *bias >> *xpu_conv;
    *xpu_conv >> *xpu_conv_out_max;

    if (_with_branch) {
      auto* branch = VarNode("branch")
                         ->assert_is_op_input("__xpu__conv2d", "Branch")
                         ->AsInput();
      *branch >> *xpu_conv;
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto conv_instruct = matched.at("xpu_conv")->stmt();
    auto op_desc = *conv_instruct->mutable_op_info();
    auto conv_old = conv_instruct->op();

    // try to find input_max
    std::string max_input_name = matched.at("input")->arg()->name + "_max";
    auto* max_input_node = graph->RetrieveArgument(max_input_name);
    if (max_input_node != nullptr &&
        (!op_desc.HasAttr("has_input_max") ||
         !op_desc.GetAttr<bool>("has_input_max"))) {
      op_desc.SetInput("InputMax", {max_input_name});
      op_desc.SetAttr("has_input_max", true);
      conv_instruct->ResetOp(op_desc, conv_old->valid_places());
      DirectedLink(max_input_node, matched.at("xpu_conv"));
    }
  }

 private:
  bool _with_branch;
};

}  // namespace fusion

class XPUConv2dLinkPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;

    fusion::XPUConv2dLinkFuser fuser1(true);
    fuser1(graph.get());

    // TODO(sunsetlh): need fix bug in no branch case
    fusion::XPUConv2dLinkFuser fuser2(false);
    fuser2(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__conv2d_link_previous_out_max_pass,
                  paddle::lite::mir::XPUConv2dLinkPass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__conv2d");
