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
/* link the previous xpu_fusion_op's OutputMax to   */
/* next xpu_fusion_op as InputMax                   */
/* For example:                                     */
/* graph[1]: sub block                              */
/*                     in_Input                     */
/*                       |                         */
/*                       |                         */
/*                    xpu_fusion_op                */
/*                       |      \                   */
/*                       |       \                  */
/*                out_Output      out_OutputMax     */
/*                       |                          */
/*                       |                          */
/*                    xpu_fusion_op                 */
/*                       |                          */
/*                       |                          */
/*                     out_Output                   */
/*                                                  */
/* After the pass is applied:                       */
/*                     in_Input                     */
/*                      |                         */
/*                      |                         */
/*                   xpu_fusion_op                */
/*                       |      \                   */
/*                       |       \                  */
/*                out_Output      out_OutputMax     */
/*                       |       /                  */
/*                       |      /                   */
/*                    xpu_fusion_op                 */
/*                       |                          */
/*                       |                          */
/*                     out_Output                   */

class XPULinkConvMaxFuser : public FuseBase {
 public:
  explicit XPULinkConvMaxFuser(bool with_branch) { with_branch_ = with_branch; }
  void BuildPattern() override {
    auto non_quant_teller = [](const Node* node) -> bool {
      auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
      return (!op_desc.HasAttr("enable_int8") ||
              !op_desc.GetAttr<bool>("enable_int8"));
    };

    auto* input =
        VarNode("input")->assert_is_op_input("__xpu__conv2d", "Input");
    auto* xpu_fusion_op =
        OpNode("xpu_fusion_op", "__xpu__conv2d")
            ->assert_node_satisfied(non_quant_teller)
            ->assert_op_attr<bool>("has_branch", with_branch_);

    PMNode* branch = nullptr;
    if (with_branch_) {
      branch = VarNode("branch")->assert_is_op_input("__xpu__conv2d", "Branch");
      *input >> *xpu_fusion_op;
      *branch >> *xpu_fusion_op;
    } else {
      *input >> *xpu_fusion_op;
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto xpu_op_instruct = matched.at("xpu_fusion_op")->stmt();
    auto op_desc = *xpu_op_instruct->mutable_op_info();
    auto xpu_op = xpu_op_instruct->op();

    // try to find input_max
    std::string max_input_name = matched.at("input")->arg()->name + "_xpu_max";
    auto* max_input_node = graph->RetrieveArgument(max_input_name);
    if (max_input_node != nullptr &&
        (!op_desc.HasAttr("has_input_max") ||
         !op_desc.GetAttr<bool>("has_input_max"))) {
      op_desc.SetInput("InputMax", {max_input_name});
      op_desc.SetAttr("has_input_max", true);
      xpu_op_instruct->ResetOp(op_desc, xpu_op->valid_places());
      DirectedLink(max_input_node, matched.at("xpu_fusion_op"));
    }
  }

 private:
  bool with_branch_;
};

class XPULinkFcMaxFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto non_quant_teller = [](const Node* node) -> bool {
      auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
      return (!op_desc.HasAttr("enable_int8") ||
              !op_desc.GetAttr<bool>("enable_int8"));
    };
    auto* input = VarNode("input")->assert_is_op_input("__xpu__fc", "Input");
    auto* xpu_fusion_op = OpNode("xpu_fusion_op", "__xpu__fc")
                              ->assert_node_satisfied(non_quant_teller);

    *input >> *xpu_fusion_op;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto xpu_op_instruct = matched.at("xpu_fusion_op")->stmt();
    auto op_desc = *xpu_op_instruct->mutable_op_info();
    auto xpu_op = xpu_op_instruct->op();

    // try to find input_max
    std::string max_input_name = matched.at("input")->arg()->name + "_xpu_max";
    auto* max_input_node = graph->RetrieveArgument(max_input_name);
    if (max_input_node != nullptr &&
        (!op_desc.HasAttr("has_input_max") ||
         !op_desc.GetAttr<bool>("has_input_max"))) {
      op_desc.SetInput("InputMax", {max_input_name});
      op_desc.SetAttr("has_input_max", true);
      xpu_op_instruct->ResetOp(op_desc, xpu_op->valid_places());
      DirectedLink(max_input_node, matched.at("xpu_fusion_op"));
    }
  }
};

}  // namespace fusion

class XPULinkMaxPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;

    for (auto with_branch : {true, false}) {
      fusion::XPULinkConvMaxFuser conv_fuser(with_branch);
      conv_fuser(graph.get());
    }
    fusion::XPULinkFcMaxFuser fc_fuser;
    fc_fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__link_previous_out_max_pass,
                  paddle::lite::mir::XPULinkMaxPass)
    .BindTargets({TARGET(kXPU)});
