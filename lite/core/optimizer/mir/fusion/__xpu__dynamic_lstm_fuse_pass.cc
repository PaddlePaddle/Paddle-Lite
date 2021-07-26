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
/* fuse __xpu__fc and lstm as xpu_block */
/*                                      */
/*               in_Input               */
/*                  |                   */
/*                  |                   */
/*              __xpu__fc               */
/*                  |                   */
/*                  |                   */
/*                 lstm                 */
/*                  |                   */
/*                  |                   */
/*              out_Output              */
/*--------------------------------------*/

class XPUDynamicLstmFuser : public FuseBase {
 public:
  explicit XPUDynamicLstmFuser(bool with_h0) { with_h0_ = with_h0; }
  void BuildPattern() override {
    // __xpu__fc
    auto* input =
        VarNode("input")->assert_is_op_input("__xpu__fc", "Input")->AsInput();
    auto* weight_0 = VarNode("weight_0")
                         ->assert_is_op_input("__xpu__fc", "Filter")
                         ->assert_is_persistable_var()
                         ->AsInput();
    auto* bias_0 = VarNode("bias_0")
                       ->assert_is_op_input("__xpu__fc", "Bias")
                       ->assert_is_persistable_var()
                       ->AsInput();
    auto* xpu_fc = OpNode("xpu_fc", "__xpu__fc")
                       ->assert_op_attr<int>("act_type", 0)
                       ->assert_op_attr<bool>("has_bias", true)
                       ->assert_op_attr<int>("in_num_col_dims", 1)
                       ->AsIntermediate();
    auto* fc_out = VarNode("fc_out")
                       ->assert_is_op_output("__xpu__fc", "Output")
                       ->AsIntermediate();
    auto* fc_out_max = VarNode("fc_out_max")
                           ->assert_is_op_output("__xpu__fc", "OutputMax")
                           ->AsIntermediate();

    // lstm
    fc_out->assert_is_op_input("lstm", "Input");
    auto* weight_1 = VarNode("weight_1")
                         ->assert_is_op_input("lstm", "Weight")
                         ->assert_is_persistable_var()
                         ->AsInput();
    auto* bias_1 = VarNode("bias_1")
                       ->assert_is_op_input("lstm", "Bias")
                       ->assert_is_persistable_var()
                       ->AsInput();
    PMNode* c0 = nullptr;
    PMNode* h0 = nullptr;
    if (with_h0_) {
      c0 = VarNode("c0")
               ->assert_is_op_input("lstm", "C0")
               ->assert_is_persistable_var()
               ->AsInput();
      h0 = VarNode("h0")
               ->assert_is_op_input("lstm", "H0")
               ->assert_is_persistable_var()
               ->AsInput();
    }
    auto* lstm =
        OpNode("lstm", "lstm")
            ->assert_op_attr<bool>("use_peepholes", false)
            ->assert_op_attr<std::string>("gate_activation", "sigmoid")
            ->assert_op_attr<std::string>("cell_activation", "tanh")
            ->assert_op_attr<std::string>("candidate_activation", "tanh")
            ->AsIntermediate();
    auto* hidden =
        VarNode("hidden")->assert_is_op_output("lstm", "Hidden")->AsOutput();
    auto* cell =
        VarNode("cell")->assert_is_op_output("lstm", "Cell")->AsIntermediate();
    auto* batchGate = VarNode("batchGate")
                          ->assert_is_op_output("lstm", "BatchGate")
                          ->AsIntermediate();
    auto* batchCellPreAct = VarNode("batchCellPreAct")
                                ->assert_is_op_output("lstm", "BatchCellPreAct")
                                ->AsIntermediate();

    *input >> *xpu_fc >> *fc_out >> *lstm >> *batchCellPreAct;
    *weight_0 >> *xpu_fc >> *fc_out_max;
    *bias_0 >> *xpu_fc;
    *weight_1 >> *lstm >> *batchGate;
    *bias_1 >> *lstm;
    *lstm >> *hidden;
    *lstm >> *cell;
    if (with_h0_) {
      *h0 >> *lstm;
      *c0 >> *lstm;
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    auto xpu_fc = matched.at("xpu_fc")->stmt()->op();
    auto* scope = xpu_fc->scope();
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("__xpu__dynamic_lstm_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetInput("Weight_0", {matched.at("weight_0")->arg()->name});
    op_desc.SetInput("Bias_0", {matched.at("bias_0")->arg()->name});
    op_desc.SetInput("Weight_1", {matched.at("weight_1")->arg()->name});
    op_desc.SetInput("Bias_1", {matched.at("bias_1")->arg()->name});
    if (with_h0_) {
      op_desc.SetInput("C0", {matched.at("c0")->arg()->name});
      op_desc.SetInput("H0", {matched.at("h0")->arg()->name});
    }
    op_desc.SetOutput("Hidden", {matched.at("hidden")->arg()->name});
    op_desc.SetAttr<bool>("has_h0", with_h0_);
    op_desc.SetAttr<bool>(
        "is_reverse",
        matched.at("lstm")->stmt()->op_info()->GetAttr<bool>("is_reverse"));
    auto& valid_places = xpu_fc->valid_places();
    auto block_op = LiteOpRegistry::Global().Create(op_desc.Type());
    block_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(block_op, valid_places);
    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(matched.at("weight_0"), new_op_node);
    IR_NODE_LINK_TO(matched.at("bias_0"), new_op_node);
    IR_NODE_LINK_TO(matched.at("weight_1"), new_op_node);
    IR_NODE_LINK_TO(matched.at("bias_1"), new_op_node);
    if (with_h0_) {
      IR_NODE_LINK_TO(matched.at("c0"), new_op_node);
      IR_NODE_LINK_TO(matched.at("h0"), new_op_node);
    }
    IR_NODE_LINK_TO(new_op_node, matched.at("hidden"));
  }

 private:
  bool with_h0_;
};

}  // namespace fusion

class XPUDynamicLstmFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto with_h0_ : {true, false}) {
      fusion::XPUDynamicLstmFuser fuser(with_h0_);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__dynamic_lstm_fuse_pass,
                  paddle::lite::mir::XPUDynamicLstmFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__dynamic_lstm_fuse_op");
