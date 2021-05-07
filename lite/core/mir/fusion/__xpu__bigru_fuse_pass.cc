// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
/* Bidirectional GRU                */
/*              in_Input            */
/*              /      \            */
/*             |        |           */
/*           FW_MUL   BW_MUL        */
/*             |        |           */
/*           FW_GRU   BW_GRU        */
/*             |        |           */
/*        FW_Output   BW_Output     */

class XPUBiGRUFuser : public FuseBase {
 public:
  explicit XPUBiGRUFuser(bool with_mul_bias) { with_mul_bias_ = with_mul_bias; }

  void BuildPattern() override {
    auto* input = VarNode("input")->assert_is_op_input("mul", "X")->AsInput();
    auto* fw_mul_w = VarNode("fw_mul_w")
                         ->assert_is_op_input("mul", "Y")
                         ->assert_is_persistable_var()
                         ->AsInput();
    auto* fw_mul_out = VarNode("fw_mul_out")
                           ->assert_is_op_output("mul", "Out")
                           ->AsIntermediate();
    PMNode* fw_mul_b = nullptr;
    PMNode* fw_mul_add_out = nullptr;
    if (with_mul_bias_) {
      fw_mul_out->assert_is_op_input("elementwise_add", "X");
      fw_mul_b = VarNode("fw_mul_b")
                     ->assert_is_op_input("elementwise_add", "Y")
                     ->assert_is_persistable_var()
                     ->AsInput();
      fw_mul_add_out = VarNode("fw_mul_add_out")
                           ->assert_is_op_output("elementwise_add", "Out")
                           ->AsIntermediate();
    } else {
      fw_mul_out->assert_is_op_input("gru", "Input");
    }
    auto* fw_gru_w = VarNode("fw_gru_w")
                         ->assert_is_op_input("gru", "Weight")
                         ->assert_is_persistable_var()
                         ->AsInput();
    auto* fw_gru_b = VarNode("fw_gru_b")
                         ->assert_is_op_input("gru", "Bias")
                         ->assert_is_persistable_var()
                         ->AsInput();
    auto* fw_output =
        VarNode("fw_output")->assert_is_op_output("gru", "Hidden")->AsOutput();
    auto* fw_gru_batch_gate = VarNode("fw_gru_batch_gate")
                                  ->assert_is_op_output("gru", "BatchGate")
                                  ->AsOutput();
    auto* fw_gru_batch_hidden = VarNode("fw_gru_batch_hidden")
                                    ->assert_is_op_output("gru", "BatchHidden")
                                    ->AsOutput();
    auto* fw_gru_batch_reset_hidden_prev =
        VarNode("fw_gru_batch_reset_hidden_prev")
            ->assert_is_op_output("gru", "BatchResetHiddenPrev")
            ->AsOutput();
    auto* bw_mul_w = VarNode("bw_mul_w")
                         ->assert_is_op_input("mul", "Y")
                         ->assert_is_persistable_var()
                         ->AsInput();
    auto* bw_mul_out = VarNode("bw_mul_out")
                           ->assert_is_op_output("mul", "Out")
                           ->AsIntermediate();
    PMNode* bw_mul_b = nullptr;
    PMNode* bw_mul_add_out = nullptr;
    if (with_mul_bias_) {
      bw_mul_out->assert_is_op_input("elementwise_add", "X");
      bw_mul_b = VarNode("bw_mul_b")
                     ->assert_is_op_input("elementwise_add", "Y")
                     ->assert_is_persistable_var()
                     ->AsInput();
      bw_mul_add_out = VarNode("bw_mul_add_out")
                           ->assert_is_op_output("elementwise_add", "Out")
                           ->AsIntermediate();
    } else {
      bw_mul_out->assert_is_op_input("gru", "Input");
    }
    auto* bw_gru_w = VarNode("bw_gru_w")
                         ->assert_is_op_input("gru", "Weight")
                         ->assert_is_persistable_var()
                         ->AsInput();
    auto* bw_gru_b = VarNode("bw_gru_b")
                         ->assert_is_op_input("gru", "Bias")
                         ->assert_is_persistable_var()
                         ->AsInput();
    auto* bw_output =
        VarNode("bw_output")->assert_is_op_output("gru", "Hidden")->AsOutput();
    auto* bw_gru_batch_gate = VarNode("bw_gru_batch_gate")
                                  ->assert_is_op_output("gru", "BatchGate")
                                  ->AsOutput();
    auto* bw_gru_batch_hidden = VarNode("bw_gru_batch_hidden")
                                    ->assert_is_op_output("gru", "BatchHidden")
                                    ->AsOutput();
    auto* bw_gru_batch_reset_hidden_prev =
        VarNode("bw_gru_batch_reset_hidden_prev")
            ->assert_is_op_output("gru", "BatchResetHiddenPrev")
            ->AsOutput();
    //
    auto* fw_mul = OpNode("fw_mul", "mul")->AsIntermediate();
    PMNode* fw_mul_add = nullptr;
    if (with_mul_bias_) {
      fw_mul_add = OpNode("fw_mul_add", "elementwise_add")->AsIntermediate();
    }
    auto* fw_gru = OpNode("fw_gru", "gru")
                       ->assert_op_attr<bool>("is_reverse", false)
                       ->AsIntermediate();
    auto* bw_mul = OpNode("bw_mul", "mul")->AsIntermediate();
    PMNode* bw_mul_add = nullptr;
    if (with_mul_bias_) {
      bw_mul_add = OpNode("bw_mul_add", "elementwise_add")->AsIntermediate();
    }
    auto* bw_gru = OpNode("bw_gru", "gru")
                       ->assert_op_attr<bool>("is_reverse", true)
                       ->AsIntermediate();
    // Pass
    *input >> *fw_mul >> *fw_mul_out;
    *fw_mul_w >> *fw_mul;
    if (with_mul_bias_) {
      *fw_mul_out >> *fw_mul_add;
      *fw_mul_b >> *fw_mul_add;
      *fw_mul_add >> *fw_mul_add_out;
      *fw_mul_add_out >> *fw_gru;
    } else {
      *fw_mul_out >> *fw_gru;
    }
    *fw_gru_w >> *fw_gru;
    *fw_gru_b >> *fw_gru;
    *fw_gru >> *fw_output;
    *fw_gru >> *fw_gru_batch_gate;
    *fw_gru >> *fw_gru_batch_hidden;
    *fw_gru >> *fw_gru_batch_reset_hidden_prev;
    *input >> *bw_mul >> *bw_mul_out;
    *bw_mul_w >> *bw_mul;
    if (with_mul_bias_) {
      *bw_mul_out >> *bw_mul_add;
      *bw_mul_b >> *bw_mul_add;
      *bw_mul_add >> *bw_mul_add_out;
      *bw_mul_add_out >> *bw_gru;
    } else {
      *bw_mul_out >> *bw_gru;
    }
    *bw_gru_w >> *bw_gru;
    *bw_gru_b >> *bw_gru;
    *bw_gru >> *bw_output;
    *bw_gru >> *bw_gru_batch_gate;
    *bw_gru >> *bw_gru_batch_hidden;
    *bw_gru >> *bw_gru_batch_reset_hidden_prev;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__bigru");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetInput("ForwardMulWeight", {matched.at("fw_mul_w")->arg()->name});
    if (with_mul_bias_) {
      op_desc.SetInput("ForwardMulBias", {matched.at("fw_mul_b")->arg()->name});
    }
    op_desc.SetInput("ForwardGRUWeight", {matched.at("fw_gru_w")->arg()->name});
    op_desc.SetInput("ForwardGRUBias", {matched.at("fw_gru_b")->arg()->name});
    op_desc.SetInput("BackwardMulWeight",
                     {matched.at("bw_mul_w")->arg()->name});
    if (with_mul_bias_) {
      op_desc.SetInput("BackwardMulBias",
                       {matched.at("bw_mul_b")->arg()->name});
    }
    op_desc.SetInput("BackwardGRUWeight",
                     {matched.at("bw_gru_w")->arg()->name});
    op_desc.SetInput("BackwardGRUBias", {matched.at("bw_gru_b")->arg()->name});
    op_desc.SetOutput("ForwardOutput", {matched.at("fw_output")->arg()->name});
    op_desc.SetOutput("BackwardOutput", {matched.at("bw_output")->arg()->name});

    auto fw_mul_op_info = matched.at("fw_mul")->stmt()->op_info();
    op_desc.SetAttr<int>("fw_mul_x_num_col_dims",
                         fw_mul_op_info->GetAttr<int>("x_num_col_dims"));
    op_desc.SetAttr<int>("fw_mul_y_num_col_dims",
                         fw_mul_op_info->GetAttr<int>("y_num_col_dims"));
    auto bw_mul_op_info = matched.at("bw_mul")->stmt()->op_info();
    op_desc.SetAttr<int>("bw_mul_x_num_col_dims",
                         bw_mul_op_info->GetAttr<int>("x_num_col_dims"));
    op_desc.SetAttr<int>("bw_mul_y_num_col_dims",
                         bw_mul_op_info->GetAttr<int>("y_num_col_dims"));
    auto fw_gru_op_info = matched.at("fw_gru")->stmt()->op_info();
    op_desc.SetAttr<std::string>(
        "fw_gru_activation",
        fw_gru_op_info->GetAttr<std::string>("activation"));
    op_desc.SetAttr<std::string>(
        "fw_gru_gate_activation",
        fw_gru_op_info->GetAttr<std::string>("gate_activation"));
    op_desc.SetAttr<bool>("fw_gru_origin_mode",
                          fw_gru_op_info->GetAttr<bool>("origin_mode"));
    auto bw_gru_op_info = matched.at("bw_gru")->stmt()->op_info();
    op_desc.SetAttr<std::string>(
        "bw_gru_activation",
        bw_gru_op_info->GetAttr<std::string>("activation"));
    op_desc.SetAttr<std::string>(
        "bw_gru_gate_activation",
        bw_gru_op_info->GetAttr<std::string>("gate_activation"));
    op_desc.SetAttr<bool>("bw_gru_origin_mode",
                          bw_gru_op_info->GetAttr<bool>("origin_mode"));
    op_desc.SetAttr<bool>("has_mul_b", with_mul_bias_);

    auto fw_mul_op = matched.at("fw_mul")->stmt()->op();
    auto* scope = fw_mul_op->scope();
    auto& valid_places = fw_mul_op->valid_places();
    auto bigru_op = LiteOpRegistry::Global().Create(op_desc.Type());
    bigru_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(bigru_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(matched.at("fw_mul_w"), new_op_node);
    if (with_mul_bias_) {
      IR_NODE_LINK_TO(matched.at("fw_mul_b"), new_op_node);
    }
    IR_NODE_LINK_TO(matched.at("fw_gru_w"), new_op_node);
    IR_NODE_LINK_TO(matched.at("fw_gru_b"), new_op_node);
    IR_NODE_LINK_TO(matched.at("bw_mul_w"), new_op_node);
    if (with_mul_bias_) {
      IR_NODE_LINK_TO(matched.at("bw_mul_b"), new_op_node);
    }
    IR_NODE_LINK_TO(matched.at("bw_gru_w"), new_op_node);
    IR_NODE_LINK_TO(matched.at("bw_gru_b"), new_op_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("fw_output"));
    IR_NODE_LINK_TO(new_op_node, matched.at("bw_output"));
  }

 private:
  bool with_mul_bias_;
};

}  // namespace fusion

class XPUBiGRUFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto with_mul_bias : {true, false}) {
      fusion::XPUBiGRUFuser fuser(with_mul_bias);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__bigru_fuse_pass, paddle::lite::mir::XPUBiGRUFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__bigru");
