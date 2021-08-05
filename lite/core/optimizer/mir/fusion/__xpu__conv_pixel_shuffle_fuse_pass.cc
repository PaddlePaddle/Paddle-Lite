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
/* fuse xpu_conv2d and pixel_shuffle and xpu_conv2d as xpu_block */
/* the purpose of this fuse is to reduce memory footprint of xpu */
/*                                                               */
/*                            in_Input                           */
/*                                |                              */
/*                                |                              */
/*                           xpu_conv2d                          */
/*                                |                              */
/*                                |                              */
/*                          pixel_shuffle                        */
/*                                |                              */
/*                                |                              */
/*                            xpu_conv2d                         */
/*                                |                              */
/*                                |                              */
/*                            out_Output                         */
/*---------------------------------------------------------------*/

class XPUConvPixelShuffleFuser : public FuseBase {
 public:
  explicit XPUConvPixelShuffleFuser(bool with_bias_0, bool with_bias_1) {
    with_bias_0_ = with_bias_0;
    with_bias_1_ = with_bias_1;
  }
  void BuildPattern() override {
    // first xpu_conv2d
    PMNode* bias_0 = nullptr;
    auto* input_0 = VarNode("input_0")
                        ->assert_is_op_input("__xpu__conv2d", "Input")
                        ->AsInput();
    auto* filter_0 = VarNode("filter_0")
                         ->assert_is_op_input("__xpu__conv2d", "Filter")
                         ->assert_is_persistable_var()
                         ->AsInput();
    if (with_bias_0_) {
      bias_0 = VarNode("bias_0")
                   ->assert_is_op_input("__xpu__conv2d", "Bias")
                   ->assert_is_persistable_var()
                   ->AsInput();
    }

    auto* conv_0 = OpNode("conv_0", "__xpu__conv2d")
                       ->assert_op_attr<bool>("has_branch", false)
                       ->assert_op_attr<bool>("has_bias", with_bias_0_)
                       ->AsIntermediate();
    auto* conv_out_0 = VarNode("conv_out_0")
                           ->assert_is_op_output("__xpu__conv2d", "Output")
                           ->AsIntermediate();
    auto* conv_out_0_max =
        VarNode("conv_out_0_max")
            ->assert_is_op_output("__xpu__conv2d", "OutputMax")
            ->AsIntermediate();

    // pixel_shuffle
    conv_out_0->assert_is_op_input("pixel_shuffle", "X");
    auto* pixel_shuffle =
        OpNode("pixel_shuffle", "pixel_shuffle")->AsIntermediate();
    auto* shuffle_out = VarNode("shuffle_out")
                            ->assert_is_op_output("pixel_shuffle", "Out")
                            ->AsIntermediate();

    // second xpu_conv2d
    PMNode* bias_1 = nullptr;
    shuffle_out->assert_is_op_input("__xpu__conv2d", "Input");
    auto* filter_1 = VarNode("filter_1")
                         ->assert_is_op_input("__xpu__conv2d", "Filter")
                         ->assert_is_persistable_var()
                         ->AsInput();
    if (with_bias_1_) {
      bias_1 = VarNode("bias_1")
                   ->assert_is_op_input("__xpu__conv2d", "Bias")
                   ->assert_is_persistable_var()
                   ->AsInput();
    }
    auto* conv_1 = OpNode("conv_1", "__xpu__conv2d")
                       ->assert_op_attr<bool>("has_branch", false)
                       ->assert_op_attr<bool>("has_bias", with_bias_1_)
                       ->AsIntermediate();
    auto* conv_out_1 = VarNode("conv_out_1")
                           ->assert_is_op_output("__xpu__conv2d", "Output")
                           ->AsOutput();
    auto* conv_out_1_max =
        VarNode("conv_out_1_max")
            ->assert_is_op_output("__xpu__conv2d", "OutputMax")
            ->AsOutput();

    *input_0 >> *conv_0 >> *conv_out_0 >> *pixel_shuffle >> *shuffle_out >>
        *conv_1 >> *conv_out_1;
    *filter_0 >> *conv_0 >> *conv_out_0_max;
    *filter_1 >> *conv_1 >> *conv_out_1_max;
    if (with_bias_0_) {
      *bias_0 >> *conv_0;
    }
    if (with_bias_1_) {
      *bias_1 >> *conv_1;
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    auto conv = matched.at("conv_0")->stmt()->op();
    auto* scope = conv->scope();
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("__xpu__conv_pixel_shuffle_fuse_op");
    op_desc.SetInput("Input", {matched.at("input_0")->arg()->name});
    op_desc.SetInput("Filter_0", {matched.at("filter_0")->arg()->name});
    op_desc.SetInput("Filter_1", {matched.at("filter_1")->arg()->name});
    if (with_bias_0_) {
      op_desc.SetInput("Bias_0", {matched.at("bias_0")->arg()->name});
    }
    if (with_bias_1_) {
      op_desc.SetInput("Bias_1", {matched.at("bias_1")->arg()->name});
    }
    op_desc.SetOutput("Output", {matched.at("conv_out_1")->arg()->name});
    op_desc.SetOutput("OutputMax", {matched.at("conv_out_1_max")->arg()->name});
    op_desc.SetAttr<std::vector<int>>(
        "strides_0",
        matched.at("conv_0")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "strides"));
    op_desc.SetAttr<std::vector<int>>(
        "strides_1",
        matched.at("conv_1")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "strides"));
    op_desc.SetAttr<std::vector<int>>(
        "paddings_0",
        matched.at("conv_0")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "paddings"));
    op_desc.SetAttr<std::vector<int>>(
        "paddings_1",
        matched.at("conv_1")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "paddings"));
    op_desc.SetAttr<std::vector<int>>(
        "dilations_0",
        matched.at("conv_0")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "dilations"));
    op_desc.SetAttr<std::vector<int>>(
        "dilations_1",
        matched.at("conv_1")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "dilations"));
    op_desc.SetAttr<std::vector<int>>(
        "groups_0",
        matched.at("conv_0")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "groups"));
    op_desc.SetAttr<std::vector<int>>(
        "groups_1",
        matched.at("conv_1")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "groups"));
    op_desc.SetAttr<std::vector<int>>(
        "act_type_0",
        matched.at("conv_0")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "act_type"));
    op_desc.SetAttr<std::vector<int>>(
        "act_type_1",
        matched.at("conv_1")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "act_type"));
    op_desc.SetAttr<std::vector<float>>(
        "act_param_0",
        matched.at("conv_0")->stmt()->op_info()->GetAttr<std::vector<float>>(
            "act_param"));
    op_desc.SetAttr<std::vector<float>>(
        "act_param_1",
        matched.at("conv_1")->stmt()->op_info()->GetAttr<std::vector<float>>(
            "act_param"));
    op_desc.SetAttr<std::string>(
        "padding_algorithm_0",
        matched.at("conv_0")->stmt()->op_info()->GetAttr<std::string>(
            "padding_algorithm"));
    op_desc.SetAttr<std::string>(
        "padding_algorithm_1",
        matched.at("conv_1")->stmt()->op_info()->GetAttr<std::string>(
            "padding_algorithm"));
    op_desc.SetAttr<int>("upscale_factor",
                         matched.at("pixel_shuffle")
                             ->stmt()
                             ->op_info()
                             ->GetAttr<int>("upscale_factor"));
    op_desc.SetAttr<bool>("has_bias_0", with_bias_0_);
    op_desc.SetAttr<bool>("has_bias_1", with_bias_1_);
    auto& valid_places = conv->valid_places();
    auto block_op = LiteOpRegistry::Global().Create(op_desc.Type());
    block_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(block_op, valid_places);
    IR_NODE_LINK_TO(matched.at("input_0"), new_op_node);
    IR_NODE_LINK_TO(matched.at("filter_0"), new_op_node);
    IR_NODE_LINK_TO(matched.at("filter_1"), new_op_node);
    if (with_bias_0_) {
      IR_NODE_LINK_TO(matched.at("bias_0"), new_op_node);
    }
    if (with_bias_1_) {
      IR_NODE_LINK_TO(matched.at("bias_1"), new_op_node);
    }
    IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_1"));
    IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_1_max"));
  }

 private:
  bool with_bias_0_;
  bool with_bias_1_;
};

}  // namespace fusion

class XPUConvPixelShuffleFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto with_bias_0_ : {true, false}) {
      for (auto with_bias_1_ : {true, false}) {
        fusion::XPUConvPixelShuffleFuser fuser(with_bias_0_, with_bias_1_);
        fuser(graph.get());
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__conv_pixel_shuffle_fuse_pass,
                  paddle::lite::mir::XPUConvPixelShuffleFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__conv_pixel_shuffle_fuse_op");
