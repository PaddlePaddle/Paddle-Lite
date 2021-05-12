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
/* fuse xpu_conv2d and concat as xpu_block         */
/*                                                 */
/*                in_Input                         */
/*                /      \                         */
/*              /          \                       */
/*             |            |                      */
/*             |            |                      */
/*             |         xpu_conv2d                */
/*             |            |                      */
/*              \          /                       */
/*                \       /                        */
/*                  concat                         */
/*                    |                            */
/*                    |                            */
/*                out_Output                       */
/*-------------------------------------------------*/

class XPUConcatConv2dFuser : public FuseBase {
 public:
  explicit XPUConcatConv2dFuser(bool input_first, bool with_bias) {
    input_first_ = input_first;
    with_bias_ = with_bias;
  }
  void BuildPattern() override {
    PMNode* bias = nullptr;
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__conv2d", "Input")
                      ->AsInput();
    if (input_first_) {
      input->assert_is_op_nth_input("concat", "X", 0);
    } else {
      input->assert_is_op_nth_input("concat", "X", 1);
    }
    auto* filter = VarNode("filter")
                       ->assert_is_op_input("__xpu__conv2d", "Filter")
                       ->assert_is_persistable_var()
                       ->AsInput();

    if (with_bias_) {
      bias = VarNode("bias")
                 ->assert_is_op_input("__xpu__conv2d", "Bias")
                 ->assert_is_persistable_var()
                 ->AsInput();
    }
    auto* conv = OpNode("conv", "__xpu__conv2d")
                     ->assert_op_attr<bool>("has_branch", false)
                     ->assert_op_attr<bool>("has_bias", with_bias_)
                     ->AsIntermediate();

    auto* conv_out = VarNode("conv_out")
                         ->assert_is_op_output("__xpu__conv2d", "Output")
                         ->AsIntermediate();
    if (input_first_) {
      conv_out->assert_is_op_nth_input("concat", "X", 1);
    } else {
      conv_out->assert_is_op_nth_input("concat", "X", 0);
    }
    auto* conv_out_max = VarNode("conv_out_max")
                             ->assert_is_op_output("__xpu__conv2d", "OutputMax")
                             ->AsIntermediate();

    auto* concat = OpNode("concat", "concat")
                       ->assert_op_attr<int>("axis", 1)
                       ->AsIntermediate();
    auto* concat_out =
        VarNode("concat_out")->assert_is_op_output("concat", "Out")->AsOutput();

    *input >> *conv >> *conv_out >> *concat >> *concat_out;
    *input >> *concat;
    *filter >> *conv;
    if (with_bias_) {
      *bias >> *conv;
    }
    *conv >> *conv_out_max;
  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto op_desc = *matched.at("conv")->stmt()->op_info();
    auto conv = matched.at("conv")->stmt()->op();
    auto* scope = conv->scope();
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("__xpu__block_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetInput("Filter", {matched.at("filter")->arg()->name});
    if (with_bias_) {
      op_desc.SetInput("Bias", {matched.at("bias")->arg()->name});
    }
    op_desc.SetOutput("Output", {matched.at("concat_out")->arg()->name});
    // add new arg output_max
    std::string max_output_name =
        matched.at("concat_out")->arg()->name + "_max";
    auto* max_output_node = graph->NewArgumentNode(max_output_name);
    max_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* max_output_tensor = scope->NewTensor(max_output_name);
    max_output_tensor->set_precision(paddle::lite_api::PrecisionType::kFloat);
    max_output_tensor->set_persistable(true);
    op_desc.SetOutput("OutputMax", {max_output_name});

    std::vector<int> block_lod{2};
    std::vector<int> op_type{0, 20};
    op_desc.SetAttr("op_type", op_type);
    op_desc.SetAttr("block_lod", block_lod);

    if (input_first_) {
      op_desc.SetAttr("place_x", std::vector<int>{0, 0});
      op_desc.SetAttr("place_y", std::vector<int>{9, 1});
      op_desc.SetAttr("place_z", std::vector<int>{1, 10});
    } else {
      op_desc.SetAttr("place_x", std::vector<int>{0, 1});
      op_desc.SetAttr("place_y", std::vector<int>{9, 0});
      op_desc.SetAttr("place_z", std::vector<int>{1, 10});
    }
    op_desc.SetAttr<bool>("has_bias", with_bias_);
    op_desc.SetAttr<bool>("has_branch", false);

    auto& valid_places = conv->valid_places();
    auto block_op = LiteOpRegistry::Global().Create(op_desc.Type());
    block_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(block_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(matched.at("filter"), new_op_node);
    if (with_bias_) {
      IR_NODE_LINK_TO(matched.at("bias"), new_op_node);
    }
    IR_NODE_LINK_TO(new_op_node, max_output_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("concat_out"));
  }

 private:
  bool with_bias_;
  bool input_first_;
};

}  // namespace fusion

class XPUConcatConv2dFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto input_first_ : {true, false}) {
      for (auto with_bias_ : {true, false}) {
        fusion::XPUConcatConv2dFuser fuser(input_first_, with_bias_);
        fuser(graph.get());
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__concat_conv2d_fuse_pass,
                  paddle::lite::mir::XPUConcatConv2dFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");
