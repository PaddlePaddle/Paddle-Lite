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
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {
/* fuse conv2d-affine_channel block in resnet50-like model to xpu_conv2d op */
/* For example:                                                 */
/* graph[1]: sub block                                          */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                     conv2d----in_Filter                      */
/*                       |                                      */
/*                       |                                      */
/*                 affine_channel ---in_Bias                    */
/*                       |                                      */
/*                       |                                      */
/*                     out_Out                                  */
/*                                                              */
/* After the pass is applied:                                   */
/*                     in_Input                                 */
/*        in_Filter      |     in_FilterMax                     */
/*                  \    |    /                                 */
/*                   \   |   /                                  */
/*     in_Bias ------- __xpu__conv2d                            */
/*                       |    \                                 */
/*                       |     \                                */
/*                       |      out_OutputMax                   */
/*                     out_Output                               */
/*                                                              */
/* ------------------------------------------------------       */
/* graph[2]: sub block                                          */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                     conv2d----in_Filter                      */
/*                       |                                      */
/*                       |                                      */
/*        in_X   affine_channel ------in_Bias                   */
/*             \         |                                      */
/*               \       |                                      */
/*                elementwise_add                               */
/*                       |                                      */
/*                       |                                      */
/*                      act                                     */
/*                       |                                      */
/*                       |                                      */
/*                     out_Out                                  */
/*                                                              */
/* After the pass is applied:                                   */
/*                     in_Input                                 */
/*        in_Filter      |     in_FilterMax                     */
/*                  \    |    /                                 */
/*                   \   |   /                                  */
/*  in_Branch ------- __xpu__conv2d ------ in_Bias              */
/*                       |    \                                 */
/*                       |     \                                */
/*                       |      out_OutputMax                   */
/*                    out_Output                                */
/* ------------------------------------------------------       */

class XPUConv2dAffineChannelFuser : public FuseBase {
 public:
  explicit XPUConv2dAffineChannelFuser(const std::string& conv_type,
                                       const std::string& act_type,
                                       bool with_branch) {
    conv_type_ = conv_type;
    act_type_ = act_type;
    with_branch_ = with_branch;
  }

  void BuildPattern() override {
    PMNode* ew_branch_add = nullptr;
    PMNode* ew_branch_add_in = nullptr;
    PMNode* ew_branch_add_out = nullptr;
    PMNode* act = nullptr;
    PMNode* act_out = nullptr;
    auto* input =
        VarNode("input")->assert_is_op_input(conv_type_, "Input")->AsInput();
    auto* conv_filter = VarNode("conv_filter")
                            ->assert_is_op_input(conv_type_, "Filter")
                            ->AsInput();
    auto* conv = OpNode("conv", conv_type_)->AsIntermediate();
    auto* conv_out = VarNode("conv_out")
                         ->assert_is_op_output(conv_type_, "Output")
                         ->AsIntermediate();

    auto* bias = VarNode("bias")
                     ->assert_is_op_input("affine_channel", "Bias")
                     ->assert_only_one_output()
                     ->assert_is_persistable_var()
                     ->AsInput();

    auto* scale = VarNode("scale")
                      ->assert_is_op_input("affine_channel", "Scale")
                      ->assert_only_one_output()
                      ->assert_is_persistable_var()
                      ->AsIntermediate();

    auto* affine_channel =
        OpNode("affine_channel", "affine_channel")->AsIntermediate();

    auto* affine_channel_out =
        VarNode("affine_channel_out")
            ->assert_is_op_output("affine_channel", "Out");
    // branch
    if (with_branch_) {
      ew_branch_add_in = VarNode("ew_branch_add_in")
                             ->assert_is_op_input("elementwise_add", "X")
                             ->AsInput();
      ew_branch_add =
          OpNode("ew_branch_add", "elementwise_add")->AsIntermediate();
      ew_branch_add_out = VarNode("ew_branch_add_out")
                              ->assert_is_op_output("elementwise_add", "Out");
    }
    // act
    if (act_type_ != "linear") {
      act = OpNode("act", act_type_)->AsIntermediate();
      act_out =
          VarNode("act_out")->assert_is_op_output(act_type_, "Out")->AsOutput();
    }
    // pass
    *input >> *conv >> *conv_out >> *affine_channel >> *affine_channel_out;
    if (with_branch_) {
      *affine_channel_out >> *ew_branch_add;
      *ew_branch_add_in >> *ew_branch_add >> *ew_branch_add_out;
    } else {
      ew_branch_add_out = affine_channel_out;
    }
    if (act_type_ != "linear") {
      ew_branch_add_out->assert_is_op_input(act_type_, "X")->AsIntermediate();
      *ew_branch_add_out >> *act >> *act_out;
    } else {
      act_out = ew_branch_add_out;
    }
    act_out->AsOutput();
    *conv_filter >> *conv;
    *bias >> *affine_channel;
    *scale >> *affine_channel;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    auto conv_old = matched.at("conv")->stmt()->op();
    auto* scope = conv_old->scope();
    op_desc.SetType("__xpu__conv2d");
    std::string input_name = matched.at("input")->arg()->name;
    op_desc.SetInput("Input", {input_name});
    op_desc.SetInput("Bias", {matched.at("bias")->arg()->name});
    auto filter_name = matched.at("conv_filter")->arg()->name;
    auto* filter_t = scope->FindMutableTensor(filter_name);
    auto& f_dims = filter_t->dims();
    std::vector<int> filter_dims{static_cast<int>(f_dims[0]),
                                 static_cast<int>(f_dims[1]),
                                 static_cast<int>(f_dims[2]),
                                 static_cast<int>(f_dims[3])};
    std::vector<int> conv_groups{
        matched.at("conv")->stmt()->op_info()->GetAttr<int>("groups")};
    std::vector<int> conv_bias{1};
    op_desc.SetAttr<bool>("has_bias", true);
    op_desc.SetAttr<std::vector<int>>("filter_dims", filter_dims);
    op_desc.SetAttr<std::vector<int>>("op_type", std::vector<int>{0});
    op_desc.SetAttr<std::vector<int>>("place_x", std::vector<int>{0});
    op_desc.SetAttr<std::vector<int>>("place_y", std::vector<int>{9});
    op_desc.SetAttr<std::vector<int>>("place_z", std::vector<int>{10});
    op_desc.SetAttr<std::vector<int>>(
        "strides",
        matched.at("conv")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "strides"));
    auto conv_paddings =
        matched.at("conv")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "paddings");
    if (conv_paddings.size() == 2) {
      for (size_t i = 0; i < 2; ++i) {
        int copy_pad = *(conv_paddings.begin() + 2 * i);
        conv_paddings.insert(conv_paddings.begin() + 2 * i + 1, copy_pad);
      }
    }
    CHECK_EQ(conv_paddings.size(), 4UL)
        << "Paddings size should be 2 or 4, But received paddings size: "
        << conv_paddings.size();
    op_desc.SetAttr<std::vector<int>>("paddings", conv_paddings);

    op_desc.SetAttr<std::vector<int>>(
        "dilations",
        matched.at("conv")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "dilations"));
    op_desc.SetAttr<std::vector<int>>("groups", conv_groups);
    op_desc.SetAttr<std::vector<int>>("block_lod", std::vector<int>{1});
    op_desc.SetAttr<std::vector<int>>("conv_bias", conv_bias);

    auto scale_name = matched.at("scale")->arg()->name;
    auto* scale_t = scope->FindMutableTensor(scale_name);
    float* scale_on_host = scale_t->mutable_data<float>();
    float* filter_on_host = filter_t->mutable_data<float>();
    int filter_len = filter_t->numel();
    int filter_stride = filter_len / filter_dims[0];
    for (int i = 0; i < filter_dims[0]; i++) {
      for (int j = 0; j < filter_stride; j++) {
        filter_on_host[i * filter_stride + j] *= scale_on_host[i];
      }
    }
    op_desc.SetInput("Filter", {filter_name});
    if (with_branch_) {
      op_desc.SetInput("Branch", {matched.at("ew_branch_add_in")->arg()->name});
    }
    op_desc.SetAttr<bool>("has_branch", with_branch_);
    std::map<std::string, int> act_map{{"linear", 0},
                                       {"relu", 1},
                                       {"sigmoid", 2},
                                       {"tanh", 3},
                                       {"leaky_relu", 5},
                                       {"hard_swish", 14},
                                       {"hard_sigmoid", 15},
                                       {"swish", 16},
                                       {"relu6", 17}};

    float act_param_ = 0.0f;
    if (act_type_ != "linear") {
      if (act_type_ == "leaky_relu") {
        auto act_op_desc = *matched.at("act")->stmt()->op_info();
        act_param_ = act_op_desc.GetAttr<float>("alpha");
      } else if (act_type_ == "hard_sigmoid") {
        auto act_op_desc = *matched.at("act")->stmt()->op_info();
        act_param_ = act_op_desc.GetAttr<float>("slope");
      }
    }
    op_desc.SetAttr<std::vector<int>>("act_type",
                                      std::vector<int>{act_map[act_type_]});
    op_desc.SetAttr<std::vector<float>>("act_param",
                                        std::vector<float>{act_param_});

    if ((matched.at("conv")->stmt()->op_info()->HasAttr("padding_algorithm"))) {
      op_desc.SetAttr<std::string>(
          "padding_algorithm",
          matched.at("conv")->stmt()->op_info()->GetAttr<std::string>(
              "padding_algorithm"));
    }

    std::string output_name, output_node_name;
    if (act_type_ != "linear") {
      output_name = matched.at("act_out")->arg()->name;
      output_node_name = "act_out";
    } else if (with_branch_) {
      output_name = matched.at("ew_branch_add_out")->arg()->name;
      output_node_name = "ew_branch_add_out";
    } else {
      output_name = matched.at("affine_channel_out")->arg()->name;
      output_node_name = "affine_channel_out";
    }
    op_desc.SetOutput("Output", {output_name});

    std::string max_output_name = output_name + "_max";
    auto* max_output_node = graph->NewArgumentNode(max_output_name);
    max_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* max_output_tensor = scope->NewTensor(max_output_name);
    max_output_tensor->set_precision(paddle::lite_api::PrecisionType::kFloat);
    max_output_tensor->set_persistable(true);
    op_desc.SetOutput("OutputMax", {max_output_name});

    auto conv_op = LiteOpRegistry::Global().Create("__xpu__conv2d");
    auto& valid_places = conv_old->valid_places();
    conv_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(conv_op, valid_places);
    DirectedLink(matched.at("input"), new_op_node);
    DirectedLink(matched.at("conv_filter"), new_op_node);
    DirectedLink(new_op_node, max_output_node);
    DirectedLink(matched.at("bias"), new_op_node);
    if (with_branch_) {
      DirectedLink(matched.at("ew_branch_add_in"), new_op_node);
    }
    DirectedLink(new_op_node, matched.at(output_node_name));
  }

 private:
  std::string conv_type_;
  std::string act_type_;
  bool with_branch_;
};

}  // namespace fusion

class XPUConv2dAffineChannelFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    for (auto conv_type : {"conv2d", "depthwise_conv2d"}) {
      for (auto with_branch : {true, false}) {
        for (auto act_type : {"relu",
                              "sigmoid",
                              "tanh",
                              "leaky_relu",
                              "hard_swish",
                              "hard_sigmoid",
                              "relu6",
                              "swish",
                              "linear"}) {
          fusion::XPUConv2dAffineChannelFuser fuser(
              conv_type, act_type, with_branch);
          fuser(graph.get());
        }
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__conv2d_affine_channel_fuse_pass,
                  paddle::lite::mir::XPUConv2dAffineChannelFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__conv2d");
