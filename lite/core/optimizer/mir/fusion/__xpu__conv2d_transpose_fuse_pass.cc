// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef LITE_WITH_XPU
#include "lite/backends/xpu/xpu_header_sitter.h"
#endif
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {
/* fuse conv2d block in resnet50-like model to xpu_conv2d op    */
/* For example:                                                 */
/* graph[1]: sub block                                          */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                conv2d_transpose----in_Filter                 */
/*                       |                                      */
/*                       |                                      */
/*                  elementwise_add -----ew_add                 */
/*                       |                                      */
/*                       |                                      */
/*                   batch_norm ------in_Bias                   */
/*                       |                                      */
/*                       |                                      */
/*                      act                                     */
/*                       |                                      */
/*                       |                                      */
/*                     out_Out                                  */
/*                                                              */

class XPUConv2dTransFuser : public FuseBase {
 public:
  explicit XPUConv2dTransFuser(const std::string& act_type,
                               bool with_conv_bias,
                               bool with_ew_bias,
                               bool with_bn) {
    act_type_ = act_type;
    with_conv_bias_ = with_conv_bias;
    with_ew_bias_ = with_ew_bias;
    with_bn_ = with_bn;
  }
  void BuildPattern() override {
    PMNode* ew_bias_add = nullptr;
    PMNode* ew_bias_add_y = nullptr;
    PMNode* ew_bias_add_out = nullptr;
    PMNode* bn_bias = nullptr;
    PMNode* bn_mean = nullptr;
    PMNode* bn_scale = nullptr;
    PMNode* bn_var = nullptr;
    PMNode* bn = nullptr;
    PMNode* bn_out = nullptr;
    PMNode* bn_mean_out = nullptr;
    PMNode* bn_saved_mean = nullptr;
    PMNode* bn_var_out = nullptr;
    PMNode* bn_saved_var = nullptr;
    PMNode* act = nullptr;
    PMNode* act_out = nullptr;

    // only support no output_padding and xpu2 now.
    auto unsupported_cond = [](const Node* node) -> bool {
      auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
      uint64_t cur_dev_attr_ = 0;
#ifdef LITE_WITH_XPU
      int cur_dev_idx = 0;
      XPU_CALL(xpu_current_device(&cur_dev_idx));
      XPU_CALL(xpu_device_get_attr(&cur_dev_attr_, XPUATTR_MODEL, cur_dev_idx));
#endif
      bool xpu2_only = (cur_dev_attr_ >= 2 && cur_dev_attr_ <= 299);
      return (!op_desc.HasAttr("output_padding") && xpu2_only);
    };

    auto* input = VarNode("input")
                      ->assert_is_op_input("conv2d_transpose", "Input")
                      ->AsInput();
    auto* conv_filter = VarNode("conv2d_trans_filter")
                            ->assert_is_op_input("conv2d_transpose", "Filter")
                            ->AsInput();
    auto* conv = OpNode("conv2d_trans", "conv2d_transpose")
                     ->assert_node_satisfied(unsupported_cond)
                     ->AsIntermediate();
    auto* conv_out = VarNode("conv2d_trans_out")
                         ->assert_is_op_output("conv2d_transpose", "Output");
    // bias
    if (with_ew_bias_) {
      conv_out->assert_is_op_input("elementwise_add", "X");
      ew_bias_add_y = VarNode("ew_bias_add_y")
                          ->assert_is_op_input("elementwise_add", "Y")
                          ->assert_is_persistable_var()
                          ->assert_only_one_output()
                          ->AsIntermediate();
      ew_bias_add = OpNode("ew_bias_add", "elementwise_add")->AsIntermediate();
      ew_bias_add_out = VarNode("ew_bias_add_out")
                            ->assert_is_op_output("elementwise_add", "Out");
    }

    // bn
    if (with_bn_) {
      bn_scale = VarNode("bn_scale")
                     ->assert_is_op_input("batch_norm", "Scale")
                     ->AsIntermediate();
      bn_bias = VarNode("bn_bias")
                    ->assert_is_op_input("batch_norm", "Bias")
                    ->AsIntermediate();
      bn_mean = VarNode("bn_mean")
                    ->assert_is_op_input("batch_norm", "Mean")
                    ->AsIntermediate();
      bn_var = VarNode("bn_variance")
                   ->assert_is_op_input("batch_norm", "Variance")
                   ->AsIntermediate();
      bn = OpNode("bn", "batch_norm")->AsIntermediate();
      bn_out = VarNode("bn_out")->assert_is_op_output("batch_norm", "Y");
      bn_mean_out = VarNode("bn_mean_out")
                        ->assert_is_op_output("batch_norm", "MeanOut")
                        ->AsIntermediate();
      bn_saved_mean = VarNode("bn_saved_mean")
                          ->assert_is_op_output("batch_norm", "SavedMean")
                          ->AsIntermediate();
      bn_var_out = VarNode("bn_var_out")
                       ->assert_is_op_output("batch_norm", "VarianceOut")
                       ->AsIntermediate();
      bn_saved_var = VarNode("bn_saved_var")
                         ->assert_is_op_output("batch_norm", "SavedVariance")
                         ->AsIntermediate();
    }
    // act
    if (act_type_ != "linear") {
      act = OpNode("act", act_type_)->AsIntermediate();
      act_out =
          VarNode("act_out")->assert_is_op_output(act_type_, "Out")->AsOutput();
    }
    if (with_conv_bias_) {
      auto* conv_bias = VarNode("conv2d_trans_bias")
                            ->assert_is_op_input("conv2d_transpose", "Bias")
                            ->AsIntermediate();
      conv->LinksFrom({input, conv_filter, conv_bias}).LinksTo({conv_out});
    } else {
      conv->LinksFrom({input, conv_filter}).LinksTo({conv_out});
    }
    if (with_ew_bias_) {
      ew_bias_add->LinksFrom({conv_out, ew_bias_add_y})
          .LinksTo({ew_bias_add_out});
    } else {
      ew_bias_add_out = conv_out;
    }
    if (with_bn_) {
      ew_bias_add_out->assert_is_op_input("batch_norm", "X")->AsIntermediate();
      bn->LinksFrom({ew_bias_add_out, bn_scale, bn_bias, bn_mean, bn_var})
          .LinksTo(
              {bn_out, bn_mean_out, bn_saved_mean, bn_saved_var, bn_var_out});
    } else {
      bn_out = ew_bias_add_out;
    }

    if (act_type_ != "linear") {
      bn_out->assert_is_op_input(act_type_, "X")->AsIntermediate();
      act->LinksFrom({bn_out}).LinksTo({act_out});
    } else {
      act_out = bn_out;
    }
    act_out->AsOutput();
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto conv_instruct = matched.at("conv2d_trans")->stmt();
    auto conv_op_desc = conv_instruct->mutable_op_info();
    auto* scope = conv_instruct->op()->scope();

    cpp::OpDesc op_desc;
    op_desc.SetType("conv2d_transpose");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    // conv
    std::string conv_weight_name =
        matched.at("conv2d_trans_filter")->arg()->name;
    auto conv_weight_t =
        scope->FindVar(conv_weight_name)->GetMutable<lite::Tensor>();

    std::string fusion_bias_name = conv_weight_name + "_conv_trans_fusion_bias";
    auto* fusion_bias_node = graph->NewArgumentNode(fusion_bias_name);
    fusion_bias_node->arg()->is_weight = true;
    fusion_bias_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* fusion_bias_t = scope->MutableParent()->NewTensor(fusion_bias_name);
    fusion_bias_t->set_precision(paddle::lite_api::PrecisionType::kFloat);

    auto groups = conv_op_desc->GetAttr<int>("groups");
    // bias's dim is same to filter's num;
    int filter_num = conv_weight_t->dims()[1] * groups;
    fusion_bias_t->Resize({filter_num});
    float* fusion_bias_ptr = fusion_bias_t->mutable_data<float>();
    for (int i = 0; i < filter_num; ++i) {
      fusion_bias_ptr[i] = 0.0f;
    }
    // conv2d_trans's Bias
    if (with_conv_bias_ && conv_op_desc->HasInput("Bias") &&
        conv_op_desc->Input("Bias").size() > 0) {
      auto conv_bias_t =
          scope->FindVar(matched.at("conv2d_trans_bias")->arg()->name)
              ->GetMutable<lite::Tensor>();
      CHECK_EQ((int)conv_bias_t->data_size(), filter_num);
      auto conv_bias_d = conv_bias_t->data<float>();
      for (int i = 0; i < filter_num; ++i) {
        fusion_bias_ptr[i] += conv_bias_d[i];
      }
    }
    // elementwise's bias
    if (with_ew_bias_) {
      auto ew_bias_tensor =
          scope->FindVar(matched.at("ew_bias_add_y")->arg()->name)
              ->GetMutable<lite::Tensor>();
      CHECK_EQ((int)ew_bias_tensor->data_size(), filter_num);
      auto ew_bias_ptr = ew_bias_tensor->data<float>();
      for (int i = 0; i < filter_num; ++i) {
        fusion_bias_ptr[i] += ew_bias_ptr[i];
      }
    }
    // bn
    if (with_bn_) {
      auto bn_scale_t = scope->FindVar(matched.at("bn_scale")->arg()->name)
                            ->GetMutable<lite::Tensor>();
      auto bn_scale_d = bn_scale_t->mutable_data<float>();
      auto bn_mean_t = scope->FindVar(matched.at("bn_mean")->arg()->name)
                           ->GetMutable<lite::Tensor>();
      auto bn_mean_d = bn_mean_t->mutable_data<float>();
      auto bn_var_t = scope->FindVar(matched.at("bn_variance")->arg()->name)
                          ->GetMutable<lite::Tensor>();
      auto bn_var_d = bn_var_t->mutable_data<float>();
      auto bn_bias_t = scope->FindVar(matched.at("bn_bias")->arg()->name)
                           ->GetMutable<lite::Tensor>();
      auto bn_bias_d = bn_bias_t->mutable_data<float>();
      auto eps = matched.at("bn")->stmt()->op_info()->GetAttr<float>("epsilon");
      CHECK_EQ(static_cast<int>(bn_scale_t->data_size()), filter_num)
          << "The BN bias's size should be equal to the size of the first "
          << "dim size of the conv weights";

      Tensor alpha_tensor, beta_tensor;
      alpha_tensor.CopyDataFrom(*bn_bias_t);
      beta_tensor.CopyDataFrom(*bn_bias_t);
      auto alpha_data = alpha_tensor.mutable_data<float>();
      auto beta_data = beta_tensor.mutable_data<float>();

      // conv(x) = kx + z = y
      // bn(conv(x)) = a(kx + z) + b = akx + az + b
      for (int i = 0; i < filter_num; i++) {
        alpha_data[i] = bn_scale_d[i] / std::sqrt(bn_var_d[i] + eps);
      }
      for (int i = 0; i < filter_num; i++) {
        beta_data[i] = (-bn_mean_d[i]) * alpha_data[i];
      }
      // compute new conv_weight
      auto conv_weight_d = conv_weight_t->mutable_data<float>();
      int cout_group = conv_weight_t->dims()[1];
      int cin_group = conv_weight_t->dims()[0] / groups;
      int c_size =
          cout_group * conv_weight_t->dims()[2] * conv_weight_t->dims()[3];
      int hw = conv_weight_t->dims()[2] * conv_weight_t->dims()[3];
      for (int g = 0; g < groups; g++) {
        for (int k = 0; k < cin_group; ++k) {
          for (int i = 0; i < cout_group; ++i) {
            auto ptr_row =
                conv_weight_d + g * cin_group * c_size + k * c_size + i * hw;
            for (int j = 0; j < hw; ++j) {
              ptr_row[j] *= alpha_data[g * cout_group + i];
            }
          }
        }
      }
      // compute new conv_bias
      for (int i = 0; i < filter_num; ++i) {
        fusion_bias_ptr[i] =
            fusion_bias_ptr[i] * alpha_data[i] + bn_bias_d[i] + beta_data[i];
      }
    }
    fusion_bias_t->set_persistable(true);
    op_desc.SetInput("Bias", {fusion_bias_name});

    if (act_type_ != "linear") {
      op_desc.SetAttr("with_act", true);
      op_desc.SetAttr("act_type", act_type_);
    } else {
      op_desc.SetAttr("with_act", false);
    }

    if ((conv_op_desc->HasAttr("padding_algorithm"))) {
      op_desc.SetAttr<std::string>(
          "padding_algorithm",
          conv_op_desc->GetAttr<std::string>("padding_algorithm"));
    }
    op_desc.SetAttr<int>("groups", conv_op_desc->GetAttr<int>("groups"));
    op_desc.SetAttr<std::vector<int>>(
        "paddings", conv_op_desc->GetAttr<std::vector<int>>("paddings"));
    op_desc.SetAttr<std::vector<int>>(
        "strides", conv_op_desc->GetAttr<std::vector<int>>("strides"));
    op_desc.SetAttr<std::vector<int>>(
        "dilations", conv_op_desc->GetAttr<std::vector<int>>("dilations"));

    std::string output_name, output_node_name;
    if (act_type_ != "linear") {
      output_name = matched.at("act_out")->arg()->name;
      output_node_name = "act_out";
    } else if (with_bn_) {
      output_name = matched.at("bn_out")->arg()->name;
      output_node_name = "bn_out";
    } else if (with_ew_bias_) {
      output_name = matched.at("ew_bias_add_out")->arg()->name;
      output_node_name = "ew_bias_add_out";
    } else {
      output_name = matched.at("conv2d_trans_out")->arg()->name;
      output_node_name = "conv2d_trans_out";
    }
    op_desc.SetInput("Filter", {conv_weight_name});
    op_desc.SetOutput("Output", {output_name});

    // new op
    auto new_conv_op = LiteOpRegistry::Global().Create("conv2d_transpose");
    auto& valid_places = conv_instruct->op()->valid_places();
    new_conv_op->Attach(op_desc, scope);
    auto* new_op_node =
        graph->GraphCreateInstructNode(new_conv_op, valid_places);
    DirectedLink(matched.at("input"), new_op_node);
    DirectedLink(matched.at("conv2d_trans_filter"), new_op_node);
    if (with_bn_ || with_conv_bias_ || with_ew_bias_) {
      DirectedLink(fusion_bias_node, new_op_node);
    }
    DirectedLink(new_op_node, matched.at(output_node_name));
  }

 private:
  std::string act_type_;
  bool with_bn_;
  bool with_conv_bias_;
  bool with_ew_bias_;
};

}  // namespace fusion

class XPUConv2dTransFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto with_conv_bias : {true, false}) {
      for (auto with_ew_bias : {true, false}) {
        for (auto with_bn : {true, false}) {
          for (auto act_type : {"relu"}) {
            fusion::XPUConv2dTransFuser fuser(
                act_type, with_conv_bias, with_ew_bias, with_bn);
            fuser(graph.get());
          }
        }
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__conv2d_transpose_fuse_pass,
                  paddle::lite::mir::XPUConv2dTransFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("conv2d_transpose");
