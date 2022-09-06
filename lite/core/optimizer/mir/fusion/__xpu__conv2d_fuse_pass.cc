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
/* fuse conv2d block in resnet50-like model to xpu_conv2d op    */
/* For example:                                                 */
/* graph[1]: sub block                                          */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                     conv2d----in_Filter                      */
/*                       |                                      */
/*                       |                                      */
/*                  elementwise_add -----conv_Bias              */
/*                       |                                      */
/*                       |                                      */
/*                  batch_norm ------in_Bias                    */
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
/*     in_Bias ------- __xpu__conv2d                            */
/*                       |    \                                 */
/*                       |     \                                */
/*                       |      out_OutputMax                   */
/*                 out_Output                                   */
/*                                                              */
/* ------------------------------------------------------       */
/* graph[2]: sub block                                          */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                     conv2d----in_Filter                      */
/*                       |                                      */
/*                       |                                      */
/*                  batch_norm ------in_Bias                    */
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
/* graph[3]: sub block                                          */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                     conv2d----in_Filter                      */
/*                       |                                      */
/*                       |                                      */
/*        in_X       batch_norm ------in_Bias                   */
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
/* graph[4]: sub block                                          */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                     conv2d----in_Filter                      */
/*                       |                                      */
/*                       |                                      */
/*                elementwise_add ------in_Bias                 */
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
/*  in_Bias ------- __xpu__conv2d                               */
/*                       |    \                                 */
/*                       |     \                                */
/*                       |      out_OutputMax                   */
/*                    out_Output                                */

class XPUConv2dFuser : public FuseBase {
 public:
  explicit XPUConv2dFuser(const std::string& conv_type,
                          const std::string& act_type,
                          bool with_conv_bias,
                          bool with_bn,
                          bool with_branch_x,
                          bool with_branch_y) {
    conv_type_ = conv_type;
    act_type_ = act_type;
    with_conv_bias_ = with_conv_bias;
    with_bn_ = with_bn;
    with_branch_ = with_branch_x | with_branch_y;
    with_branch_x_ = with_branch_x;
    with_branch_y_ = with_branch_y;
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
    auto* conv_out =
        VarNode("conv_out")->assert_is_op_output(conv_type_, "Output");
    // conv_bias
    if (with_conv_bias_) {
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
      bn_bias = VarNode("bn_bias")
                    ->assert_is_op_input("batch_norm", "Bias")
                    ->assert_only_one_output()
                    ->AsIntermediate();
      bn_mean = VarNode("bn_mean")
                    ->assert_is_op_input("batch_norm", "Mean")
                    ->assert_only_one_output()
                    ->AsIntermediate();
      bn_scale = VarNode("bn_scale")
                     ->assert_is_op_input("batch_norm", "Scale")
                     ->assert_only_one_output()
                     ->AsIntermediate();
      bn_var = VarNode("bn_variance")
                   ->assert_is_op_input("batch_norm", "Variance")
                   ->assert_only_one_output()
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
    // branch
    if (with_branch_ && with_branch_x_) {
      ew_branch_add_in = VarNode("ew_branch_add_in")
                             ->assert_is_op_input("elementwise_add", "X")
                             ->assert_var_not_persistable()
                             ->AsInput();
      ew_branch_add =
          OpNode("ew_branch_add", "elementwise_add")->AsIntermediate();
      ew_branch_add_out = VarNode("ew_branch_add_out")
                              ->assert_is_op_output("elementwise_add", "Out");
    } else if (with_branch_ && with_branch_y_) {
      ew_branch_add_in = VarNode("ew_branch_add_in")
                             ->assert_is_op_input("elementwise_add", "Y")
                             ->assert_var_not_persistable()
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
    *input >> *conv >> *conv_out;
    if (with_conv_bias_) {
      conv_out->AsIntermediate();
      *conv_out >> *ew_bias_add >> *ew_bias_add_out;
      *ew_bias_add_y >> *ew_bias_add;
    } else {
      ew_bias_add_out = conv_out;
    }
    if (with_bn_) {
      ew_bias_add_out->assert_is_op_input("batch_norm", "X")->AsIntermediate();
      *ew_bias_add_out >> *bn >> *bn_out;
      *bn_bias >> *bn;
      *bn_mean >> *bn;
      *bn_scale >> *bn;
      *bn_var >> *bn;
      *bn >> *bn_mean_out;
      *bn >> *bn_saved_mean;
      *bn >> *bn_saved_var;
      *bn >> *bn_var_out;
    } else {
      bn_out = ew_bias_add_out;
    }
    if (with_branch_ && with_branch_x_) {
      bn_out->assert_is_op_input("elementwise_add", "Y")->AsIntermediate();
      *bn_out >> *ew_branch_add >> *ew_branch_add_out;
      *ew_branch_add_in >> *ew_branch_add;
    } else if (with_branch_ && with_branch_y_) {
      bn_out->assert_is_op_input("elementwise_add", "X")->AsIntermediate();
      *bn_out >> *ew_branch_add >> *ew_branch_add_out;
      *ew_branch_add_in >> *ew_branch_add;
    } else {
      ew_branch_add_out = bn_out;
    }
    if (act_type_ != "linear") {
      ew_branch_add_out->assert_is_op_input(act_type_, "X")->AsIntermediate();
      *ew_branch_add_out >> *act >> *act_out;
    } else {
      act_out = ew_branch_add_out;
    }
    act_out->AsOutput();
    *conv_filter >> *conv;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    auto conv_old = matched.at("conv")->stmt()->op();
    auto* scope = conv_old->scope();
    op_desc.SetType("__xpu__conv2d");
    std::string input_name = matched.at("input")->arg()->name;
    op_desc.SetInput("Input", {input_name});
    auto filter_name = matched.at("conv_filter")->arg()->name;
    auto* filter_t = scope->FindMutableTensor(filter_name);
    auto& f_dims = filter_t->dims();
    std::vector<int> filter_dims{static_cast<int>(f_dims[0]),
                                 static_cast<int>(f_dims[1]),
                                 static_cast<int>(f_dims[2]),
                                 static_cast<int>(f_dims[3])};
    std::vector<int> conv_groups{
        matched.at("conv")->stmt()->op_info()->GetAttr<int>("groups")};
    std::vector<int> conv_bias;
    if (with_bn_ || with_conv_bias_) {
      conv_bias.push_back(1);
    } else {
      conv_bias.push_back(0);
    }
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

    std::string fusion_bias_name = filter_name + "_conv_fusion_bias";
    auto* fusion_bias_node = graph->NewArgumentNode(fusion_bias_name);
    fusion_bias_node->arg()->is_weight = true;
    fusion_bias_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* fusion_bias_t = scope->MutableParent()->NewTensor(fusion_bias_name);
    fusion_bias_t->set_precision(paddle::lite_api::PrecisionType::kFloat);

    op_desc.SetAttr<bool>("has_bias", (with_bn_ || with_conv_bias_));
    if (with_bn_ || with_conv_bias_) {
      fusion_bias_t->Resize({f_dims[0]});
      float* fusion_bias_ptr = fusion_bias_t->mutable_data<float>();
      if (with_conv_bias_) {
        auto ew_bias_add_y_name = matched.at("ew_bias_add_y")->arg()->name;
        auto* ew_bias_add_y_t = scope->FindMutableTensor(ew_bias_add_y_name);
        float* ew_bias_add_y_on_host = ew_bias_add_y_t->mutable_data<float>();
        auto ew_bias_add_y_size = ew_bias_add_y_t->numel();
        if (ew_bias_add_y_size != f_dims[0] && ew_bias_add_y_size == 1) {
          for (int i = 0; i < f_dims[0]; ++i) {
            fusion_bias_ptr[i] = ew_bias_add_y_on_host[0];
          }
        } else if (ew_bias_add_y_size == f_dims[0]) {
          for (int i = 0; i < f_dims[0]; ++i) {
            fusion_bias_ptr[i] = ew_bias_add_y_on_host[i];
          }
        } else {
          LOG(WARNING)
              << "Elements size of `elemwise_bias` and 'conv_filter_channels` "
                 "should be the same, but get size of `elemwise_bias` "
                 "is: "
              << ew_bias_add_y_size
              << ", size of `conv_filter_channels` is: " << f_dims[0];
          return;
        }

      } else {
        for (int i = 0; i < f_dims[0]; ++i) {
          fusion_bias_ptr[i] = 0.0f;
        }
      }
      if (with_bn_) {
        auto scale_name = matched.at("bn_scale")->arg()->name;
        auto bias_name = matched.at("bn_bias")->arg()->name;
        auto mean_name = matched.at("bn_mean")->arg()->name;
        auto var_name = matched.at("bn_variance")->arg()->name;

        auto* scale_t = scope->FindMutableTensor(scale_name);
        auto* bias_t = scope->FindMutableTensor(bias_name);
        auto* mean_t = scope->FindMutableTensor(mean_name);
        auto* var_t = scope->FindMutableTensor(var_name);

        int mean_len = mean_t->numel();
        int filter_len = filter_t->numel();
        int filter_stride = filter_len / mean_len;

        float* filter_on_host = filter_t->mutable_data<float>();
        float* scale_on_host = scale_t->mutable_data<float>();
        float* bias_on_host = bias_t->mutable_data<float>();
        float* mean_on_host = mean_t->mutable_data<float>();
        float* var_on_host = var_t->mutable_data<float>();

        for (int i = 0; i < mean_len; ++i) {
          scale_on_host[i] =
              scale_on_host[i] / sqrtf(var_on_host[i] + 0.00001f);
        }
        for (int i = 0; i < mean_len; ++i) {
          for (int j = 0; j < filter_stride; ++j) {
            filter_on_host[i * filter_stride + j] *= scale_on_host[i];
          }
        }
        for (int i = 0; i < mean_len; ++i) {
          bias_on_host[i] +=
              (fusion_bias_ptr[i] - mean_on_host[i]) * scale_on_host[i];
        }
        memcpy(fusion_bias_ptr, bias_on_host, mean_len * sizeof(float));
      }
      fusion_bias_t->set_persistable(true);
      op_desc.SetInput("Bias", {fusion_bias_name});
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
    } else if (with_bn_) {
      output_name = matched.at("bn_out")->arg()->name;
      output_node_name = "bn_out";
    } else if (with_conv_bias_) {
      output_name = matched.at("ew_bias_add_out")->arg()->name;
      output_node_name = "ew_bias_add_out";
    } else {
      output_name = matched.at("conv_out")->arg()->name;
      output_node_name = "conv_out";
    }
    op_desc.SetOutput("Output", {output_name});

    std::string max_output_name = output_name + "_xpu_max";
    auto* max_output_node = graph->NewArgumentNode(max_output_name);
    max_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* max_output_tensor = scope->NewTensor(max_output_name);
    max_output_tensor->set_precision(paddle::lite_api::PrecisionType::kFloat);
    max_output_tensor->set_persistable(true);
    op_desc.SetOutput("OutputMax", {max_output_name});

    // set conv2d int8 attributes
    if (matched.at("conv")->stmt()->op_info()->HasAttr("enable_int8") &&
        matched.at("conv")->stmt()->op_info()->GetAttr<bool>("enable_int8")) {
      op_desc.SetAttr<bool>("enable_int8", true);
      auto op_info = matched.at("conv")->stmt()->op_info();

      auto get_scale_name = [&op_info](const std::string& name) {
        std::string argname;
        int index;
        CHECK(op_info->GetInputArgname(name, &argname));
        CHECK(op_info->GetInputIndex(name, &index));
        std::string scale_name = argname + to_string(index) + "_scale";
        return scale_name;
      };

      op_desc.SetAttr<std::vector<float>>(
          get_scale_name(input_name),
          {127 *
           matched.at("conv")->stmt()->op_info()->GetInputScale(
               input_name)[0]});

      op_desc.SetAttr<std::vector<float>>(
          get_scale_name(filter_name),
          {127 *
           matched.at("conv")->stmt()->op_info()->GetInputScale(
               filter_name)[0]});

      if (with_branch_) {
        std::string branch_name = matched.at("ew_branch_add_in")->arg()->name;
        op_desc.SetAttr<std::vector<float>>(
            "Branch0_scale",
            {127 *
             matched.at("ew_branch_add")
                 ->stmt()
                 ->op_info()
                 ->GetInputScale(branch_name)[0]});
      }

      std::string op_name{};
      if (act_type_ != "linear") {
        op_name = "act";
      } else if (with_branch_) {
        op_name = "ew_branch_add";
      } else if (with_conv_bias_) {
        op_name = "ew_bias_add";
      } else {
        op_name = "conv";
      }
      op_desc.SetAttr<std::vector<float>>(
          "Output0_scale",
          {matched.at(op_name)->stmt()->op_info()->GetAttr<float>(
              "out_threshold")});
    }

    // set conv2d int16 attributes
    if (matched.at("conv")->stmt()->op_info()->HasAttr("enable_int16") &&
        matched.at("conv")->stmt()->op_info()->GetAttr<bool>("enable_int16")) {
      op_desc.SetAttr<bool>("enable_int16", true);
      op_desc.SetAttr<std::vector<float>>(
          "Input0_scale",
          {((2 << 15) - 1) *
           matched.at("conv")->stmt()->op_info()->GetInputScale(
               input_name)[0]});

      op_desc.SetAttr<std::vector<float>>(
          "Filter0_scale",
          {((2 << 15) - 1) *
           matched.at("conv")->stmt()->op_info()->GetInputScale(
               filter_name)[0]});
    }

    auto conv_op = LiteOpRegistry::Global().Create("__xpu__conv2d");
    auto& valid_places = conv_old->valid_places();
    conv_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(conv_op, valid_places);
    DirectedLink(matched.at("input"), new_op_node);
    DirectedLink(matched.at("conv_filter"), new_op_node);
    DirectedLink(new_op_node, max_output_node);
    if (with_bn_ || with_conv_bias_) {
      DirectedLink(fusion_bias_node, new_op_node);
    }
    if (with_branch_) {
      DirectedLink(matched.at("ew_branch_add_in"), new_op_node);
    }
    DirectedLink(new_op_node, matched.at(output_node_name));
  }

 private:
  std::string conv_type_;
  std::string act_type_;
  bool with_conv_bias_;
  bool with_bn_;
  bool with_branch_;
  bool with_branch_x_;
  bool with_branch_y_;
};

}  // namespace fusion

class XPUConv2dFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    for (auto conv_type : {"conv2d", "depthwise_conv2d"}) {
      for (auto with_branch_x : {true, false}) {
        for (auto with_branch_y : {true, false}) {
          for (auto with_conv_bias : {true, false}) {
            for (auto with_bn : {true, false}) {
              for (auto act_type : {"relu",
                                    "sigmoid",
                                    "tanh",
                                    "leaky_relu",
                                    "hard_swish",
                                    "hard_sigmoid",
                                    "relu6",
                                    "swish",
                                    "linear"}) {
                if (with_branch_x && with_branch_y) continue;
                fusion::XPUConv2dFuser fuser(conv_type,
                                             act_type,
                                             with_conv_bias,
                                             with_bn,
                                             with_branch_x,
                                             with_branch_y);
                fuser(graph.get());
              }
            }
          }
        }
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__conv2d_fuse_pass, paddle::lite::mir::XPUConv2dFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__conv2d");
