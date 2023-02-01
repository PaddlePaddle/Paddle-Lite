// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
/* Fuse conv2d scale block model to xpu_conv2d op one more step      */
/* For example:                                                      */
/* graph: sub block                                                  */
/*                           in_Input                                */
/*                     _______|________________________              */
/*   xpu__conv2d <----|       |                        |             */
/*                    |     conv2d----in_Filter        |             */
/*                    |       |                        |             */
/*                    |       |                        |             */
/*                    |  elementwise_add-----conv_Bias |             */
/*                    |       |                        |             */
/*                    |       |                        |             */
/*                    | batch_norm------in_Bias        |             */
/*                    |_______|____\___________________|             */
/*                            |     \                                */
/*                            |     scale                            */
/*                            |     /                                */
/*                            |    /                                 */
/*                          concat                                   */
/*                            |                                      */
/*                            |                                      */
/*                     elementwise_mul -----in_EwMulY                */
/*                            |                                      */
/*                            |                                      */
/*                     elementwise_add -----in_EwAddY                */
/*                            |                                      */
/*                            |                                      */
/*                           relu                                    */
/*                            |                                      */
/*                            |                                      */
/*                        out_Out                                    */
/*                                                                   */
/* After the pass is applied:                                        */
/*                          in_Input                                 */
/*             in_Filter      |     in_FilterMax                     */
/*                       \    |    /                                 */
/*                        \   |   /                                  */
/*          in_Bias ------- __xpu__conv2d ------ in_EwMulY           */
/*                        /   |    \                                 */
/*                       /    |     \                                */
/*          out_OutputMax     |      in_EwAddY                       */
/*                      out_Output                                   */
/*                                                                   */
/* ----------------------------------------------------------------- */

class XPUConv2dScaleFuser : public FuseBase {
 public:
  explicit XPUConv2dScaleFuser(const std::string& act_type) {
    act_type_ = act_type;
  }

  void BuildPattern() override {
    auto non_branch_teller = [](const Node* node) -> bool {
      auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
      return (!op_desc.HasAttr("has_branch") ||
              !op_desc.GetAttr<bool>("has_branch"));
    };

    auto non_act_teller = [](const Node* node) -> bool {
      auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
      return (!op_desc.HasAttr("act_type") ||
              op_desc.GetAttr<std::vector<int>>("act_type")[0] == 0);
    };

    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__conv2d", "Input")
                      ->AsInput();
    auto* conv_filter = VarNode("conv_filter")
                            ->assert_is_op_input("__xpu__conv2d", "Filter")
                            ->AsInput();
    auto* conv_bias = VarNode("conv_bias")
                          ->assert_is_op_input("__xpu__conv2d", "Bias")
                          ->AsInput();
    auto* conv2d = OpNode("conv2d", "__xpu__conv2d")
                       ->assert_node_satisfied(non_branch_teller)
                       ->assert_node_satisfied(non_act_teller)
                       ->AsIntermediate();
    auto* conv_out = VarNode("conv_out")
                         ->assert_is_op_output("__xpu__conv2d", "Output")
                         ->assert_is_op_input("scale", "X")
                         ->AsIntermediate()
                         ->assert_is_op_input("concat", "X")
                         ->AsIntermediate();
    auto* max_output = VarNode("max_output")
                           ->assert_is_op_output("__xpu__conv2d", "OutputMax");

    // for scale
    auto* scale = OpNode("scale", "scale")->AsIntermediate();
    auto* scale_out = VarNode("scale_out")
                          ->assert_is_op_input("concat")
                          ->assert_is_op_output("scale", "Out")
                          ->AsIntermediate();

    // for concat
    auto* concat = OpNode("concat", "concat")->AsIntermediate();
    auto* concat_out = VarNode("concat_out")
                           ->assert_is_op_input("elementwise_mul", "X")
                           ->assert_is_op_output("concat", "Out")
                           ->AsIntermediate();

    // for ele mul&add
    auto* ew_mul = OpNode("ew_mul", "elementwise_mul")->AsIntermediate();
    auto* ew_mul_y = VarNode("ew_mul_y")
                         ->assert_is_op_input("elementwise_mul", "Y")
                         ->assert_is_persistable_var()
                         ->assert_only_one_output()
                         ->AsIntermediate();
    auto* ew_mul_out = VarNode("ew_mul_out")
                           ->assert_is_op_input("elementwise_add", "X")
                           ->assert_is_op_output("elementwise_mul", "Out")
                           ->AsIntermediate();

    auto* ew_add = OpNode("ew_add", "elementwise_add")->AsIntermediate();
    auto* ew_add_y = VarNode("ew_add_y")
                         ->assert_is_op_input("elementwise_add", "Y")
                         ->assert_is_persistable_var()
                         ->assert_only_one_output()
                         ->AsIntermediate();
    auto* ew_add_out = VarNode("ew_add_out")
                           ->assert_is_op_input("relu", "X")
                           ->assert_is_op_output("elementwise_add", "Out")
                           ->AsIntermediate();

    // act
    PMNode* act = OpNode("act", act_type_)->AsIntermediate();
    PMNode* act_out = VarNode("act_out")->assert_is_op_output(act_type_, "Out");

    // pass
    *input >> *conv2d >> *conv_out;
    *conv2d >> *max_output;
    *conv_filter >> *conv2d;
    *conv_bias >> *conv2d;
    *conv_out >> *scale >> *scale_out;

    // for concat pass
    *scale_out >> *concat >> *concat_out;
    *conv_out >> *concat;

    // for ele mul&add pass
    *concat_out >> *ew_mul >> *ew_mul_out;
    *ew_mul_y >> *ew_mul;

    *ew_mul_out >> *ew_add >> *ew_add_out;
    *ew_add_y >> *ew_add;

    // for act pass
    *ew_add_out >> *act >> *act_out;
    act_out->AsOutput();
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__conv2d");

    auto conv_old = matched.at("conv2d")->stmt()->op();
    auto* scope = conv_old->scope();
    std::string input_name = matched.at("input")->arg()->name;
    op_desc.SetInput("Input", {input_name});

    auto* scale_op_info = matched.at("scale")->stmt()->op_info();
    float scale_val = scale_op_info->GetAttr<float>("scale");
    float scale_bias = scale_op_info->GetAttr<float>("bias");
    bool bias_after_scale = scale_op_info->GetAttr<bool>("bias_after_scale");

    auto filter_name = matched.at("conv_filter")->arg()->name;
    auto* filter_old_tensor = scope->FindMutableTensor(filter_name);
    auto& filter_old_dims = filter_old_tensor->dims();
    std::vector<int> filter_dims{static_cast<int>(filter_old_dims[0] * 2),
                                 static_cast<int>(filter_old_dims[1]),
                                 static_cast<int>(filter_old_dims[2]),
                                 static_cast<int>(filter_old_dims[3])};

    std::string fusion_filter_name = filter_name + "_conv_scale_fusion_filter";
    auto* fusion_filter_tensor =
        scope->MutableParent()->NewTensor(fusion_filter_name);
    fusion_filter_tensor->Resize(
        {filter_dims[0], filter_dims[1], filter_dims[2], filter_dims[3]});
    fusion_filter_tensor->set_persistable(true);
    fusion_filter_tensor->set_precision(
        paddle::lite_api::PrecisionType::kFloat);

    int filter_old_len = filter_old_tensor->numel();
    memcpy(fusion_filter_tensor->mutable_data<float>(),
           filter_old_tensor->mutable_data<float>(),
           filter_old_len * sizeof(float));

    auto* filter_in = filter_old_tensor->mutable_data<float>();
    auto* filter_out =
        fusion_filter_tensor->mutable_data<float>() + filter_old_len;
    for (int i = 0; i < filter_old_len; ++i) {
      filter_out[i] = scale_val * filter_in[i];
    }

    auto* fusion_filter_node = graph->NewArgumentNode(fusion_filter_name);
    fusion_filter_node->arg()->is_weight = true;
    fusion_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    op_desc.SetInput("Filter", {fusion_filter_name});

    //// for conv bias
    auto bias_name = matched.at("conv_bias")->arg()->name;
    auto* bias_old_t = scope->FindMutableTensor(bias_name);
    auto& bias_old_dims = bias_old_t->dims();
    std::vector<int> bias_dims{static_cast<int>(bias_old_dims[0] * 2)};

    std::string fusion_bias_name = bias_name + "_conv_scale_fusion_filter";
    auto* fusion_bias_tensor =
        scope->MutableParent()->NewTensor(fusion_bias_name);
    fusion_bias_tensor->set_precision(paddle::lite_api::PrecisionType::kFloat);
    fusion_bias_tensor->Resize({bias_dims[0]});
    fusion_bias_tensor->set_persistable(true);
    fusion_bias_tensor->set_precision(paddle::lite_api::PrecisionType::kFloat);

    int bias_old_len = bias_old_t->numel();
    memcpy(fusion_bias_tensor->mutable_data<float>(),
           bias_old_t->mutable_data<float>(),
           bias_old_len * sizeof(float));

    auto* bias_in = bias_old_t->mutable_data<float>();
    auto* bias_out = fusion_bias_tensor->mutable_data<float>() + bias_old_len;
    for (int i = 0; i < bias_old_len; ++i) {
      if (bias_after_scale)
        bias_out[i] = bias_in[i] * scale_val + scale_bias;
      else
        bias_out[i] = (bias_in[i] + scale_bias) * scale_val;
    }

    auto* fusion_bias_node = graph->NewArgumentNode(fusion_bias_name);
    fusion_bias_node->arg()->is_weight = true;
    fusion_bias_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    op_desc.SetInput("Bias", {fusion_bias_name});

    // for conv other info
    std::vector<int> conv_groups =
        matched.at("conv2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "groups");

    op_desc.SetAttr<bool>("has_bias", true);
    op_desc.SetAttr<bool>("has_branch", false);
    op_desc.SetAttr<std::vector<int>>("filter_dims", filter_dims);
    op_desc.SetAttr<std::vector<int>>(
        "op_type",
        matched.at("conv2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "op_type"));
    op_desc.SetAttr<std::vector<int>>(
        "place_x",
        matched.at("conv2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "place_x"));
    op_desc.SetAttr<std::vector<int>>(
        "place_y",
        matched.at("conv2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "place_y"));
    op_desc.SetAttr<std::vector<int>>(
        "place_z",
        matched.at("conv2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "place_z"));
    op_desc.SetAttr<std::vector<int>>(
        "strides",
        matched.at("conv2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "strides"));
    op_desc.SetAttr<std::vector<int>>(
        "paddings",
        matched.at("conv2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "paddings"));
    op_desc.SetAttr<std::vector<int>>(
        "dilations",
        matched.at("conv2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "dilations"));
    op_desc.SetAttr<std::vector<int>>(
        "groups",
        matched.at("conv2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "groups"));
    op_desc.SetAttr<std::vector<int>>(
        "block_lod",
        matched.at("conv2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "block_lod"));
    op_desc.SetAttr<std::vector<int>>(
        "conv_bias",
        matched.at("conv2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "conv_bias"));

    // for ele_mul&&ele_add
    auto ele_mul_y_name = matched.at("ew_mul_y")->arg()->name;
    auto* ele_mul_t = scope->FindMutableTensor(ele_mul_y_name);
    float* ele_mul_y_on_host = ele_mul_t->mutable_data<float>();

    auto ele_add_y_name = matched.at("ew_add_y")->arg()->name;
    auto* ele_add_y_t = scope->FindMutableTensor(ele_add_y_name);
    float* ele_add_y_on_host = ele_add_y_t->mutable_data<float>();

    int ele_mul_len = ele_mul_t->numel();
    int filter_len = fusion_filter_tensor->numel();
    int filter_stride = filter_len / ele_mul_len;

    float* filter_on_host = fusion_filter_tensor->mutable_data<float>();
    for (int i = 0; i < ele_mul_len; ++i) {
      for (int j = 0; j < filter_stride; ++j) {
        filter_on_host[i * filter_stride + j] *= ele_mul_y_on_host[i];
      }
    }

    float* fusion_bias_ptr = fusion_bias_tensor->mutable_data<float>();
    for (int i = 0; i < ele_mul_len; ++i) {
      fusion_bias_ptr[i] *= ele_mul_y_on_host[i];
      fusion_bias_ptr[i] += ele_add_y_on_host[i];
    }

    std::map<std::string, int> act_map{{"linear", 0},
                                       {"relu", 1},
                                       {"sigmoid", 2},
                                       {"tanh", 3},
                                       {"leaky_relu", 5},
                                       {"hard_swish", 14},
                                       {"hard_sigmoid", 15},
                                       {"swish", 16},
                                       {"relu6", 17}};

    float act_param = 0.0f;
    if (act_type_ != "linear") {
      if (act_type_ == "leaky_relu") {
        auto act_op_desc = *matched.at("act")->stmt()->op_info();
        act_param = act_op_desc.GetAttr<float>("alpha");
      } else if (act_type_ == "hard_sigmoid") {
        auto act_op_desc = *matched.at("act")->stmt()->op_info();
        act_param = act_op_desc.GetAttr<float>("slope");
      }
    }
    op_desc.SetAttr<std::vector<int>>("act_type",
                                      std::vector<int>{act_map[act_type_]});
    op_desc.SetAttr<std::vector<float>>("act_param",
                                        std::vector<float>{act_param});

    if ((matched.at("conv2d")->stmt()->op_info()->HasAttr(
            "padding_algorithm"))) {
      op_desc.SetAttr<std::string>(
          "padding_algorithm",
          matched.at("conv2d")->stmt()->op_info()->GetAttr<std::string>(
              "padding_algorithm"));
    }

    std::string output_name = matched.at("act_out")->arg()->name;
    op_desc.SetOutput("Output", {output_name});

    std::string max_output_name = output_name + "_xpu_max";
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
    DirectedLink(fusion_filter_node, new_op_node);
    DirectedLink(fusion_bias_node, new_op_node);
    DirectedLink(new_op_node, max_output_node);
    DirectedLink(new_op_node, matched.at("act_out"));
  }

 private:
  std::string act_type_;
};

}  // namespace fusion

class XPUConv2dScaleFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto act_type : {"relu"}) {  // [TO DO] add more activate algorithm
      fusion::XPUConv2dScaleFuser fuser(act_type);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__conv2d_scale_fuse_pass,
                  paddle::lite::mir::XPUConv2dScaleFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__conv2d");
