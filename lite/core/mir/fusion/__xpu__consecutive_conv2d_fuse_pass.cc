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
/* fuse xpu_conv2d * 3 as xpu_block          */
/* graph                                     */
/*                in_Input                   */
/*                    |                      */
/*                    |                      */
/*                __xpu__conv2d              */
/*                    |                      */
/*                    |                      */
/*                __xpu__conv2d              */
/*                    |                      */
/*                    |                      */
/*                __xpu__conv2d              */
/*                    |                      */
/*                    |                      */
/*                out_Output                 */
/*-------------------------------------------*/
/* After the pass is applied:                */
/*                  in_Input                 */
/*     in_Filter      |     in_FilterMax     */
/*               \    |    /                 */
/*                \   |   /                  */
/*  in_Bias ------- __xpu__block_fuse        */
/*                    |    \                 */
/*                    |     \                */
/*                    |      out_OutputMax   */
/*              out_Output                   */
/*                                           */

#define STR1(R) #R
#define STR2(R) STR1(R)

#define CONV_PATTERN(num)                                                      \
  auto* weight_##num = VarNode(STR2(weight_##num))                             \
                           ->assert_is_op_input("__xpu__conv2d", "Filter")     \
                           ->AsIntermediate();                                 \
  auto* bias_##num = VarNode(STR2(bias_##num))                                 \
                         ->assert_is_op_input("__xpu__conv2d", "Bias")         \
                         ->AsIntermediate();                                   \
  auto* conv_##num = OpNode(STR2(conv_##num), "__xpu__conv2d")                 \
                         ->assert_op_attr<bool>("has_branch", false)           \
                         ->assert_op_attr<bool>("has_bias", true)              \
                         ->AsIntermediate();                                   \
  auto* conv_out_##num = VarNode(STR2(conv_out_##num))                         \
                             ->assert_is_op_output("__xpu__conv2d", "Output"); \
  auto* conv_out_max_##num =                                                   \
      VarNode(STR2(conv_out_max_##num))                                        \
          ->assert_is_op_output("__xpu__conv2d", "OutputMax");

#define CONV_CONNECT(num)       \
  *weight_##num >> *conv_##num; \
  *bias_##num >> *conv_##num;   \
  *conv_##num >> *conv_out_max_##num;

class XPUConsecutiveConv2dFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__conv2d", "Input")
                      ->AsInput();
    CONV_PATTERN(0);
    CONV_PATTERN(1);
    CONV_PATTERN(2);
    conv_out_0->assert_is_op_input("__xpu__conv2d", "Input")->AsIntermediate();
    conv_out_max_0->AsIntermediate();
    conv_out_1->assert_is_op_input("__xpu__conv2d", "Input")->AsIntermediate();
    conv_out_max_1->AsIntermediate();
    conv_out_2->AsOutput();
    conv_out_max_2->AsOutput();
    *input >> *conv_0 >> *conv_out_0 >> *conv_1 >> *conv_out_1 >> *conv_2 >>
        *conv_out_2;
    CONV_CONNECT(0);
    CONV_CONNECT(1);
    CONV_CONNECT(2);
  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::vector<std::string> conv_name;
    std::vector<std::string> filter_name;
    std::vector<std::string> bias_name;
    for (int i = 0; i < 3; i++) {
      std::string cur_conv_name = "conv_" + std::to_string(i);
      std::string cur_weight_name = "weight_" + std::to_string(i);
      std::string cur_bias_name = "bias_" + std::to_string(i);
      conv_name.push_back(cur_conv_name);
      filter_name.push_back(matched.at(cur_weight_name)->arg()->name);
      bias_name.push_back(matched.at(cur_bias_name)->arg()->name);
    }
    cpp::OpDesc op_desc;
    auto conv_0 = matched.at("conv_0")->stmt()->op();
    auto* scope = conv_0->scope();
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("__xpu__block_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetOutput("Output", {matched.at("conv_out_2")->arg()->name});
    op_desc.SetOutput("OutputMax", {matched.at("conv_out_max_2")->arg()->name});

    std::vector<int> place_x{0, 0, 0};
    std::vector<int> place_y{9, 9, 9};
    std::vector<int> place_z{10, 10, 10};
    std::vector<int> block_lod{1, 1, 1};
    std::vector<int> op_type(3, 0);
    std::vector<int> filter_dims;
    std::vector<int> conv_strides;
    std::vector<int> conv_paddings;
    std::vector<int> conv_dilations;
    std::vector<int> conv_groups;
    std::vector<int> act_type;
    std::vector<float> act_param;
    std::vector<int> encode_filter_size{0};
    std::vector<int> encode_bias_size{0};
    std::vector<int> conv_bias{1, 1, 1};
    for (auto name : conv_name) {
      auto cur_filter_dims =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "filter_dims");
      auto cur_strides =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "strides");
      auto cur_paddings =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "paddings");
      auto cur_dilations =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "dilations");
      auto cur_groups =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "groups");
      auto cur_act_type =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "act_type");
      auto cur_act_param =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<float>>(
              "act_param");
      filter_dims.insert(
          filter_dims.end(), cur_filter_dims.begin(), cur_filter_dims.end());
      encode_filter_size.push_back(encode_filter_size.back() +
                                   cur_filter_dims[0] * cur_filter_dims[1] *
                                       cur_filter_dims[2] * cur_filter_dims[3]);
      encode_bias_size.push_back(encode_bias_size.back() + cur_filter_dims[0]);
      conv_strides.insert(
          conv_strides.end(), cur_strides.begin(), cur_strides.end());
      if (cur_paddings.size() == 2) {
        for (size_t i = 0; i < cur_strides.size(); ++i) {
          int copy_pad = *(cur_paddings.begin() + 2 * i);
          cur_paddings.insert(cur_paddings.begin() + 2 * i + 1, copy_pad);
        }
      }
      CHECK_EQ(cur_paddings.size(), 4UL)
          << "Paddings size should be 2 or 4, But received paddings size: "
          << cur_paddings.size();
      conv_paddings.insert(
          conv_paddings.end(), cur_paddings.begin(), cur_paddings.end());
      conv_dilations.insert(
          conv_dilations.end(), cur_dilations.begin(), cur_dilations.end());
      conv_groups.push_back(cur_groups[0]);
      act_type.push_back(cur_act_type[0]);
      act_param.push_back(cur_act_param[0]);
    }
    op_desc.SetAttr("op_type", op_type);
    op_desc.SetAttr("place_x", place_x);
    op_desc.SetAttr("place_y", place_y);
    op_desc.SetAttr("place_z", place_z);
    op_desc.SetAttr("filter_dims", filter_dims);
    op_desc.SetAttr("strides", conv_strides);
    op_desc.SetAttr("paddings", conv_paddings);
    op_desc.SetAttr("dilations", conv_dilations);
    op_desc.SetAttr("groups", conv_groups);
    op_desc.SetAttr("act_type", act_type);
    op_desc.SetAttr("act_param", act_param);
    op_desc.SetAttr("block_lod", block_lod);
    op_desc.SetAttr("conv_bias", conv_bias);
    op_desc.SetAttr<bool>("has_bias", true);
    op_desc.SetAttr<bool>("has_branch", false);

    std::unique_ptr<float[]> encode_filter_float(
        new float[encode_filter_size.back()]);
    for (int i = 0; i < filter_name.size(); i++) {
      auto* filter_t = scope->FindMutableTensor(filter_name[i]);
      float* filter_on_host = filter_t->mutable_data<float>();
      memcpy(
          encode_filter_float.get() + encode_filter_size[i],
          filter_on_host,
          (encode_filter_size[i + 1] - encode_filter_size[i]) * sizeof(float));
    }
    std::string new_filter_name = "block_" + filter_name[0];
    auto* new_filter_node = graph->NewArgumentNode(new_filter_name);
    new_filter_node->arg()->is_weight = true;
    new_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_filter_t = scope->NewTensor(new_filter_name);
    new_filter_t->set_precision(paddle::lite_api::PrecisionType::kFloat);
    new_filter_t->set_persistable(true);
    new_filter_t->Resize({encode_filter_size.back()});
    float* new_filter_ptr = new_filter_t->mutable_data<float>();
    memcpy(new_filter_ptr,
           encode_filter_float.get(),
           encode_filter_size.back() * sizeof(float));
    op_desc.SetInput("Filter", {new_filter_name});

    std::unique_ptr<float[]> encode_bias(new float[encode_bias_size.back()]);
    for (int i = 0; i < bias_name.size(); i++) {
      auto* bias_t = scope->FindMutableTensor(bias_name[i]);
      float* bias_on_host = bias_t->mutable_data<float>();
      memcpy(encode_bias.get() + encode_bias_size[i],
             bias_on_host,
             (encode_bias_size[i + 1] - encode_bias_size[i]) * sizeof(float));
    }
    std::string new_bias_name = "block_" + bias_name[0];
    auto* new_bias_node = graph->NewArgumentNode(new_bias_name);
    new_bias_node->arg()->is_weight = true;
    new_bias_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_bias_t = scope->NewTensor(new_bias_name);
    new_bias_t->set_precision(paddle::lite_api::PrecisionType::kFloat);
    new_bias_t->set_persistable(true);
    new_bias_t->Resize({encode_bias_size.back()});
    float* new_bias_ptr = new_bias_t->mutable_data<float>();
    memcpy(new_bias_ptr,
           encode_bias.get(),
           encode_bias_size.back() * sizeof(float));
    op_desc.SetInput("Bias", {new_bias_name});

    auto& valid_places = conv_0->valid_places();
    auto block_op = LiteOpRegistry::Global().Create(op_desc.Type());
    block_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(block_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    IR_NODE_LINK_TO(new_bias_node, new_op_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_2"));
    IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_max_2"));
  }
};
#undef CONV_CONNECT
#undef CONV_PATTERN
#undef STR1
#undef STR2
}  // namespace fusion

class XPUConsecutiveConv2dFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUConsecutiveConv2dFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__consecutive_conv2d_fuse_pass,
                  paddle::lite::mir::XPUConsecutiveConv2dFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");
