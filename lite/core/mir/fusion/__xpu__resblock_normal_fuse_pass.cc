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

/* fuse 2 or 3xpu_conv2d op as resnet50-like block */
/* graph[1]: has_mid_conv = true                   */
/*                in_Input                         */
/*                /      \                         */
/*              /          \                       */
/*             |            |                      */
/*         xpu_conv2d       |                      */
/*             |            |                      */
/*             |            |                      */
/*         xpu_conv2d       |                      */
/*             |            |                      */
/*              \          /                       */
/*                \       /                        */
/*                xpu_conv2d                       */
/*               (with branch)                     */
/*                    |                            */
/*                    |                            */
/*                out_Output                       */
/*-------------------------------------------------*/
/* graph[2]: has_mid_conv = false                  */
/*                in_Input                         */
/*                /      \                         */
/*              /          \                       */
/*             |            |                      */
/*             |            |                      */
/*             |            |                      */
/*         xpu_conv2d       |                      */
/*             |            |                      */
/*              \          /                       */
/*                \       /                        */
/*                xpu_conv2d                       */
/*               (with branch)                     */
/*                    |                            */
/*                    |                            */
/*                out_Output                       */
/*-------------------------------------------------*/
/* After the pass is applied:                      */
/*                     in_Input                    */
/*        in_Filter      |     in_FilterMax        */
/*                  \    |    /                    */
/*                   \   |   /                     */
/*     in_Bias ------- __xpu__block_fuse           */
/*                       |    \                    */
/*                       |     \                   */
/*                       |      out_OutputMax      */
/*                 out_Output                      */
/*                                                 */

class XPUResBlockNormalFuser : public FuseBase {
 public:
  explicit XPUResBlockNormalFuser(bool has_mid_conv) {
    has_mid_conv_ = has_mid_conv;
  }
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__conv2d", "Input")
                      ->assert_is_op_input("__xpu__conv2d", "Branch")
                      ->AsInput();
    auto* left_conv1_weight =
        VarNode("left_conv1_weight")
            ->assert_is_op_input("__xpu__conv2d", "Filter")
            ->assert_is_persistable_var()
            ->AsIntermediate();
    auto* left_conv1_weight_max =
        VarNode("left_conv1_weight_max")
            ->assert_is_op_input("__xpu__conv2d", "FilterMax")
            ->assert_is_persistable_var()
            ->AsIntermediate();
    auto* left_conv1_bias = VarNode("left_conv1_bias")
                                ->assert_is_op_input("__xpu__conv2d", "Bias")
                                ->assert_is_persistable_var()
                                ->AsIntermediate();
    auto* left_conv1 = OpNode("left_conv1", "__xpu__conv2d")
                           ->assert_op_attr<bool>("has_branch", false)
                           ->AsIntermediate();
    auto* left_conv1_out = VarNode("left_conv1_out")
                               ->assert_is_op_output("__xpu__conv2d", "Output")
                               ->assert_is_op_input("__xpu__conv2d", "Input")
                               ->AsIntermediate();
    auto* left_conv1_out_max =
        VarNode("left_conv1_out_max")
            ->assert_is_op_output("__xpu__conv2d", "OutputMax")
            ->AsIntermediate();
    PMNode* left_conv2_weight = nullptr;
    PMNode* left_conv2_weight_max = nullptr;
    PMNode* left_conv2_bias = nullptr;
    PMNode* left_conv2 = nullptr;
    PMNode* left_conv2_out = nullptr;
    PMNode* left_conv2_out_max = nullptr;
    if (has_mid_conv_) {
      left_conv2_weight = VarNode("left_conv2_weight")
                              ->assert_is_op_input("__xpu__conv2d", "Filter")
                              ->assert_is_persistable_var()
                              ->AsIntermediate();
      left_conv2_weight_max =
          VarNode("left_conv2_weight_max")
              ->assert_is_op_input("__xpu__conv2d", "FilterMax")
              ->assert_is_persistable_var()
              ->AsIntermediate();
      left_conv2_bias = VarNode("left_conv2_bias")
                            ->assert_is_op_input("__xpu__conv2d", "Bias")
                            ->assert_is_persistable_var()
                            ->AsIntermediate();
      left_conv2 = OpNode("left_conv2", "__xpu__conv2d")
                       ->assert_op_attr<bool>("has_branch", false)
                       ->AsIntermediate();
      left_conv2_out = VarNode("left_conv2_out")
                           ->assert_is_op_output("__xpu__conv2d", "Output")
                           ->assert_is_op_input("__xpu__conv2d", "Input")
                           ->AsIntermediate();
      left_conv2_out_max =
          VarNode("left_conv2_out_max")
              ->assert_is_op_output("__xpu__conv2d", "OutputMax")
              ->AsIntermediate();
    }
    auto* left_conv3_weight =
        VarNode("left_conv3_weight")
            ->assert_is_persistable_var()
            ->assert_is_op_input("__xpu__conv2d", "Filter")
            ->AsIntermediate();
    auto* left_conv3_weight_max =
        VarNode("left_conv3_weight_max")
            ->assert_is_op_input("__xpu__conv2d", "FilterMax")
            ->assert_is_persistable_var()
            ->AsIntermediate();
    auto* left_conv3_bias = VarNode("left_conv3_bias")
                                ->assert_is_op_input("__xpu__conv2d", "Bias")
                                ->assert_is_persistable_var()
                                ->AsIntermediate();
    auto* left_conv3 = OpNode("left_conv3", "__xpu__conv2d")
                           ->assert_op_attr<bool>("has_branch", true)
                           ->AsIntermediate();
    auto* left_conv3_out = VarNode("left_conv3_out")
                               ->assert_is_op_output("__xpu__conv2d", "Output")
                               ->AsOutput();
    auto* left_conv3_out_max =
        VarNode("left_conv3_out_max")
            ->assert_is_op_output("__xpu__conv2d", "OutputMax")
            ->AsOutput();

    if (has_mid_conv_) {
      *input >> *left_conv1 >> *left_conv1_out >> *left_conv2 >>
          *left_conv2_out >> *left_conv3;
      *input >> *left_conv3;
      *left_conv3 >> *left_conv3_out;
      *left_conv2_weight >> *left_conv2;
      *left_conv2_weight_max >> *left_conv2;
      *left_conv2_bias >> *left_conv2;
      *left_conv2 >> *left_conv2_out_max;
    } else {
      *input >> *left_conv1 >> *left_conv1_out >> *left_conv3;
      *input >> *left_conv3;
      *left_conv3 >> *left_conv3_out;
    }
    *left_conv1_weight >> *left_conv1;
    *left_conv1_weight_max >> *left_conv1;
    *left_conv1_bias >> *left_conv1;
    *left_conv1 >> *left_conv1_out_max;
    *left_conv3_weight >> *left_conv3;
    *left_conv3_weight_max >> *left_conv3;
    *left_conv3_bias >> *left_conv3;
    *left_conv3 >> *left_conv3_out_max;
  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::vector<std::string> conv_name{"left_conv1", "left_conv3"};
    std::vector<std::string> filter_name{
        matched.at("left_conv1_weight")->arg()->name,
        matched.at("left_conv3_weight")->arg()->name};
    std::vector<std::string> bias_name = {
        matched.at("left_conv1_bias")->arg()->name,
        matched.at("left_conv3_bias")->arg()->name};
    std::vector<std::string> filter_max_name{
        matched.at("left_conv1_weight_max")->arg()->name,
        matched.at("left_conv3_weight_max")->arg()->name};
    if (has_mid_conv_) {
      conv_name.insert(conv_name.begin() + 1, "left_conv2");
      filter_name.insert(filter_name.begin() + 1,
                         matched.at("left_conv2_weight")->arg()->name);
      bias_name.insert(bias_name.begin() + 1,
                       matched.at("left_conv2_bias")->arg()->name);
      filter_max_name.insert(filter_max_name.begin() + 1,
                             matched.at("left_conv2_weight_max")->arg()->name);
    }

    cpp::OpDesc op_desc;
    auto left_conv1 = matched.at("left_conv1")->stmt()->op();
    auto* scope = left_conv1->scope();

    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();

    op_desc.SetType("__xpu__block_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetOutput("Output", {matched.at("left_conv3_out")->arg()->name});
    op_desc.SetOutput("OutputMax",
                      {matched.at("left_conv3_out_max")->arg()->name});

    std::vector<int> op_type;
    std::vector<int> place_x;
    std::vector<int> place_y;
    std::vector<int> place_z;
    std::vector<int> block_lod;
    if (has_mid_conv_) {
      op_type = {0, 0, 0};
      place_x = {0, 1, 2};
      place_y = {9, 9, 0};
      place_z = {1, 2, 10};
      block_lod = {3};
    } else {
      op_type = {0, 0};
      place_x = {0, 1};
      place_y = {9, 0};
      place_z = {1, 10};
      block_lod = {2};
    }
    std::vector<int> filter_dims;
    std::vector<int> conv_strides;
    std::vector<int> conv_paddings;
    std::vector<int> conv_dilations;
    std::vector<int> conv_groups;
    std::vector<int> act_type;
    std::vector<float> act_param;
    std::vector<int> encode_filter_size{0};
    std::vector<int> encode_bias_size{0};
    std::vector<int> encode_filter_max_size{0};
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
          matched.at(name)->stmt()->op_info()->GetAttr<int>("groups");
      auto cur_act_type =
          matched.at(name)->stmt()->op_info()->GetAttr<int>("act_type");
      auto cur_act_param =
          matched.at(name)->stmt()->op_info()->GetAttr<float>("act_param");
      filter_dims.insert(
          filter_dims.end(), cur_filter_dims.begin(), cur_filter_dims.end());
      encode_filter_size.push_back(encode_filter_size.back() +
                                   cur_filter_dims[0] * cur_filter_dims[1] *
                                       cur_filter_dims[2] * cur_filter_dims[3]);
      encode_bias_size.push_back(encode_bias_size.back() + cur_filter_dims[0]);
      encode_filter_max_size.push_back(encode_filter_max_size.back() + 4);
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
      conv_groups.push_back(cur_groups);
      act_type.push_back(cur_act_type);
      act_param.push_back(cur_act_param);
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

    std::unique_ptr<int16_t[]> encode_filter_int16(
        new int16_t[encode_filter_size.back()]);
    for (int i = 0; i < filter_name.size(); i++) {
      auto* filter_t = scope->FindMutableTensor(filter_name[i]);
      int16_t* filter_on_host = filter_t->mutable_data<int16_t>();
      memcpy(encode_filter_int16.get() + encode_filter_size[i],
             filter_on_host,
             (encode_filter_size[i + 1] - encode_filter_size[i]) *
                 sizeof(int16_t));
    }
    std::string new_filter_name = "block_" + filter_name[0];
    auto* new_filter_node = graph->NewArgumentNode(new_filter_name);
    new_filter_node->arg()->is_weight = true;
    new_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kInt16), DATALAYOUT(kNCHW));
    auto* new_filter_t = scope->NewTensor(new_filter_name);
    new_filter_t->Resize({encode_filter_size.back()});
    int16_t* new_filter_ptr = new_filter_t->mutable_data<int16_t>();
    memcpy(new_filter_ptr,
           encode_filter_int16.get(),
           encode_filter_size.back() * sizeof(int16_t));
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
    new_bias_t->Resize({encode_bias_size.back()});
    float* new_bias_ptr = new_bias_t->mutable_data<float>();
    memcpy(new_bias_ptr,
           encode_bias.get(),
           encode_bias_size.back() * sizeof(float));
    op_desc.SetInput("Bias", {new_bias_name});

    std::unique_ptr<float[]> encode_filter_max(
        new float[encode_filter_max_size.back()]);
    for (int i = 0; i < filter_max_name.size(); i++) {
      auto* filter_max_t = scope->FindMutableTensor(filter_max_name[i]);
      float* filter_max_on_host = filter_max_t->mutable_data<float>();
      memcpy(encode_filter_max.get() + encode_filter_max_size[i],
             filter_max_on_host,
             (encode_filter_max_size[i + 1] - encode_filter_max_size[i]) *
                 sizeof(float));
    }
    std::string new_filter_max_name = "block_" + filter_max_name[0];
    auto* new_filter_max_node = graph->NewArgumentNode(new_filter_max_name);
    new_filter_max_node->arg()->is_weight = true;
    new_filter_max_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_filter_max_t = scope->NewTensor(new_filter_max_name);
    new_filter_max_t->Resize({encode_filter_max_size.back()});
    float* new_filter_max_ptr = new_filter_max_t->mutable_data<float>();
    memcpy(new_filter_max_ptr,
           encode_filter_max.get(),
           encode_filter_max_size.back() * sizeof(float));
    op_desc.SetInput("FilterMax", {new_filter_max_name});

    auto& valid_places = left_conv1->valid_places();
    auto resblock_normal_op = LiteOpRegistry::Global().Create(op_desc.Type());
    resblock_normal_op->Attach(op_desc, scope);
    auto* new_op_node =
        graph->GraphCreateInstructNode(resblock_normal_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    IR_NODE_LINK_TO(new_filter_max_node, new_op_node);
    IR_NODE_LINK_TO(new_bias_node, new_op_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("left_conv3_out"));
    IR_NODE_LINK_TO(new_op_node, matched.at("left_conv3_out_max"));
  }

 private:
  bool has_mid_conv_;
};

}  // namespace fusion

class XPUResBlockNormalFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUResBlockNormalFuser fuser1(true);
    fuser1(graph.get());
    fusion::XPUResBlockNormalFuser fuser2(false);
    fuser2(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__resblock_normal_fuse_pass,
                  paddle::lite::mir::XPUResBlockNormalFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");
