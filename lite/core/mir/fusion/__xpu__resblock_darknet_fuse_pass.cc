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
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite_metal {
namespace mir {
namespace fusion {

/* fuse 2 xpu_conv2d and ew_add op as darknet53-like block */
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
/*                  ew_add                         */
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

class XPUResBlockDarknetFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__conv2d", "Input")
                      ->assert_is_op_input("elementwise_add", "X")
                      ->AsInput();

    PMNode* conv1_weight = nullptr;
    PMNode* conv2_weight = nullptr;

    PMNode* conv1 = nullptr;
    PMNode* conv2 = nullptr;

    PMNode* conv1_bias = nullptr;
    PMNode* conv2_bias = nullptr;

    PMNode* conv1_out = nullptr;
    PMNode* conv2_out = nullptr;

    PMNode* conv1_out_max = nullptr;
    PMNode* conv2_out_max = nullptr;

    // first
    conv1_weight = VarNode("conv1_weight")
                       ->assert_is_op_input("__xpu__conv2d", "Filter")
                       ->assert_is_persistable_var()
                       ->AsIntermediate();
    conv1 = OpNode("conv1", "__xpu__conv2d")
                ->assert_op_attr<bool>("has_branch", false)
                ->assert_op_attr<bool>("has_bias", true)
                ->AsIntermediate();
    conv1_bias = VarNode("conv1_bias")
                     ->assert_is_op_input("__xpu__conv2d", "Bias")
                     ->assert_is_persistable_var()
                     ->AsIntermediate();
    conv1_out = VarNode("conv1_out")
                    ->assert_is_op_output("__xpu__conv2d", "Output")
                    ->AsIntermediate();
    conv1_out_max = VarNode("conv1_out_max")
                        ->assert_is_op_output("__xpu__conv2d", "OutputMax")
                        ->AsIntermediate();

    // second
    conv1_out->assert_is_op_input("__xpu__conv2d", "Input");
    conv2 = OpNode("conv2", "__xpu__conv2d")
                ->assert_op_attr<bool>("has_branch", false)
                ->assert_op_attr<bool>("has_bias", true)
                ->AsIntermediate();
    conv2_weight = VarNode("conv2_weight")
                       ->assert_is_op_input("__xpu__conv2d", "Filter")
                       ->assert_is_persistable_var()
                       ->AsIntermediate();
    conv2_bias = VarNode("conv2_bias")
                     ->assert_is_op_input("__xpu__conv2d", "Bias")
                     ->assert_is_persistable_var()
                     ->AsIntermediate();
    conv2_out = VarNode("conv2_out")
                    ->assert_is_op_output("__xpu__conv2d", "Output")
                    ->AsIntermediate();
    conv2_out_max = VarNode("conv2_out_max")
                        ->assert_is_op_output("__xpu__conv2d", "OutputMax")
                        ->AsIntermediate();
    // third
    conv2_out->assert_is_op_input("elementwise_add", "Y");
    PMNode* ewadd = OpNode("ewadd", "elementwise_add")->AsIntermediate();
    PMNode* out = VarNode("output")
                      ->assert_is_op_output("elementwise_add", "Out")
                      ->AsOutput();

    *conv1_weight >> *conv1;
    *conv1_bias >> *conv1;
    *conv1 >> *conv1_out_max;
    *input >> *conv1 >> *conv1_out >> *conv2 >> *conv2_out >> *ewadd >> *out;
    *input >> *ewadd;
    *conv2_weight >> *conv2;
    *conv2_bias >> *conv2;
    *conv2 >> *conv2_out_max;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::vector<std::string> conv_name{"conv1", "conv2"};
    std::vector<std::string> filter_name{
        matched.at("conv1_weight")->arg()->name,
        matched.at("conv2_weight")->arg()->name};
    std::vector<std::string> bias_name{matched.at("conv1_bias")->arg()->name,
                                       matched.at("conv2_bias")->arg()->name};

    cpp::OpDesc op_desc;
    auto conv1 = matched.at("conv1")->stmt()->op();
    auto* scope = conv1->scope();
    op_desc.SetType("__xpu__block_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetOutput("Output", {matched.at("output")->arg()->name});

    std::string max_output_name = matched.at("output")->arg()->name + "_max";
    auto* max_output_node = graph->NewArgumentNode(max_output_name);
    max_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* max_output_tensor = scope->NewTensor(max_output_name);
    max_output_tensor->set_precision(paddle::lite_metal_api::PrecisionType::kFloat);
    max_output_tensor->set_persistable(true);
    op_desc.SetOutput("OutputMax", {max_output_name});

    std::vector<int> op_type{
        0, 0, 10};  // "__xpu__conv2d", "__xpu__conv2d", "elementwise_add"
    std::vector<int> place_x{0, 1, 2};
    std::vector<int> place_y{9, 9, 0};
    std::vector<int> place_z{1, 2, 10};
    std::vector<int> block_lod{3};

    std::vector<int> conv_bias;
    std::vector<int> filter_dims;
    std::vector<int> conv_strides;
    std::vector<int> conv_paddings;
    std::vector<int> conv_dilations;
    std::vector<int> conv_groups;
    std::vector<int> act_type;
    std::vector<float> act_param;
    std::vector<int> encode_filter_size{0};
    std::vector<int> encode_bias_size{0};
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
      auto cur_conv_bias =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "conv_bias");
      conv_strides.insert(
          conv_strides.end(), cur_strides.begin(), cur_strides.end());
      conv_dilations.insert(
          conv_dilations.end(), cur_dilations.begin(), cur_dilations.end());
      conv_groups.insert(
          conv_groups.end(), cur_groups.begin(), cur_groups.end());
      act_type.insert(act_type.end(), cur_act_type.begin(), cur_act_type.end());
      act_param.insert(
          act_param.end(), cur_act_param.begin(), cur_act_param.end());
      filter_dims.insert(
          filter_dims.end(), cur_filter_dims.begin(), cur_filter_dims.end());
      conv_bias.insert(
          conv_bias.end(), cur_conv_bias.begin(), cur_conv_bias.end());
      if (cur_filter_dims.size() == 4) {
        encode_filter_size.push_back(encode_filter_size.back() +
                                     cur_filter_dims[0] * cur_filter_dims[1] *
                                         cur_filter_dims[2] *
                                         cur_filter_dims[3]);
        if (cur_conv_bias[0] == 1) {
          encode_bias_size.push_back(encode_bias_size.back() +
                                     cur_filter_dims[0]);
        }
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
      } else if (cur_filter_dims.size() == 2) {
        encode_filter_size.push_back(encode_filter_size.back() +
                                     cur_filter_dims[1] * cur_filter_dims[1] /
                                         cur_filter_dims[0] * 2);
        if (cur_conv_bias[0] == 1) {
          encode_bias_size.push_back(encode_bias_size.back() +
                                     cur_filter_dims[1] / cur_filter_dims[0] +
                                     cur_filter_dims[1]);
        }
      } else {
        LOG(FATAL) << "Invalid filter dims: " << cur_filter_dims.size();
      }
    }
    // for ewadd_act_findmax fuse
    act_type.push_back(0);
    act_param.push_back(0.f);
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
    std::string new_filter_name = "resblock_darknet_" + filter_name[0];
    auto* new_filter_node = graph->NewArgumentNode(new_filter_name);
    new_filter_node->arg()->is_weight = true;
    new_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_filter_t = scope->NewTensor(new_filter_name);
    new_filter_t->set_precision(paddle::lite_metal_api::PrecisionType::kFloat);
    new_filter_t->set_persistable(true);
    new_filter_t->Resize({encode_filter_size.back()});
    float* new_filter_ptr = new_filter_t->mutable_data<float>();
    memcpy(new_filter_ptr,
           encode_filter_float.get(),
           encode_filter_size.back() * sizeof(float));
    op_desc.SetInput("Filter", {new_filter_name});

    std::string new_bias_name = "block_bias_" + new_filter_name;
    auto* new_bias_node = graph->NewArgumentNode(new_bias_name);
    new_bias_node->arg()->is_weight = true;
    new_bias_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    if (encode_bias_size.back() > 0) {
      std::unique_ptr<float[]> encode_bias(new float[encode_bias_size.back()]);
      for (int i = 0; i < bias_name.size(); i++) {
        auto* bias_t = scope->FindMutableTensor(bias_name[i]);
        float* bias_on_host = bias_t->mutable_data<float>();
        memcpy(encode_bias.get() + encode_bias_size[i],
               bias_on_host,
               (encode_bias_size[i + 1] - encode_bias_size[i]) * sizeof(float));
      }
      auto* new_bias_t = scope->NewTensor(new_bias_name);
      new_bias_t->set_precision(paddle::lite_metal_api::PrecisionType::kFloat);
      new_bias_t->set_persistable(true);
      new_bias_t->Resize({encode_bias_size.back()});
      float* new_bias_ptr = new_bias_t->mutable_data<float>();
      memcpy(new_bias_ptr,
             encode_bias.get(),
             encode_bias_size.back() * sizeof(float));
      op_desc.SetInput("Bias", {new_bias_name});
    }

    auto& valid_places = conv1->valid_places();
    auto resblock_darknet_op = LiteOpRegistry::Global().Create(op_desc.Type());
    resblock_darknet_op->Attach(op_desc, scope);
    auto* new_op_node =
        graph->GraphCreateInstructNode(resblock_darknet_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    if (encode_bias_size.back() > 0) {
      IR_NODE_LINK_TO(new_bias_node, new_op_node);
    }
    IR_NODE_LINK_TO(new_op_node, matched.at("output"));
    IR_NODE_LINK_TO(new_op_node, max_output_node);
  }
};

}  // namespace fusion

class XPUResBlockDarknetFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUResBlockDarknetFuser fuser0;
    fuser0(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__resblock_darknet_fuse_pass,
                  paddle::lite_metal::mir::XPUResBlockDarknetFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");
