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
/* fuse xpu_conv2d and xpu_block as xpu_block*/
/* graph[1]: block0_type = xpu_conv2d        */
/*           block1_type = xpu_block         */
/*                in_Input                   */
/*                    |                      */
/*                    |                      */
/*                __xpu__conv2d              */
/*                    |                      */
/*                    |                      */
/*                __xpu__block               */
/*                    |                      */
/*                    |                      */
/*                out_Output                 */
/*-------------------------------------------*/
/* graph[2]: block0_type = xpu_block         */
/*           block1_type = xpu_conv2d        */
/*                in_Input                   */
/*                    |                      */
/*                    |                      */
/*                __xpu__block               */
/*                    |                      */
/*                    |                      */
/*                __xpu__conv2d              */
/*                    |                      */
/*                    |                      */
/*                out_Output                 */
/*-------------------------------------------*/
/* graph[3]: block0_type = xpu_block         */
/*           block1_type = xpu_block         */
/*                in_Input                   */
/*                    |                      */
/*                    |                      */
/*                __xpu__block               */
/*                    |                      */
/*                    |                      */
/*                __xpu__block               */
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

class XPUConsecutiveBlockFuser : public FuseBase {
 public:
  explicit XPUConsecutiveBlockFuser(const std::string& block0_type,
                                    const std::string& block1_type,
                                    bool block0_with_bias,
                                    bool block1_with_bias) {
    block0_type_ = block0_type;
    block1_type_ = block1_type;
    block0_with_bias_ = block0_with_bias;
    block1_with_bias_ = block1_with_bias;
  }
  void BuildPattern() override {
    auto* input =
        VarNode("input")->assert_is_op_input(block0_type_, "Input")->AsInput();
    auto* filter0 = VarNode("filter0")
                        ->assert_is_op_input(block0_type_, "Filter")
                        ->AsIntermediate();
    PMNode* bias0 = nullptr;
    PMNode* bias1 = nullptr;
    if (block0_with_bias_) {
      bias0 = VarNode("bias0")->assert_is_op_input(block0_type_, "Bias");
    }
    auto* block0 = OpNode("block0", block0_type_)
                       ->assert_op_attr<bool>("has_bias", block0_with_bias_)
                       ->assert_op_attr<bool>("has_branch", false)
                       ->AsIntermediate();

    auto* block_out0 = VarNode("block_out0")
                           ->assert_is_op_output(block0_type_, "Output")
                           ->AsIntermediate();
    auto* block_out_max0 = VarNode("block_out_max0")
                               ->assert_is_op_output(block0_type_, "OutputMax")
                               ->AsIntermediate();
    auto* filter1 = VarNode("filter1")
                        ->assert_is_op_input(block1_type_, "Filter")
                        ->AsIntermediate();
    if (block1_with_bias_) {
      bias1 = VarNode("bias1")->assert_is_op_input(block1_type_, "Bias");
    }
    auto* block1 = OpNode("block1", block1_type_)
                       ->assert_op_attr<bool>("has_bias", block1_with_bias_)
                       ->assert_op_attr<bool>("has_branch", false)
                       ->AsIntermediate();
    auto* block_out1 = VarNode("block_out1")
                           ->assert_is_op_output(block1_type_, "Output")
                           ->AsOutput();
    auto* block_out_max1 = VarNode("block_out_max1")
                               ->assert_is_op_output(block1_type_, "OutputMax")
                               ->AsOutput();

    if (block0_with_bias_ && block1_with_bias_) {
      bias0->AsIntermediate();
      bias1->AsIntermediate();
    } else if (block0_with_bias_) {
      bias0->AsInput();
    } else if (block1_with_bias_) {
      bias1->AsInput();
    }

    *input >> *block0 >> *block_out0 >> *block1 >> *block_out1;
    *filter0 >> *block0;
    if (block0_with_bias_) {
      *bias0 >> *block0;
    }
    *block0 >> *block_out_max0;
    *filter1 >> *block1;
    if (block1_with_bias_) {
      *bias1 >> *block1;
    }
    *block1 >> *block_out_max1;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::vector<std::string> block_name{"block0", "block1"};
    std::vector<std::string> filter_name{matched.at("filter0")->arg()->name,
                                         matched.at("filter1")->arg()->name};
    std::vector<std::string> bias_name;
    if (block0_with_bias_) {
      bias_name.push_back(matched.at("bias0")->arg()->name);
    }
    if (block1_with_bias_) {
      bias_name.push_back(matched.at("bias1")->arg()->name);
    }
    cpp::OpDesc op_desc;
    auto block_0 = matched.at("block0")->stmt()->op();
    auto* scope = block_0->scope();
    op_desc.SetType("__xpu__block_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetOutput("Output", {matched.at("block_out1")->arg()->name});
    op_desc.SetOutput("OutputMax", {matched.at("block_out_max1")->arg()->name});

    std::vector<int> op_type;
    std::vector<int> place_x;
    std::vector<int> place_y;
    std::vector<int> place_z;
    std::vector<int> filter_dims;
    std::vector<int> conv_strides;
    std::vector<int> conv_paddings;
    std::vector<int> conv_dilations;
    std::vector<int> conv_groups;
    std::vector<int> act_type;
    std::vector<float> act_param;
    std::vector<int> block_lod;
    std::vector<int> conv_bias;

    for (auto name : block_name) {
      auto cur_filter_dims =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "filter_dims");
      auto cur_strides =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "strides");
      auto cur_paddings =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "paddings");
      auto cur_op_type =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "op_type");
      auto cur_place_x =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "place_x");
      auto cur_place_y =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "place_y");
      auto cur_place_z =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "place_z");
      auto cur_conv_groups =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "groups");
      auto cur_act_type =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "act_type");
      auto cur_act_param =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<float>>(
              "act_param");
      auto cur_block_lod =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "block_lod");
      auto cur_conv_bias =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "conv_bias");
      op_type.insert(op_type.end(), cur_op_type.begin(), cur_op_type.end());
      place_x.insert(place_x.end(), cur_place_x.begin(), cur_place_x.end());
      place_y.insert(place_y.end(), cur_place_y.begin(), cur_place_y.end());
      place_z.insert(place_z.end(), cur_place_z.begin(), cur_place_z.end());
      conv_groups.insert(
          conv_groups.end(), cur_conv_groups.begin(), cur_conv_groups.end());
      act_type.insert(act_type.end(), cur_act_type.begin(), cur_act_type.end());
      act_param.insert(
          act_param.end(), cur_act_param.begin(), cur_act_param.end());
      block_lod.insert(
          block_lod.end(), cur_block_lod.begin(), cur_block_lod.end());
      conv_bias.insert(
          conv_bias.end(), cur_conv_bias.begin(), cur_conv_bias.end());
      if (cur_paddings.size() == 2) {
        for (size_t i = 0; i < cur_strides.size(); ++i) {
          int copy_pad = *(cur_paddings.begin() + 2 * i);
          cur_paddings.insert(cur_paddings.begin() + 2 * i + 1, copy_pad);
        }
      }
      auto cur_conv_dilations =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "dilations");
      filter_dims.insert(
          filter_dims.end(), cur_filter_dims.begin(), cur_filter_dims.end());
      conv_strides.insert(
          conv_strides.end(), cur_strides.begin(), cur_strides.end());
      conv_paddings.insert(
          conv_paddings.end(), cur_paddings.begin(), cur_paddings.end());
      conv_dilations.insert(conv_dilations.end(),
                            cur_conv_dilations.begin(),
                            cur_conv_dilations.end());
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
    op_desc.SetAttr<bool>("has_bias", (block0_with_bias_ || block1_with_bias_));
    op_desc.SetAttr<bool>("has_branch", false);

    auto* filter0_t = scope->FindMutableTensor(filter_name[0]);
    auto* filter1_t = scope->FindMutableTensor(filter_name[1]);
    int filter0_numel = filter0_t->numel();
    int filter1_numel = filter1_t->numel();
    std::unique_ptr<float[]> encode_filter_int16(
        new float[filter0_numel + filter1_numel]);
    float* filter0_on_host = filter0_t->mutable_data<float>();
    float* filter1_on_host = filter1_t->mutable_data<float>();
    memcpy(encode_filter_int16.get(),
           filter0_on_host,
           filter0_numel * sizeof(float));
    memcpy(encode_filter_int16.get() + filter0_numel,
           filter1_on_host,
           filter1_numel * sizeof(float));
    std::string new_filter_name = "block_" + filter_name[0];
    auto* new_filter_node = graph->NewArgumentNode(new_filter_name);
    new_filter_node->arg()->is_weight = true;
    new_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_filter_t = scope->NewTensor(new_filter_name);
    new_filter_t->set_precision(paddle::lite_api::PrecisionType::kFloat);
    new_filter_t->set_persistable(true);
    new_filter_t->Resize({filter0_numel + filter1_numel});
    float* new_filter_ptr = new_filter_t->mutable_data<float>();
    memcpy(new_filter_ptr,
           encode_filter_int16.get(),
           (filter0_numel + filter1_numel) * sizeof(float));
    op_desc.SetInput("Filter", {new_filter_name});

    std::string new_bias_name = new_filter_name + "_bias";
    auto* new_bias_node = graph->NewArgumentNode(new_bias_name);
    new_bias_node->arg()->is_weight = true;
    new_bias_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_bias_t = scope->NewTensor(new_bias_name);
    new_bias_t->set_precision(paddle::lite_api::PrecisionType::kFloat);
    new_bias_t->set_persistable(true);
    if (block0_with_bias_ && block1_with_bias_) {
      auto* bias0_t = scope->FindMutableTensor(bias_name[0]);
      auto* bias1_t = scope->FindMutableTensor(bias_name[1]);
      int bias0_numel = bias0_t->numel();
      int bias1_numel = bias1_t->numel();
      std::unique_ptr<float[]> encode_bias(
          new float[bias0_numel + bias1_numel]);
      float* bias0_on_host = bias0_t->mutable_data<float>();
      float* bias1_on_host = bias1_t->mutable_data<float>();
      memcpy(encode_bias.get(), bias0_on_host, bias0_numel * sizeof(float));
      memcpy(encode_bias.get() + bias0_numel,
             bias1_on_host,
             bias1_numel * sizeof(float));
      new_bias_t->Resize({bias0_numel + bias1_numel});
      float* new_bias_ptr = new_bias_t->mutable_data<float>();
      memcpy(new_bias_ptr,
             encode_bias.get(),
             (bias0_numel + bias1_numel) * sizeof(float));
      op_desc.SetInput("Bias", {new_bias_name});
    } else if (block0_with_bias_ || block1_with_bias_) {
      op_desc.SetInput("Bias", {bias_name[0]});
    }

    auto& valid_places = block_0->valid_places();
    auto block_op = LiteOpRegistry::Global().Create(op_desc.Type());
    block_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(block_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    if (block0_with_bias_ && block1_with_bias_) {
      IR_NODE_LINK_TO(new_bias_node, new_op_node);
    } else if (block0_with_bias_) {
      IR_NODE_LINK_TO(matched.at("bias0"), new_op_node);
    } else if (block1_with_bias_) {
      IR_NODE_LINK_TO(matched.at("bias1"), new_op_node);
    }
    IR_NODE_LINK_TO(new_op_node, matched.at("block_out1"));
    IR_NODE_LINK_TO(new_op_node, matched.at("block_out_max1"));
  }

 private:
  std::string block0_type_;
  std::string block1_type_;
  bool block0_with_bias_;
  bool block1_with_bias_;
};

}  // namespace fusion

class XPUConsecutiveBlockFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto block0_with_bias : {true, false}) {
      for (auto block1_with_bias : {true, false}) {
        fusion::XPUConsecutiveBlockFuser fuser(
            "__xpu__conv2d",
            "__xpu__squeeze_excitation_block",
            block1_with_bias,
            block0_with_bias);
        fuser(graph.get());
      }
    }
    for (auto block0_with_bias : {true, false}) {
      for (auto block1_with_bias : {true, false}) {
        fusion::XPUConsecutiveBlockFuser fuser(
            "__xpu__block_fuse_op",
            "__xpu__squeeze_excitation_block",
            block1_with_bias,
            block0_with_bias);
        fuser(graph.get());
      }
    }
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto block0_with_bias : {true, false}) {
        for (auto block1_with_bias : {true, false}) {
          fusion::XPUConsecutiveBlockFuser fuser1("__xpu__conv2d",
                                                  "__xpu__block_fuse_op",
                                                  block1_with_bias,
                                                  block0_with_bias);
          bool cur_changed = fuser1(graph.get());
          changed = cur_changed ? true : changed;
          fusion::XPUConsecutiveBlockFuser fuser2("__xpu__block_fuse_op",
                                                  "__xpu__conv2d",
                                                  block1_with_bias,
                                                  block0_with_bias);
          cur_changed = fuser2(graph.get());
          changed = cur_changed ? true : changed;
        }
      }
    }
    changed = true;
    while (changed) {
      changed = false;
      for (auto block0_with_bias : {true, false}) {
        for (auto block1_with_bias : {true, false}) {
          fusion::XPUConsecutiveBlockFuser fuser3("__xpu__block_fuse_op",
                                                  "__xpu__block_fuse_op",
                                                  block1_with_bias,
                                                  block0_with_bias);
          bool cur_changed = fuser3(graph.get());
          changed = cur_changed ? true : changed;
        }
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__consecutive_block_fuse_pass,
                  paddle::lite::mir::XPUConsecutiveBlockFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");
