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
namespace lite_metal {
namespace mir {
namespace fusion {
/* fuse xpu_conv2d and avg pool2d as resnet50-like block */
/* graph[1]: has_mid_conv = true                         */
/*           has_avg_pool2d = true                       */
/*                in_Input                               */
/*                /      \                               */
/*              /          \                             */
/*             |            |                            */
/*         xpu_conv2d   avg_pool2d                       */
/*             |            |                            */
/*             |            |                            */
/*         xpu_conv2d   xpu_conv2d                       */
/*             |            |                            */
/*              \          /                             */
/*                \       /                              */
/*                xpu_conv2d                             */
/*               (with branch)                           */
/*                    |                                  */
/*                    |                                  */
/*                out_Output                             */
/*-------------------------------------------------      */
/* graph[2]: has_mid_conv = true                         */
/*           has_avg_pool2d = false                      */
/*                in_Input                               */
/*                /      \                               */
/*              /          \                             */
/*             |            |                            */
/*         xpu_conv2d       |                            */
/*             |            |                            */
/*             |            |                            */
/*         xpu_conv2d   xpu_conv2d                       */
/*             |            |                            */
/*              \          /                             */
/*                \       /                              */
/*                xpu_conv2d                             */
/*               (with branch)                           */
/*                    |                                  */
/*                    |                                  */
/*                out_Output                             */
/*-------------------------------------------------      */
/* graph[3]: has_mid_conv = false                        */
/*           has_avg_pool2d = true                       */
/*                                                       */
/*                in_Input                               */
/*                /      \                               */
/*              /          \                             */
/*             |        avg_pool2d                       */
/*             |            |                            */
/*         xpu_conv2d   xpu_conv2d                       */
/*             |            |                            */
/*             |            |                            */
/*              \          /                             */
/*                \       /                              */
/*                xpu_conv2d                             */
/*               (with branch)                           */
/*                    |                                  */
/*                    |                                  */
/*                out_Output                             */
/*-------------------------------------------------      */
/* graph[3]: has_mid_conv = false                        */
/*           has_avg_pool2d = false                      */
/*                                                       */
/*                in_Input                               */
/*                /      \                               */
/*              /          \                             */
/*             |            |                            */
/*             |            |                            */
/*         xpu_conv2d   xpu_conv2d                       */
/*             |            |                            */
/*             |            |                            */
/*              \          /                             */
/*                \       /                              */
/*                xpu_conv2d                             */
/*               (with branch)                           */
/*                    |                                  */
/*                    |                                  */
/*                out_Output                             */
/*-------------------------------------------------      */
/* After the pass is applied:                            */
/*                     in_Input                          */
/*        in_Filter      |     in_FilterMax              */
/*                  \    |    /                          */
/*                   \   |   /                           */
/*     in_Bias ------- __xpu__block_fuse                 */
/*                       |    \                          */
/*                       |     \                         */
/*                       |      out_OutputMax            */
/*                 out_Output                            */
/*                                                       */

class XPUResBlockReductionFuser : public FuseBase {
 public:
  explicit XPUResBlockReductionFuser(bool has_mid_conv, bool has_avg_pool2d) {
    has_mid_conv_ = has_mid_conv;
    has_avg_pool2d_ = has_avg_pool2d;
  }

  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__conv2d", "Input")
                      ->AsInput();
    auto* left_conv1_weight =
        VarNode("left_conv1_weight")
            ->assert_is_persistable_var()
            ->assert_is_op_input("__xpu__conv2d", "Filter")
            ->AsIntermediate();
    auto* left_conv1_bias = VarNode("left_conv1_bias")
                                ->assert_is_persistable_var()
                                ->assert_is_op_input("__xpu__conv2d", "Bias")
                                ->AsIntermediate();
    auto* left_conv1 = OpNode("left_conv1", "__xpu__conv2d")
                           ->assert_op_attr<bool>("has_branch", false)
                           ->assert_op_attr<bool>("has_bias", true)
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
    PMNode* left_conv2_bias = nullptr;
    PMNode* left_conv2 = nullptr;
    PMNode* left_conv2_out = nullptr;
    PMNode* left_conv2_out_max = nullptr;
    if (has_mid_conv_) {
      left_conv2_weight = VarNode("left_conv2_weight")
                              ->assert_is_op_input("__xpu__conv2d", "Filter")
                              ->assert_is_persistable_var()
                              ->AsIntermediate();
      left_conv2_bias = VarNode("left_conv2_bias")
                            ->assert_is_op_input("__xpu__conv2d", "Bias")
                            ->assert_is_persistable_var()
                            ->AsIntermediate();
      left_conv2 = OpNode("left_conv2", "__xpu__conv2d")
                       ->assert_op_attr<bool>("has_branch", false)
                       ->assert_op_attr<bool>("has_bias", true)
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
    PMNode* pool2d = nullptr;
    PMNode* pool2d_out = nullptr;
    if (has_avg_pool2d_) {
      auto pool2d_teller = [](const Node* x) -> bool {
        if (x && x->IsStmt()) {
          auto* op_info = x->stmt()->op_info();
          if (op_info->HasAttr("adaptive") &&
              op_info->GetAttr<bool>("adaptive")) {
            return false;
          }
          if (op_info->HasAttr("padding_algorithm") &&
              op_info->GetAttr<std::string>("padding_algorithm") == "SAME") {
            return false;
          }
        }
        return true;
      };
      input->assert_is_op_input("pool2d", "X");
      pool2d = OpNode("pool2d", "pool2d")
                   ->assert_op_attr<bool>("global_pooling", false)
                   ->assert_op_attr<std::string>("pooling_type", "avg")
                   ->assert_node_satisfied(pool2d_teller)
                   ->AsIntermediate();
      pool2d_out = VarNode("pool2d_out")
                       ->assert_is_op_input("__xpu__conv2d", "Input")
                       ->assert_is_op_output("pool2d", "Out");
    }
    auto* right_conv1_weight =
        VarNode("right_conv1_weight")
            ->assert_is_op_input("__xpu__conv2d", "Filter")
            ->assert_is_persistable_var()
            ->AsIntermediate();
    auto* right_conv1_bias = VarNode("right_conv1_bias")
                                 ->assert_is_op_input("__xpu__conv2d", "Bias")
                                 ->assert_is_persistable_var()
                                 ->AsIntermediate();
    auto* right_conv1 = OpNode("right_conv1", "__xpu__conv2d")
                            ->assert_op_attr<bool>("has_branch", false)
                            ->assert_op_attr<bool>("has_bias", true)
                            ->AsIntermediate();
    auto* right_conv1_out = VarNode("right_conv1_out")
                                ->assert_is_op_output("__xpu__conv2d", "Output")
                                ->assert_is_op_input("__xpu__conv2d", "Branch")
                                ->AsIntermediate();
    auto* right_conv1_out_max =
        VarNode("right_conv1_out_max")
            ->assert_is_op_output("__xpu__conv2d", "OutputMax")
            ->AsIntermediate();
    auto* left_conv3_weight =
        VarNode("left_conv3_weight")
            ->assert_is_op_input("__xpu__conv2d", "Filter")
            ->assert_is_persistable_var()
            ->AsIntermediate();
    auto* left_conv3_bias = VarNode("left_conv3_bias")
                                ->assert_is_op_input("__xpu__conv2d", "Bias")
                                ->assert_is_persistable_var()
                                ->AsIntermediate();
    auto* left_conv3 = OpNode("left_conv3", "__xpu__conv2d")
                           ->assert_op_attr<bool>("has_branch", true)
                           ->assert_op_attr<bool>("has_bias", true)
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
    } else {
      *input >> *left_conv1 >> *left_conv1_out >> *left_conv3;
    }
    if (has_avg_pool2d_) {
      *input >> *pool2d >> *pool2d_out >> *right_conv1 >> *right_conv1_out >>
          *left_conv3;
    } else {
      *input >> *right_conv1 >> *right_conv1_out >> *left_conv3;
    }
    *left_conv3 >> *left_conv3_out;
    *left_conv1_weight >> *left_conv1;
    *left_conv1_bias >> *left_conv1;
    *left_conv1 >> *left_conv1_out_max;
    if (has_mid_conv_) {
      *left_conv2_weight >> *left_conv2;
      *left_conv2_bias >> *left_conv2;
      *left_conv2 >> *left_conv2_out_max;
    }
    *right_conv1_weight >> *right_conv1;
    *right_conv1_bias >> *right_conv1;
    *right_conv1 >> *right_conv1_out_max;
    *left_conv3_weight >> *left_conv3;
    *left_conv3_bias >> *left_conv3;
    *left_conv3 >> *left_conv3_out_max;
  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::vector<std::string> conv_name{
        "left_conv1", "right_conv1", "left_conv3"};
    std::vector<std::string> filter_name{
        matched.at("left_conv1_weight")->arg()->name,
        matched.at("right_conv1_weight")->arg()->name,
        matched.at("left_conv3_weight")->arg()->name};
    std::vector<std::string> bias_name = {
        matched.at("left_conv1_bias")->arg()->name,
        matched.at("right_conv1_bias")->arg()->name,
        matched.at("left_conv3_bias")->arg()->name};
    if (has_mid_conv_) {
      conv_name.insert(conv_name.begin() + 1, "left_conv2");
      filter_name.insert(filter_name.begin() + 1,
                         matched.at("left_conv2_weight")->arg()->name);
      bias_name.insert(bias_name.begin() + 1,
                       matched.at("left_conv2_bias")->arg()->name);
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
    std::vector<int> conv_bias;
    if (has_avg_pool2d_) {
      int pooling_type = -1;
      if (matched.at("pool2d")->stmt()->op_info()->GetAttr<bool>("exclusive") ==
          true) {
        pooling_type = 1;
      } else {
        pooling_type = 2;
      }
      if (has_mid_conv_) {
        op_type = {0, 0, pooling_type, 0, 0};
        place_x = {0, 1, 0, 3, 2};
        place_y = {9, 9, 9, 9, 4};
        place_z = {1, 2, 3, 4, 10};
        block_lod = {5};
        conv_bias = {1, 1, 1, 1};
      } else {
        op_type = {0, pooling_type, 0, 0};
        place_x = {0, 0, 2, 1};
        place_y = {9, 9, 9, 3};
        place_z = {1, 2, 3, 10};
        block_lod = {4};
        conv_bias = {1, 1, 1};
      }
    } else {
      if (has_mid_conv_) {
        op_type = {0, 0, 0, 0};
        place_x = {0, 1, 0, 2};
        place_y = {9, 9, 9, 4};
        place_z = {1, 2, 4, 10};
        block_lod = {4};
        conv_bias = {1, 1, 1, 1};
      } else {
        op_type = {0, 0, 0};
        place_x = {0, 0, 1};
        place_y = {9, 9, 2};
        place_z = {1, 2, 10};
        block_lod = {3};
        conv_bias = {1, 1, 1};
      }
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
      conv_groups.insert(
          conv_groups.end(), cur_groups.begin(), cur_groups.end());
      act_type.insert(act_type.end(), cur_act_type.begin(), cur_act_type.end());
      act_param.insert(
          act_param.end(), cur_act_param.begin(), cur_act_param.end());
    }
    if (has_avg_pool2d_) {
      auto pool_kernel =
          matched.at("pool2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
              "ksize");
      filter_dims.insert(
          filter_dims.end() - 2 * 4, pool_kernel.begin(), pool_kernel.end());

      auto pool_strides =
          matched.at("pool2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
              "strides");
      conv_strides.insert(
          conv_strides.end() - 2 * 2, pool_strides.begin(), pool_strides.end());

      auto pool_paddings =
          matched.at("pool2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
              "paddings");
      if (pool_paddings.size() == 2) {
        for (size_t i = 0; i < pool_strides.size(); ++i) {
          int copy_pad = *(pool_paddings.begin() + 2 * i);
          pool_paddings.insert(pool_paddings.begin() + 2 * i + 1, copy_pad);
        }
      }
      CHECK_EQ(pool_paddings.size(), 4UL)
          << "Paddings size should be 2 or 4, But received paddings size: "
          << pool_paddings.size();
      if ((matched.at("pool2d")->stmt()->op_info()->HasAttr(
              "padding_algorithm")) &&
          (matched.at("pool2d")->stmt()->op_info()->GetAttr<std::string>(
               "padding_algorithm") == "VALID")) {
        pool_paddings[0] = 0;
        pool_paddings[1] = 0;
        pool_paddings[2] = 0;
        pool_paddings[3] = 0;
      }

      if ((matched.at("pool2d")->stmt()->op_info()->HasAttr("ceil_mode")) &&
          (matched.at("pool2d")->stmt()->op_info()->GetAttr<bool>(
              "ceil_mode"))) {
        pool_paddings[1] += pool_strides[0] - 1;
        pool_paddings[3] += pool_strides[1] - 1;
      }
      conv_paddings.insert(conv_paddings.end() - 2 * 4,
                           pool_paddings.begin(),
                           pool_paddings.end());
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
    for (size_t i = 0; i < filter_name.size(); i++) {
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
    new_filter_t->set_precision(paddle::lite_metal_api::PrecisionType::kFloat);
    new_filter_t->set_persistable(true);
    new_filter_t->Resize({encode_filter_size.back()});
    float* new_filter_ptr = new_filter_t->mutable_data<float>();
    memcpy(new_filter_ptr,
           encode_filter_float.get(),
           encode_filter_size.back() * sizeof(float));
    op_desc.SetInput("Filter", {new_filter_name});

    std::unique_ptr<float[]> encode_bias(new float[encode_bias_size.back()]);
    for (size_t i = 0; i < bias_name.size(); i++) {
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
    new_bias_t->set_precision(paddle::lite_metal_api::PrecisionType::kFloat);
    new_bias_t->set_persistable(true);
    new_bias_t->Resize({encode_bias_size.back()});
    float* new_bias_ptr = new_bias_t->mutable_data<float>();
    memcpy(new_bias_ptr,
           encode_bias.get(),
           encode_bias_size.back() * sizeof(float));
    op_desc.SetInput("Bias", {new_bias_name});

    auto& valid_places = left_conv1->valid_places();
    auto resblock_reduction_op =
        LiteOpRegistry::Global().Create(op_desc.Type());
    resblock_reduction_op->Attach(op_desc, scope);
    auto* new_op_node =
        graph->GraphCreateInstructNode(resblock_reduction_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    IR_NODE_LINK_TO(new_bias_node, new_op_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("left_conv3_out"));
    IR_NODE_LINK_TO(new_op_node, matched.at("left_conv3_out_max"));
  }

 private:
  bool has_mid_conv_;
  bool has_avg_pool2d_;
};

}  // namespace fusion

class XPUResBlockReductionFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto has_mid_conv : {true, false}) {
      for (auto has_avg_pool2d : {true, false}) {
        fusion::XPUResBlockReductionFuser fuser(has_mid_conv, has_avg_pool2d);
        fuser(graph.get());
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__resblock_reduction_fuse_pass,
                  paddle::lite_metal::mir::XPUResBlockReductionFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");
