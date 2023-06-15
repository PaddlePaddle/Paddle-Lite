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

#include <math.h>
#include <memory>
#include <string>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/vec_trans.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {
typedef std::vector<std::map<std::string, PMNode*>> NodeContainer;

class XPUUpDecoderFuser : public FuseBase {
 public:
  XPUUpDecoderFuser(int num_resblocks,
                    bool has_interp,
                    bool post_interp_conv,
                    bool post_interp_conv_input_max,
                    bool first_resblock_has_input_max)
      : num_resblocks_(num_resblocks),
        has_interp_(has_interp),
        post_interp_conv_(post_interp_conv),
        post_interp_conv_input_max_(post_interp_conv_input_max),
        first_resblock_has_input_max_(first_resblock_has_input_max) {
    all_inputs_map_["Input"] = {};
    all_inputs_map_["ResblockConvBias"] = {};
    all_inputs_map_["ResblockConvFilter"] = {};
    all_inputs_map_["ResblockGNScale"] = {};
    all_inputs_map_["ResblockGNBias"] = {};
    all_inputs_map_["ResblockInputMax"] = {};
    all_inputs_map_["PostConvFilter"] = {};
    all_inputs_map_["PostConvBias"] = {};
    all_inputs_map_["PostConvInputMax"] = {};
    all_resblocks_op_keys_.clear();
  }

  NodeContainer BuildSingleResblock(int op_pos) {
    // op_pos ranges from 0 to N;
    // 0: begining op; not zero: intermediate op;
    // Single resblock inputs
    std::map<std::string, PMNode*> input_nodes;
    std::map<std::string, PMNode*> op_node;
    std::map<std::string, PMNode*> output_nodes;
    NodeContainer nodes_pack;
    PMNode* input_1 = nullptr;
    PMNode* input_max = nullptr;
    if (op_pos == 0) {
      all_inputs_map_["Input"].push_back("input_1_" + to_string(op_pos));
      input_1 = VarNode("input_1_" + to_string(op_pos))
                    ->assert_is_op_input("__xpu__spatial_transformer_resblock",
                                         "Input1")
                    ->AsInput();
      if (first_resblock_has_input_max_) {
        all_inputs_map_["ResblockInputMax"].push_back("input_max_" +
                                                      to_string(op_pos));
        input_max = VarNode("input_max_" + to_string(op_pos))
                        ->assert_is_op_input(
                            "__xpu__spatial_transformer_resblock", "InputMax")
                        ->AsInput();
      }
    }
    all_inputs_map_["ResblockConvBias"].push_back("conv_bias_0_" +
                                                  to_string(op_pos));
    PMNode* conv_bias_0 =
        VarNode("conv_bias_0_" + to_string(op_pos))
            ->assert_is_op_nth_input(
                "__xpu__spatial_transformer_resblock", "ConvBias", 0)
            ->AsInput();
    all_inputs_map_["ResblockConvBias"].push_back("conv_bias_1_" +
                                                  to_string(op_pos));
    PMNode* conv_bias_1 =
        VarNode("conv_bias_1_" + to_string(op_pos))
            ->assert_is_op_nth_input(
                "__xpu__spatial_transformer_resblock", "ConvBias", 1)
            ->AsInput();
    PMNode* conv_bias_2 = nullptr;
    if (op_pos == 0 && first_resblock_has_input_max_) {
      all_inputs_map_["ResblockConvBias"].push_back("conv_bias_2_" +
                                                    to_string(op_pos));
      conv_bias_2 =
          VarNode("conv_bias_2_" + to_string(op_pos))
              ->assert_is_op_nth_input(
                  "__xpu__spatial_transformer_resblock", "ConvBias", 2)
              ->AsInput();
    }
    all_inputs_map_["ResblockConvFilter"].push_back("conv_filter_0_" +
                                                    to_string(op_pos));
    PMNode* conv_filter_0 =
        VarNode("conv_filter_0_" + to_string(op_pos))
            ->assert_is_op_nth_input(
                "__xpu__spatial_transformer_resblock", "ConvFilter", 0)
            ->AsInput();
    all_inputs_map_["ResblockConvFilter"].push_back("conv_filter_1_" +
                                                    to_string(op_pos));
    PMNode* conv_filter_1 =
        VarNode("conv_filter_1_" + to_string(op_pos))
            ->assert_is_op_nth_input(
                "__xpu__spatial_transformer_resblock", "ConvFilter", 1)
            ->AsInput();
    PMNode* conv_filter_2 = nullptr;
    if (op_pos == 0 && first_resblock_has_input_max_) {
      all_inputs_map_["ResblockConvFilter"].push_back("conv_filter_2_" +
                                                      to_string(op_pos));
      conv_filter_2 =
          VarNode("conv_filter_2_" + to_string(op_pos))
              ->assert_is_op_nth_input(
                  "__xpu__spatial_transformer_resblock", "ConvFilter", 2)
              ->AsInput();
    }
    all_inputs_map_["ResblockGNBias"].push_back("gn_bias_0_" +
                                                to_string(op_pos));
    PMNode* gn_bias_0 =
        VarNode("gn_bias_0_" + to_string(op_pos))
            ->assert_is_op_nth_input(
                "__xpu__spatial_transformer_resblock", "GNBias", 0)
            ->AsInput();
    all_inputs_map_["ResblockGNBias"].push_back("gn_bias_1_" +
                                                to_string(op_pos));
    PMNode* gn_bias_1 =
        VarNode("gn_bias_1_" + to_string(op_pos))
            ->assert_is_op_nth_input(
                "__xpu__spatial_transformer_resblock", "GNBias", 1)
            ->AsInput();
    all_inputs_map_["ResblockGNScale"].push_back("gn_scale_0_" +
                                                 to_string(op_pos));
    PMNode* gn_scale_0 =
        VarNode("gn_scale_0_" + to_string(op_pos))
            ->assert_is_op_nth_input(
                "__xpu__spatial_transformer_resblock", "GNScale", 0)
            ->AsInput();
    all_inputs_map_["ResblockGNScale"].push_back("gn_scale_1_" +
                                                 to_string(op_pos));
    PMNode* gn_scale_1 =
        VarNode("gn_scale_1_" + to_string(op_pos))
            ->assert_is_op_nth_input(
                "__xpu__spatial_transformer_resblock", "GNScale", 1)
            ->AsInput();

    // Single resblock output
    PMNode* output = VarNode("resblock_output_" + to_string(op_pos))
                         ->assert_is_op_output(
                             "__xpu__spatial_transformer_resblock", "Output");
    if (op_pos == (num_resblocks_ - 1) && has_interp_ == false) {
      output->AsOutput();
    } else {
      output->AsIntermediate();
    }

    // Single block op node
    PMNode* resblock_op = OpNode("resblock_op_" + to_string(op_pos),
                                 "__xpu__spatial_transformer_resblock")
                              ->AsIntermediate();
    all_resblocks_op_keys_.push_back("resblock_op_" + to_string(op_pos));
    input_nodes["input_1"] = input_1;
    input_nodes["conv_bias_0"] = conv_bias_0;
    input_nodes["conv_bias_1"] = conv_bias_1;
    input_nodes["conv_bias_2"] = conv_bias_2;
    input_nodes["conv_filter_0"] = conv_filter_0;
    input_nodes["conv_filter_1"] = conv_filter_1;
    input_nodes["conv_filter_2"] = conv_filter_2;
    input_nodes["gn_bias_0"] = gn_bias_0;
    input_nodes["gn_bias_1"] = gn_bias_1;
    input_nodes["gn_scale_0"] = gn_scale_0;
    input_nodes["gn_scale_1"] = gn_scale_1;
    if (first_resblock_has_input_max_) {
      input_nodes["input_max"] = input_max;
    }

    output_nodes["output"] = output;
    op_node["resblock_op"] = resblock_op;

    nodes_pack.emplace_back(input_nodes);
    nodes_pack.emplace_back(output_nodes);
    nodes_pack.emplace_back(op_node);

    return nodes_pack;
  }

  void BuildPattern() override {
    std::vector<NodeContainer> resblocks;
    for (int i = 0; i < num_resblocks_; i++) {
      resblocks.push_back(BuildSingleResblock(i));
    }
    // Build reesblock sequence.
    for (int i = 0; i < resblocks.size(); i++) {
      std::vector<PMNode*> single_resblock_inputs = {
          resblocks[i][0]["conv_bias_0"],
          resblocks[i][0]["conv_bias_1"],
          resblocks[i][0]["conv_filter_0"],
          resblocks[i][0]["conv_filter_1"],
          resblocks[i][0]["gn_bias_0"],
          resblocks[i][0]["gn_bias_1"],
          resblocks[i][0]["gn_scale_0"],
          resblocks[i][0]["gn_scale_1"],
      };
      if (i == 0) {
        single_resblock_inputs.push_back(resblocks[i][0]["input_1"]);
        if (first_resblock_has_input_max_) {
          single_resblock_inputs.push_back(resblocks[i][0]["input_max"]);
          single_resblock_inputs.push_back(resblocks[i][0]["conv_filter_2"]);
          single_resblock_inputs.push_back(resblocks[i][0]["conv_bias_2"]);
        }
      } else {
        single_resblock_inputs.push_back(resblocks[i - 1][1]["output"]);
      }

      PMNode* single_resblock_output = resblocks[i][1]["output"];
      PMNode* single_resblock_op = resblocks[i][2]["resblock_op"];
      single_resblock_inputs >> *single_resblock_op >> *single_resblock_output;
    }
    PMNode* post_interp_op = nullptr;
    PMNode* post_interp_out = nullptr;
    PMNode* post_conv_op = nullptr;
    PMNode* post_conv_input_max = nullptr;
    PMNode* post_conv_filter = nullptr;
    PMNode* post_conv_bias = nullptr;
    PMNode* post_conv_out = nullptr;
    PMNode* post_conv_out_max = nullptr;

    if (has_interp_) {
      post_interp_op =
          OpNode("post_interp", "nearest_interp_v2")->AsIntermediate();
      post_interp_out = VarNode("post_interp_out")
                            ->assert_is_op_output("nearest_interp_v2", "Out");
      if (post_interp_conv_) {
        post_interp_out->AsIntermediate();
      } else {
        post_interp_out->AsOutput();
      }
      *resblocks[num_resblocks_ - 1][1]["output"] >> *post_interp_op >>
          *post_interp_out;
      if (post_interp_conv_) {
        post_conv_op = OpNode("post_conv", "__xpu__conv2d")->AsIntermediate();
        post_conv_filter = VarNode("post_conv_filter")
                               ->assert_is_op_input("__xpu__conv2d", "Filter")
                               ->AsInput();
        post_conv_bias = VarNode("post_conv_bias")
                             ->assert_is_op_input("__xpu__conv2d", "Bias")
                             ->AsInput();
        post_conv_out_max =
            VarNode("post_conv_out_max")
                ->assert_is_op_output("__xpu__conv2d", "OutputMax")
                ->AsOutput();
        post_conv_out = VarNode("post_conv_out")
                            ->assert_is_op_output("__xpu__conv2d", "Output")
                            ->AsOutput();
        all_inputs_map_["PostConvFilter"].push_back("post_conv_filter");
        all_inputs_map_["PostConvBias"].push_back("post_conv_bias");

        std::vector<PMNode*> post_conv_inputs = {
            post_interp_out, post_conv_filter, post_conv_bias};
        std::vector<PMNode*> post_conv_outputs = {post_conv_out,
                                                  post_conv_out_max};
        if (post_interp_conv_input_max_) {
          post_conv_input_max =
              VarNode("post_conv_input_max")
                  ->assert_is_op_input("__xpu__conv2d", "InputMax")
                  ->AsInput();
          post_conv_inputs.push_back(post_conv_input_max);
          all_inputs_map_["PostConvInputMax"].push_back("post_conv_input_max");
        }
        post_conv_inputs >> *post_conv_op >> post_conv_outputs;
      }
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    std::vector<std::string> all_conv_weight_names;
    std::vector<std::string> all_conv_weight_max_names;
    std::vector<std::string> post_conv_weight_names;
    std::vector<std::string> post_conv_weight_max_names;
    std::vector<std::string> all_inputs_node_names;
    std::vector<int> all_resblock_conv_fix;
    std::vector<std::vector<int>> all_resblock_dilations;
    std::vector<std::vector<int>> all_resblock_filter_dims;
    std::vector<std::vector<float>> all_resblock_gneps;
    std::vector<std::vector<int>> all_resblock_gngroups;
    std::vector<std::vector<int>> all_resblock_groups;
    std::vector<std::vector<int>> all_resblock_paddings;
    std::vector<std::vector<int>> all_resblock_strides;
    std::string subgraph_output_key;

    op_desc.SetType("__xpu__up_decoder");
    if (has_interp_ && post_interp_conv_) {
      post_conv_weight_names.push_back(
          matched.at("post_conv_filter")->arg()->name);
      post_conv_weight_max_names.push_back(
          matched.at("post_conv_filter")->arg()->name + "_max");
    }

    for (auto input_ele : all_inputs_map_) {
      std::vector<std::string> input_names;
      for (int i = 0; i < input_ele.second.size(); i++) {
        input_names.push_back(matched.at(input_ele.second[i])->arg()->name);
        all_inputs_node_names.push_back(input_ele.second[i]);
      }

      op_desc.SetInput(input_ele.first, {input_names});
    }
    if (has_interp_) {
      if (post_interp_conv_) {
        op_desc.SetOutput("Output", {matched.at("post_conv_out")->arg()->name});
        subgraph_output_key = "post_conv_out";
      } else {
        op_desc.SetOutput("Output",
                          {matched.at("post_interp_out")->arg()->name});
        subgraph_output_key = "post_interp_out";
      }
    } else {
      op_desc.SetOutput(
          "Output",
          {matched.at("resblock_output_" + to_string(num_resblocks_ - 1))
               ->arg()
               ->name});
      subgraph_output_key = "resblock_output_" + to_string(num_resblocks_ - 1);
    }

    for (int i = 0; i < all_resblocks_op_keys_.size(); i++) {
      auto* resblock_op_info =
          matched.at(all_resblocks_op_keys_[i])->stmt()->op_info();
      for (const auto& name : resblock_op_info->Input("ConvFilter")) {
        all_conv_weight_names.emplace_back(name);
      }
      for (const auto& name :
           resblock_op_info->GetAttr<std::vector<std::string>>(
               "ConvFilterMax")) {
        all_conv_weight_max_names.emplace_back(name);
      }
      all_resblock_conv_fix.push_back(
          static_cast<int>(resblock_op_info->GetAttr<bool>("ConvFix")));

      auto dil_tmp = resblock_op_info->GetAttr<std::vector<int>>("Dilations");
      auto filter_dim_tmp =
          resblock_op_info->GetAttr<std::vector<int>>("FilterDims");
      auto gneps_tmp = resblock_op_info->GetAttr<std::vector<float>>("GNEps");
      auto gngroups_tmp =
          resblock_op_info->GetAttr<std::vector<int>>("GNGroups");
      auto group_tmp = resblock_op_info->GetAttr<std::vector<int>>("Groups");
      auto padding_tmp =
          resblock_op_info->GetAttr<std::vector<int>>("Paddings");
      auto stride_tmp = resblock_op_info->GetAttr<std::vector<int>>("Strides");
      // this elementwise is to seperate attrs from different resblock.
      all_resblock_dilations.push_back(dil_tmp);
      all_resblock_filter_dims.push_back(filter_dim_tmp);
      all_resblock_gneps.push_back(gneps_tmp);
      all_resblock_gngroups.push_back(gngroups_tmp);
      all_resblock_groups.push_back(group_tmp);
      all_resblock_paddings.push_back(padding_tmp);
      all_resblock_strides.push_back(stride_tmp);
    }

    op_desc.SetAttr<std::vector<std::string>>("ResblockConvFilterMaxs",
                                              all_conv_weight_max_names);
    op_desc.SetAttr<std::vector<std::string>>("PostConvFilterMax",
                                              post_conv_weight_max_names);
    op_desc.SetAttr<std::vector<int>>("ResblockConvFix", all_resblock_conv_fix);
    op_desc.SetAttr<std::vector<int>>(
        "ResblockConvDilations",
        paddle::lite::xpu::vec::Vec2DTo1D(all_resblock_dilations));
    op_desc.SetAttr<std::vector<int>>(
        "ResblockConvFilterDims",
        paddle::lite::xpu::vec::Vec2DTo1D(all_resblock_filter_dims));
    op_desc.SetAttr<std::vector<float>>(
        "ResblockGNEps", paddle::lite::xpu::vec::Vec2DTo1D(all_resblock_gneps));
    op_desc.SetAttr<std::vector<int>>(
        "ResblockGNGroups",
        paddle::lite::xpu::vec::Vec2DTo1D(all_resblock_gngroups));
    op_desc.SetAttr<std::vector<int>>(
        "ResblockConvGroups",
        paddle::lite::xpu::vec::Vec2DTo1D(all_resblock_groups));
    op_desc.SetAttr<std::vector<int>>(
        "ResblockConvPaddings",
        paddle::lite::xpu::vec::Vec2DTo1D(all_resblock_paddings));
    op_desc.SetAttr<std::vector<int>>(
        "ResblockConvStrides",
        paddle::lite::xpu::vec::Vec2DTo1D(all_resblock_strides));
    op_desc.SetAttr<bool>("PostInterp", has_interp_);
    op_desc.SetAttr<bool>("PostInterpConv", post_interp_conv_);
    op_desc.SetAttr<int>("NumResblocks", num_resblocks_);
    if (has_interp_) {
      auto* interp_op_info = matched.at("post_interp")->stmt()->op_info();
      auto align_corners = interp_op_info->GetAttr<bool>("align_corners");
      auto interp_method =
          interp_op_info->GetAttr<std::string>("interp_method");
      auto out_d = interp_op_info->GetAttr<int>("out_d");
      auto out_h = interp_op_info->GetAttr<int>("out_h");
      auto out_w = interp_op_info->GetAttr<int>("out_w");
      auto scale = interp_op_info->GetAttr<std::vector<float>>("scale");

      op_desc.SetAttr<bool>("interp_align_corners", align_corners);
      op_desc.SetAttr<std::string>("interp_method", interp_method);
      op_desc.SetAttr<int>("interp_out_d", out_d);
      op_desc.SetAttr<int>("interp_out_h", out_h);
      op_desc.SetAttr<int>("interp_out_w", out_w);
      op_desc.SetAttr<std::vector<float>>("interp_scale", scale);
      if (post_interp_conv_) {
        auto* post_conv_op_info = matched.at("post_conv")->stmt()->op_info();
        op_desc.SetAttr<std::vector<int>>(
            "PostConvDilations",
            post_conv_op_info->GetAttr<std::vector<int>>("dilations"));
        op_desc.SetAttr<std::vector<int>>(
            "PostConvStrides",
            post_conv_op_info->GetAttr<std::vector<int>>("strides"));
        op_desc.SetAttr<std::vector<int>>(
            "PostConvPaddings",
            post_conv_op_info->GetAttr<std::vector<int>>("paddings"));
        op_desc.SetAttr<std::vector<int>>(
            "PostConvGroups",
            post_conv_op_info->GetAttr<std::vector<int>>("groups"));
        op_desc.SetAttr<std::vector<int>>(
            "PostConvFilterDims",
            post_conv_op_info->GetAttr<std::vector<int>>("filter_dims"));
      }
    }
    // Achieve scope of subgraph's firt op.
    auto* scope = matched.at(all_resblocks_op_keys_[0])->stmt()->op()->scope();
    UpdateWeight(
        scope, post_conv_weight_names, post_conv_weight_max_names, false);

    auto up_decoder_op = LiteOpRegistry::Global().Create(op_desc.Type());
    up_decoder_op->Attach(op_desc, scope);
    up_decoder_op->SetValidPlaces(
        matched.at(all_resblocks_op_keys_[0])->stmt()->op()->valid_places());
    auto kernels = up_decoder_op->CreateKernels(up_decoder_op->valid_places());
    auto* new_op_node = graph->GraphCreateInstructNode(
        up_decoder_op, up_decoder_op->valid_places());

    for (auto& from : all_inputs_node_names) {
      IR_NODE_LINK_TO(matched.at(from), new_op_node);
    }
    IR_OP_VAR_LINK(new_op_node, matched.at(subgraph_output_key));
  }

 private:
  int num_resblocks_;
  bool has_interp_;
  bool post_interp_conv_;
  bool post_interp_conv_input_max_;
  bool first_resblock_has_input_max_;
  std::map<std::string, std::vector<std::string>> all_inputs_map_;
  std::vector<std::string> all_resblocks_op_keys_;

  void UpdateWeight(Scope* scope,
                    const std::vector<std::string>& fc_weight_names,
                    const std::vector<std::string>& fc_weight_max_names,
                    bool trans) {
    std::vector<Tensor*> weight_tensor_vec(fc_weight_names.size(), nullptr);
    std::vector<DDimLite> weight_dims_vec(fc_weight_names.size());
    std::vector<int> weight_len_vec(fc_weight_names.size());

    for (size_t i = 0; i < fc_weight_names.size(); ++i) {
      weight_tensor_vec[i] = scope->FindMutableTensor(fc_weight_names[i]);
      CHECK(weight_tensor_vec[i] != nullptr);
      weight_dims_vec[i] = weight_tensor_vec[i]->dims();
      weight_len_vec[i] = weight_tensor_vec[i]->numel();
      if (trans && i > 0) {
        CHECK_EQ(weight_dims_vec[i][0], weight_dims_vec[i - 1][0]);
      }
    }
    for (size_t i = 0; i < fc_weight_names.size(); ++i) {
      float* weight_host_ptr = weight_tensor_vec[i]->mutable_data<float>();
      std::unique_ptr<float[]> weight_host_trans(new float[weight_len_vec[i]]);
      std::unique_ptr<int16_t[]> weight_host_trans_int16(
          new int16_t[weight_len_vec[i]]);
      if (trans) {
        paddle::lite::xpu::math::Transpose<float>(weight_host_ptr,
                                                  weight_host_trans.get(),
                                                  weight_dims_vec[i][0],
                                                  weight_dims_vec[i][1]);
      } else {
        memcpy(weight_host_trans.get(),
               weight_host_ptr,
               weight_len_vec[i] * sizeof(float));
      }
      float max_f = paddle::lite::xpu::math::FindMaxAbs(weight_host_trans.get(),
                                                        weight_len_vec[i]);
      paddle::lite::xpu::math::ConvertFP32ToInt16(weight_host_trans.get(),
                                                  weight_host_trans_int16.get(),
                                                  max_f,
                                                  weight_len_vec[i]);
      memcpy(weight_tensor_vec[i]->mutable_data<int16_t>(),
             weight_host_trans_int16.get(),
             weight_len_vec[i] * sizeof(int16_t));
      scope->NewTensor(fc_weight_max_names[i]);
      Tensor* weight_maxptr_tensor =
          scope->FindMutableTensor(fc_weight_max_names[i]);
      weight_maxptr_tensor->Resize({6});
      std::vector<float> weight_maxptr_host(6, max_f);
      memcpy(weight_maxptr_tensor->mutable_data<float>(),
             weight_maxptr_host.data(),
             weight_maxptr_host.size() * sizeof(float));
    }
  }
};
}  // namespace fusion

/*
Fuse multi resblocks + interp + conv2d into up_decoder_fuse op.

Original subgraph:

                     Input
                       |
                    resblock
                       |
                    resblock
                       |
                      ...
                    resblock
                       |
                neareast_interp_op
                       |
                  __xpu__conv2d
                       |
                     output

Fuse to:
                     Input
                       |
            __xpu__up_decoder_op
                       |
                     Output
*/

class XPUUpDecoderFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    // TODO(shenyijun01): Supporting two inputs will be completed later on.
    // UpDecoder block only has 3~4 resblocks.
    for (auto num_resblock : {4, 3}) {
      for (auto first_resblock_has_input_max : {true, false}) {
        for (auto has_interp : {true, false}) {
          if (has_interp) {
            for (auto post_interp_conv : {true}) {
              if (post_interp_conv) {
                for (auto post_conv_input_max : {true, false}) {
                  fusion::XPUUpDecoderFuser fuser(num_resblock,
                                                  has_interp,
                                                  post_interp_conv,
                                                  post_conv_input_max,
                                                  first_resblock_has_input_max);
                  fuser(graph.get());
                }
              } else {
                fusion::XPUUpDecoderFuser fuser(num_resblock,
                                                has_interp,
                                                false,
                                                false,
                                                first_resblock_has_input_max);
                fuser(graph.get());
              }
            }
          } else {
            fusion::XPUUpDecoderFuser fuser(num_resblock,
                                            false,
                                            false,
                                            false,
                                            first_resblock_has_input_max);
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

REGISTER_MIR_PASS(__xpu__up_decoder_fuse_pass,
                  paddle::lite::mir::XPUUpDecoderFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__up_decoder");
