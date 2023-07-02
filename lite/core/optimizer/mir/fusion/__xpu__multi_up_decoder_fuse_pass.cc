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

class XPUMultiUpDecoderFusePass : public FuseBase {
 public:
  XPUMultiUpDecoderFusePass(
      int num_up_decoder,
      std::vector<int> num_resblock_per_up_decoder,
      std::vector<bool> has_post_interp_conv_per_up_decoder,
      std::vector<bool> has_post_interp_conv_input_max_per_up_decoder,
      std::vector<bool> has_input_max_per_up_decoder,
      bool has_last_gn_silu)
      : num_up_decoders_(num_up_decoder),
        num_resblock_per_up_decoder_(num_resblock_per_up_decoder),
        has_post_interp_conv_per_up_decoder_(
            has_post_interp_conv_per_up_decoder),
        has_post_interp_conv_input_max_per_up_decoder_(
            has_post_interp_conv_input_max_per_up_decoder),
        has_input_max_per_up_decoder_(has_input_max_per_up_decoder),
        has_last_gn_silu_(has_last_gn_silu) {
    all_inputs_map_["Input"] = {};
    all_inputs_map_["AllUpDecoderResConvBias"] = {};
    all_inputs_map_["AllUpDecoderResConvFilter"] = {};
    all_inputs_map_["AllUpDecoderGNScale"] = {};
    all_inputs_map_["AllUpDecoderGNBias"] = {};
    all_inputs_map_["AllUpDecoderInputMax"] = {};
    all_inputs_map_["AllUpDecoderPostConvFilter"] = {};
    all_inputs_map_["AllUpDecoderPostConvBias"] = {};
    all_inputs_map_["AllUpDecoderPostConvInputMax"] = {};
    all_inputs_map_["LastGNScale"] = {};
    all_inputs_map_["LastGNBias"] = {};

    all_up_decoders_op_keys_.clear();
  }

  NodeContainer BuildSingleUpDecoder(int op_pos,
                                     int num_resblock,
                                     bool has_post_interp_conv,
                                     bool has_post_interp_conv_input_max,
                                     bool has_input_max) {
    // op_pos ranges from 0 to N;
    // 0: begining op; (1~N-1): intermediate op, N: the end op;
    // Single up_decoder inputs
    std::map<std::string, PMNode*> input_nodes;
    std::map<std::string, PMNode*> op_node;
    std::map<std::string, PMNode*> output_nodes;
    NodeContainer nodes_pack;
    PMNode* input_1 = nullptr;
    PMNode* input_max = nullptr;
    PMNode* post_conv_input_max = nullptr;
    PMNode* post_conv_filter = nullptr;
    PMNode* post_conv_bias = nullptr;
    int num_resblock_conv =
        has_input_max ? num_resblock * 2 + 1 : num_resblock * 2;
    int num_resblock_gn = num_resblock * 2;

    // Single up_decoder inputs
    if (op_pos == 0) {
      input_1 = VarNode("input_1_" + to_string(op_pos))
                    ->assert_is_op_input("__xpu__up_decoder", "Input")
                    ->AsInput();
      all_inputs_map_["Input"].push_back("input_1_" + to_string(op_pos));
    }
    if (has_input_max) {
      input_max =
          VarNode("input_max_" + to_string(op_pos))
              ->assert_is_op_input("__xpu__up_decoder", "ResblockInputMax")
              ->AsInput();
      all_inputs_map_["AllUpDecoderInputMax"].push_back("input_max_" +
                                                        to_string(op_pos));
    }
    if (has_post_interp_conv) {
      post_conv_filter =
          VarNode("post_conv_filter_" + to_string(op_pos))
              ->assert_is_op_input("__xpu__up_decoder", "PostConvFilter")
              ->AsInput();
      all_inputs_map_["AllUpDecoderPostConvFilter"].push_back(
          "post_conv_filter_" + to_string(op_pos));
      post_conv_bias =
          VarNode("post_conv_bias_" + to_string(op_pos))
              ->assert_is_op_input("__xpu__up_decoder", "PostConvBias")
              ->AsInput();
      all_inputs_map_["AllUpDecoderPostConvBias"].push_back("post_conv_bias_" +
                                                            to_string(op_pos));
      if (has_post_interp_conv_input_max) {
        post_conv_input_max =
            VarNode("post_conv_input_max_" + to_string(op_pos))
                ->assert_is_op_input("__xpu__up_decoder", "PostConvInputMax")
                ->AsInput();
        all_inputs_map_["AllUpDecoderPostConvInputMax"].push_back(
            "post_conv_input_max_" + to_string(op_pos));
      }
    }
    input_nodes["input_1"] = input_1;
    input_nodes["input_max"] = input_max;
    input_nodes["post_conv_input_max"] = post_conv_input_max;
    input_nodes["post_conv_filter"] = post_conv_filter;
    input_nodes["post_conv_bias"] = post_conv_bias;

    // In each up_decoder, all resblock conv params in each up_decoder
    for (int i = 0; i < num_resblock_conv; i++) {
      auto resblock_conv_filter =
          VarNode("resblock_conv_filter_" + to_string(i) + "_" +
                  to_string(op_pos))
              ->assert_is_op_nth_input(
                  "__xpu__up_decoder", "ResblockConvFilter", i)
              ->AsInput();
      all_inputs_map_["AllUpDecoderResConvFilter"].push_back(
          "resblock_conv_filter_" + to_string(i) + "_" + to_string(op_pos));

      auto resblock_conv_bias =
          VarNode("resblock_conv_bias_" + to_string(i) + "_" +
                  to_string(op_pos))
              ->assert_is_op_nth_input(
                  "__xpu__up_decoder", "ResblockConvBias", i)
              ->AsInput();
      all_inputs_map_["AllUpDecoderResConvBias"].push_back(
          "resblock_conv_bias_" + to_string(i) + "_" + to_string(op_pos));

      input_nodes["resblock_conv_filter_" + to_string(i)] =
          resblock_conv_filter;
      input_nodes["resblock_conv_bias_" + to_string(i)] = resblock_conv_bias;
    }

    // In each up_decoder, all resblock gn params in each up_decoder
    for (int i = 0; i < num_resblock_gn; i++) {
      auto resblock_gn_scale =
          VarNode("resblock_gn_scale_" + to_string(i) + "_" + to_string(op_pos))
              ->assert_is_op_nth_input(
                  "__xpu__up_decoder", "ResblockGNScale", i)
              ->AsInput();
      all_inputs_map_["AllUpDecoderGNScale"].push_back(
          "resblock_gn_scale_" + to_string(i) + "_" + to_string(op_pos));

      auto resblock_gn_bias =
          VarNode("resblock_gn_bias_" + to_string(i) + "_" + to_string(op_pos))
              ->assert_is_op_nth_input("__xpu__up_decoder", "ResblockGNBias", i)
              ->AsInput();
      all_inputs_map_["AllUpDecoderGNBias"].push_back(
          "resblock_gn_bias_" + to_string(i) + "_" + to_string(op_pos));

      input_nodes["resblock_gn_scale_" + to_string(i)] = resblock_gn_scale;
      input_nodes["resblock_gn_bias_" + to_string(i)] = resblock_gn_bias;
    }

    // Single up_decoder output
    PMNode* output = VarNode("output_" + to_string(op_pos))
                         ->assert_is_op_output("__xpu__up_decoder", "Output");
    if (op_pos == (num_up_decoders_ - 1) && has_last_gn_silu_ == false) {
      output->AsOutput();
    } else {
      output->AsIntermediate();
    }
    output_nodes["output"] = output;

    // Single up_decoder op node
    PMNode* up_decoder_op =
        OpNode("up_decoder_" + to_string(op_pos), "__xpu__up_decoder")
            ->AsIntermediate();
    all_up_decoders_op_keys_.push_back("up_decoder_" + to_string(op_pos));
    op_node["up_decoder_op"] = up_decoder_op;

    nodes_pack.emplace_back(input_nodes);
    nodes_pack.emplace_back(output_nodes);
    nodes_pack.emplace_back(op_node);

    return nodes_pack;
  }

  void BuildPattern() override {
    std::vector<NodeContainer> up_decoders;
    for (int i = 0; i < num_up_decoders_; ++i) {
      up_decoders.emplace_back(BuildSingleUpDecoder(
          i,
          num_resblock_per_up_decoder_[i],
          has_post_interp_conv_per_up_decoder_[i],
          has_post_interp_conv_input_max_per_up_decoder_[i],
          has_input_max_per_up_decoder_[i]));
    }
    for (int i = 0; i < up_decoders.size(); ++i) {
      std::vector<PMNode*> single_up_decoder_inputs;
      for (auto ele : up_decoders[i][0]) {
        if (ele.second != nullptr) {
          single_up_decoder_inputs.push_back(ele.second);
        }
      }
      if (i != 0) {
        single_up_decoder_inputs.push_back(up_decoders[i - 1][1]["output"]);
      }
      PMNode* single_up_decoder_output = up_decoders[i][1]["output"];
      PMNode* single_up_decoder_op = up_decoders[i][2]["up_decoder_op"];
      single_up_decoder_inputs >> *single_up_decoder_op >>
          *single_up_decoder_output;
    }

    // The last group norm + silu op.
    PMNode* last_gn_silu_scale = nullptr;
    PMNode* last_gn_silu_bias = nullptr;
    PMNode* last_gn_silu_output = nullptr;
    PMNode* last_gn_silu_op = nullptr;

    if (has_last_gn_silu_) {
      last_gn_silu_scale = VarNode("last_gn_silu_scale")
                               ->assert_is_op_input("__xpu__gn_silu", "GNScale")
                               ->AsInput();
      last_gn_silu_bias = VarNode("last_gn_silu_bias")
                              ->assert_is_op_input("__xpu__gn_silu", "GNBias")
                              ->AsInput();
      all_inputs_map_["LastGNScale"].push_back("last_gn_silu_scale");
      all_inputs_map_["LastGNBias"].push_back("last_gn_silu_bias");

      last_gn_silu_output =
          VarNode("last_gn_silu_output")
              ->assert_is_op_output("__xpu__gn_silu", "Output")
              ->AsOutput();
      last_gn_silu_op =
          OpNode("last_gn_silu", "__xpu__gn_silu")->AsIntermediate();
      std::vector<PMNode*> last_gn_silu_input = {
          up_decoders[num_up_decoders_ - 1][1]["output"],
          last_gn_silu_scale,
          last_gn_silu_bias};
      last_gn_silu_input >> *last_gn_silu_op >> *last_gn_silu_output;
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__multi_up_decoder");

    // Set new op input/output.
    std::vector<std::string> all_resblocks_conv_weight_max_names;
    std::vector<std::string> all_post_conv_weight_max_names;
    std::vector<std::string> all_inputs_node_names;
    std::vector<std::vector<int>> all_up_decoders_conv_fix;
    std::vector<std::vector<int>> all_up_decoders_res_conv_dilations;
    std::vector<std::vector<int>> all_up_decoders_res_conv_filter_dims;
    std::vector<std::vector<int>> all_up_decoders_res_conv_paddings;
    std::vector<std::vector<int>> all_up_decoders_res_conv_strides;
    std::vector<std::vector<int>> all_up_decoders_res_conv_groups;
    std::vector<std::vector<float>> all_up_decoders_res_gneps;
    std::vector<std::vector<int>> all_up_decoders_res_gngroups;
    std::vector<std::vector<int>> all_up_decoders_post_conv_dilations;
    std::vector<std::vector<int>> all_up_decoders_post_conv_filter_dims;
    std::vector<std::vector<int>> all_up_decoders_post_conv_paddings;
    std::vector<std::vector<int>> all_up_decoders_post_conv_groups;
    std::vector<std::vector<int>> all_up_decoders_post_conv_strides;
    std::vector<std::vector<int>> all_up_decoders_interp_dhw;
    std::vector<std::vector<float>> all_up_decoders_interp_scale;
    std::vector<std::string> all_up_decoders_interp_method;
    std::vector<int> all_up_decoders_interp_align_corners;
    std::vector<int> all_up_decoders_has_interp_conv;
    int tmp_interp_d = -1000;
    int tmp_interp_h = -1000;
    int tmp_interp_w = -1000;
    std::string subgraph_output_key;

    for (auto input_ele : all_inputs_map_) {
      std::vector<std::string> input_names;
      for (int i = 0; i < input_ele.second.size(); i++) {
        input_names.push_back(matched.at(input_ele.second[i])->arg()->name);
        all_inputs_node_names.push_back(input_ele.second[i]);
      }
      op_desc.SetInput(input_ele.first, {input_names});
    }
    if (has_last_gn_silu_) {
      op_desc.SetOutput("Output",
                        {matched.at("last_gn_silu_output")->arg()->name});
      subgraph_output_key = "last_gn_silu_output";
    } else {
      op_desc.SetOutput(
          "Output",
          {matched.at("output_" + to_string(num_up_decoders_ - 1))
               ->arg()
               ->name});
      subgraph_output_key = "output_" + to_string(num_up_decoders_ - 1);
    }
    // Set new op attributes.
    for (int i = 0; i < all_up_decoders_op_keys_.size(); i++) {
      // acquire conv max names
      auto* up_decoder_op_info =
          matched.at(all_up_decoders_op_keys_[i])->stmt()->op_info();
      for (const auto& name :
           up_decoder_op_info->GetAttr<std::vector<std::string>>(
               "ResblockConvFilterMaxs")) {
        all_resblocks_conv_weight_max_names.push_back(name);
      }
      auto has_interp_conv =
          up_decoder_op_info->GetAttr<bool>("PostInterpConv");

      if (has_interp_conv) {
        for (const auto& name :
             up_decoder_op_info->GetAttr<std::vector<std::string>>(
                 "PostConvFilterMax")) {
          all_post_conv_weight_max_names.push_back(name);
        }

        all_up_decoders_post_conv_dilations.push_back(
            up_decoder_op_info->GetAttr<std::vector<int>>("PostConvDilations"));
        all_up_decoders_post_conv_paddings.push_back(
            up_decoder_op_info->GetAttr<std::vector<int>>("PostConvPaddings"));
        all_up_decoders_post_conv_strides.push_back(
            up_decoder_op_info->GetAttr<std::vector<int>>("PostConvStrides"));
        all_up_decoders_post_conv_filter_dims.push_back(
            up_decoder_op_info->GetAttr<std::vector<int>>(
                "PostConvFilterDims"));
        all_up_decoders_post_conv_groups.push_back(
            up_decoder_op_info->GetAttr<std::vector<int>>("PostConvGroups"));

        tmp_interp_d = up_decoder_op_info->GetAttr<int>("interp_out_d");
        tmp_interp_h = up_decoder_op_info->GetAttr<int>("interp_out_h");
        tmp_interp_w = up_decoder_op_info->GetAttr<int>("interp_out_w");
        all_up_decoders_interp_scale.push_back(
            up_decoder_op_info->GetAttr<std::vector<float>>("interp_scale"));
        all_up_decoders_interp_align_corners.push_back(static_cast<int>(
            up_decoder_op_info->GetAttr<bool>("interp_align_corners")));
        all_up_decoders_interp_method.push_back(
            up_decoder_op_info->GetAttr<std::string>("interp_method"));
        all_up_decoders_has_interp_conv.push_back(1);
      } else {
        all_post_conv_weight_max_names.push_back("none");
        all_up_decoders_post_conv_dilations.push_back({-1000, -1000});
        all_up_decoders_post_conv_paddings.push_back(
            {-1000, -1000, -1000, -1000});
        all_up_decoders_post_conv_strides.push_back({-1000, -1000});
        all_up_decoders_post_conv_filter_dims.push_back(
            {-1000, -1000, -1000, -1000});
        all_up_decoders_post_conv_groups.push_back({-1000});
        all_up_decoders_interp_scale.push_back({-1000, -1000});
        all_up_decoders_interp_align_corners.push_back(-1000);
        all_up_decoders_interp_method.push_back("none");
        all_up_decoders_has_interp_conv.push_back(0);
        tmp_interp_d = -1000;
        tmp_interp_h = -1000;
        tmp_interp_w = -1000;
      }
      // acquire up-decoders convs/group norm params.
      all_up_decoders_conv_fix.push_back(
          up_decoder_op_info->GetAttr<std::vector<int>>("ResblockConvFix"));
      all_up_decoders_res_conv_dilations.push_back(
          up_decoder_op_info->GetAttr<std::vector<int>>(
              "ResblockConvDilations"));
      all_up_decoders_res_conv_paddings.push_back(
          up_decoder_op_info->GetAttr<std::vector<int>>(
              "ResblockConvPaddings"));
      all_up_decoders_res_conv_strides.push_back(
          up_decoder_op_info->GetAttr<std::vector<int>>("ResblockConvStrides"));
      all_up_decoders_res_conv_filter_dims.push_back(
          up_decoder_op_info->GetAttr<std::vector<int>>(
              "ResblockConvFilterDims"));
      all_up_decoders_res_conv_groups.push_back(
          up_decoder_op_info->GetAttr<std::vector<int>>("ResblockConvGroups"));

      all_up_decoders_res_gneps.push_back(
          up_decoder_op_info->GetAttr<std::vector<float>>("ResblockGNEps"));
      all_up_decoders_res_gngroups.push_back(
          up_decoder_op_info->GetAttr<std::vector<int>>("ResblockGNGroups"));

      all_up_decoders_interp_dhw.push_back(
          {tmp_interp_d, tmp_interp_h, tmp_interp_w});
    }
    // Set all attributes
    op_desc.SetAttr<std::vector<std::string>>(
        "ResConvFilterMaxs", all_resblocks_conv_weight_max_names);
    op_desc.SetAttr<std::vector<std::string>>("PostConvFilterMaxs",
                                              all_post_conv_weight_max_names);
    op_desc.SetAttr<std::vector<int>>(
        "ResConvFixInfo",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_conv_fix));
    op_desc.SetAttr<std::vector<int>>(
        "ResConvDilations",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_res_conv_dilations));
    op_desc.SetAttr<std::vector<int>>(
        "ResConvFilterDims",
        paddle::lite::xpu::vec::Vec2DTo1D(
            all_up_decoders_res_conv_filter_dims));
    op_desc.SetAttr<std::vector<int>>(
        "ResConvPaddings",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_res_conv_paddings));
    op_desc.SetAttr<std::vector<int>>(
        "ResConvStrides",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_res_conv_strides));
    op_desc.SetAttr<std::vector<int>>(
        "ResConvGroups",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_res_conv_groups));
    op_desc.SetAttr<std::vector<int>>(
        "PostConvDilations",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_post_conv_dilations));
    op_desc.SetAttr<std::vector<int>>(
        "PostConvFilterDims",
        paddle::lite::xpu::vec::Vec2DTo1D(
            all_up_decoders_post_conv_filter_dims));
    op_desc.SetAttr<std::vector<int>>(
        "PostConvPaddings",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_post_conv_paddings));
    op_desc.SetAttr<std::vector<int>>(
        "PostConvStrides",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_post_conv_strides));
    op_desc.SetAttr<std::vector<int>>(
        "PostConvGroups",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_post_conv_groups));

    op_desc.SetAttr<std::vector<float>>(
        "ResblockGNEps",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_res_gneps));
    op_desc.SetAttr<std::vector<int>>(
        "ResblockGNGroups",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_res_gngroups));

    op_desc.SetAttr<std::vector<int>>(
        "PostInterpOutDHW",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_interp_dhw));
    op_desc.SetAttr<std::vector<float>>(
        "PostInterpOutScale",
        paddle::lite::xpu::vec::Vec2DTo1D(all_up_decoders_interp_scale));
    op_desc.SetAttr<std::vector<int>>("PostInterpAlignCorners",
                                      all_up_decoders_interp_align_corners);
    op_desc.SetAttr<std::vector<std::string>>("PostInterpMethods",
                                              all_up_decoders_interp_method);

    op_desc.SetAttr<std::vector<int>>("ResblockNumPerUpDecoder",
                                      num_resblock_per_up_decoder_);
    op_desc.SetAttr<std::vector<int>>("HasInterp",
                                      all_up_decoders_has_interp_conv);
    op_desc.SetAttr<bool>("HasLastGNSilu", has_last_gn_silu_);
    if (has_last_gn_silu_) {
      auto* last_gn_silu_op_info =
          matched.at("last_gn_silu")->stmt()->op_info();
      auto eps = last_gn_silu_op_info->GetAttr<float>("epsilon");
      auto groups = last_gn_silu_op_info->GetAttr<int>("groups");
      op_desc.SetAttr<float>("LastGNEps", eps);
      op_desc.SetAttr<int>("LastGNGroups", groups);
    }
    // Achieve scope of subgraph's first op and use it to set new op.
    auto* scope =
        matched.at(all_up_decoders_op_keys_[0])->stmt()->op()->scope();
    auto multi_up_decoder_op = LiteOpRegistry::Global().Create(op_desc.Type());
    multi_up_decoder_op->Attach(op_desc, scope);
    multi_up_decoder_op->SetValidPlaces(
        matched.at(all_up_decoders_op_keys_[0])->stmt()->op()->valid_places());
    auto kernels =
        multi_up_decoder_op->CreateKernels(multi_up_decoder_op->valid_places());
    auto* new_op_node = graph->GraphCreateInstructNode(
        multi_up_decoder_op, multi_up_decoder_op->valid_places());

    for (auto& from : all_inputs_node_names) {
      IR_NODE_LINK_TO(matched.at(from), new_op_node);
    }
    IR_OP_VAR_LINK(new_op_node, matched.at(subgraph_output_key));
  }

 private:
  int num_up_decoders_;
  std::vector<int> num_resblock_per_up_decoder_;
  std::vector<bool> has_post_interp_conv_per_up_decoder_;
  std::vector<bool> has_post_interp_conv_input_max_per_up_decoder_;
  std::vector<bool> has_input_max_per_up_decoder_;
  std::map<std::string, std::vector<std::string>> all_inputs_map_;
  std::vector<std::string> all_up_decoders_op_keys_;
  bool has_last_gn_silu_;
};
}  // namespace fusion

/*
Fuse multiple up-decoders and group-norm+silu(optional) into one layer.

Original subgraph:

                     Input
                       |
                  updecoder_1
                       |
                  updecoder_2
                       |
                      ...
                  updecoder_n
                       |
                 __xpu__gn_silu (optional)
                       |
                     output

Fuse to:
                     Input
                       |
          __xpu__multi_up_decoder_op
                       |
                     Output
*/

class XPUMultiUpDecoderFusePass : public ProgramPass {
 public:
  // TODO(shenyijun01): Currently, the multi-up--decoder op will be fused by
  // fixed pattern with fixed num of up-decoders.
  const std::vector<std::vector<int>> num_resblock_per_up_decoder = {
      {4, 3, 3, 3, 3}, {4, 3, 3, 3}};
  const std::vector<std::vector<bool>> has_post_interp_conv_per_up_decoder = {
      {true, true, true, true, false}, {true, true, true, false}};
  const std::vector<std::vector<bool>>
      has_post_interp_conv_input_max_per_up_decoder = {
          {false, false, false, false, false}, {false, false, false, false}};
  const std::vector<std::vector<bool>> has_input_max_per_up_decoder = {
      {false, false, false, true, true}, {false, false, true, true}};

  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (int i = 0; i < num_resblock_per_up_decoder.size(); ++i) {
      const int num_up_decoder =
          static_cast<int>(num_resblock_per_up_decoder[i].size());
      CHECK_EQ(num_up_decoder,
               static_cast<int>(has_post_interp_conv_per_up_decoder[i].size()));
      CHECK_EQ(num_up_decoder,
               static_cast<int>(
                   has_post_interp_conv_input_max_per_up_decoder[i].size()));
      CHECK_EQ(num_up_decoder,
               static_cast<int>(has_input_max_per_up_decoder[i].size()));
      for (auto has_last_gn_silu : {true, false}) {
        fusion::XPUMultiUpDecoderFusePass fuser(
            num_up_decoder,
            num_resblock_per_up_decoder[i],
            has_post_interp_conv_per_up_decoder[i],
            has_post_interp_conv_input_max_per_up_decoder[i],
            has_input_max_per_up_decoder[i],
            has_last_gn_silu);
        fuser(graph.get());
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__multi_up_decoder_fuse_pass,
                  paddle::lite::mir::XPUMultiUpDecoderFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__multi_up_decoder");
