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
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

static std::vector<int> vec2DTo1D_int(
    const std::vector<std::vector<int>>& vec) {
  std::vector<int> res;
  for (const auto& v : vec) {
    for (const auto& ele : v) {
      res.emplace_back(ele);
    }
  }
  return res;
}

class UnetResBlockfuser : public FuseBase {
 public:
  explicit UnetResBlockfuser(bool conv_fix, bool input_max)
      : conv_fix_(conv_fix), input_max_(input_max) {}

  void BuildPattern() override {
    auto* input_1 = VarNode("input_1")
                        ->assert_is_op_input("__xpu__gn_silu", "Input")
                        ->AsInput();
    auto* gn_scale_1 = VarNode("gn_scale_1")
                           ->assert_is_op_input("__xpu__gn_silu", "GNScale")
                           ->AsInput();
    auto* gn_bias_1 = VarNode("gn_bias_1")
                          ->assert_is_op_input("__xpu__gn_silu", "GNBias")
                          ->AsInput();
    auto* gn_silu_1 = OpNode("gn_silu_1", "__xpu__gn_silu")->AsIntermediate();
    auto* gn_silu_out_1 = VarNode("gn_silu_1_out")
                              ->assert_is_op_output("__xpu__gn_silu", "Output")
                              ->assert_is_op_input("__xpu__conv2d", "Input")
                              ->AsIntermediate();
    auto* input_2 =
        VarNode("input_2")->assert_is_op_input("silu", "X")->AsInput();
    auto* silu = OpNode("silu", "silu")->AsIntermediate();
    auto* silu_out = VarNode("silu_out")
                         ->assert_is_op_output("silu", "Out")
                         ->assert_is_op_input("__xpu__fc", "Input")
                         ->AsIntermediate();
    auto* fc_weight = VarNode("fc_weight")
                          ->assert_is_op_input("__xpu__fc", "Filter")
                          ->AsInput();
    auto* fc_bias =
        VarNode("fc_bias")->assert_is_op_input("__xpu__fc", "Bias")->AsInput();
    auto* fc = OpNode("fc", "__xpu__fc")->AsIntermediate();
    auto* fc_out = VarNode("fc_out")
                       ->assert_is_op_output("__xpu__fc", "Output")
                       ->assert_is_op_input("unsqueeze2", "X")
                       ->AsIntermediate();
    auto* fc_max = VarNode("fc_max")
                       ->assert_is_op_output("__xpu__fc", "OutputMax")
                       ->AsIntermediate();
    auto* unsqueeze = OpNode("unsqueeze", "unsqueeze2")->AsIntermediate();
    auto* unsqueeze_out = VarNode("unsqueeze_out")
                              ->assert_is_op_output("unsqueeze2", "Out")
                              ->assert_is_op_input("__xpu__conv2d", "Branch")
                              ->AsOutput();
    auto* unsqueeze_out_shape =
        VarNode("unsqueeze_out_shape")
            ->assert_is_op_output("unsqueeze2", "XShape")
            ->AsIntermediate();
    auto* conv_filter_1 = VarNode("conv_filter_1")
                              ->assert_is_op_input("__xpu__conv2d", "Filter")
                              ->AsInput();
    auto* conv_bias_1 = VarNode("conv_bias_1")
                            ->assert_is_op_input("__xpu__conv2d", "Bias")
                            ->AsInput();
    auto* conv_1 = OpNode("conv_1", "__xpu__conv2d")->AsIntermediate();
    auto* conv_out_1 = VarNode("conv_out_1")
                           ->assert_is_op_output("__xpu__conv2d", "Output")
                           ->assert_is_op_input("__xpu__gn_silu", "Input")
                           ->AsIntermediate();
    auto* conv_max_1 = VarNode("conv_max_1")
                           ->assert_is_op_output("__xpu__conv2d", "OutputMax")
                           ->AsIntermediate();
    auto* gn_scale_2 = VarNode("gn_scale_2")
                           ->assert_is_op_input("__xpu__gn_silu", "GNScale")
                           ->AsInput();
    auto* gn_bias_2 = VarNode("gn_bias_2")
                          ->assert_is_op_input("__xpu__gn_silu", "GNBias")
                          ->AsInput();
    auto* gn_silu_2 = OpNode("gn_silu_2", "__xpu__gn_silu");
    auto* gn_silu_out_2 = VarNode("gn_silu_2_out")
                              ->assert_is_op_output("__xpu__gn_silu", "Output")
                              ->assert_is_op_input("__xpu__conv2d", "Input")
                              ->AsIntermediate();
    PMNode* conv_3 = nullptr;
    PMNode* conv_filter_3 = nullptr;
    PMNode* conv_bias_3 = nullptr;
    PMNode* conv_out_3 = nullptr;
    PMNode* conv_max_3 = nullptr;
    PMNode* input_max_3 = nullptr;
    if (conv_fix_) {
      conv_filter_3 = VarNode("conv_filter_3")
                          ->assert_is_op_input("__xpu__conv2d", "Filter")
                          ->AsInput();
      conv_bias_3 = VarNode("conv_bias_3")
                        ->assert_is_op_input("__xpu__conv2d", "Bias")
                        ->AsInput();
      if (input_max_) {
        input_max_3 = VarNode("input_max_3")
                          ->assert_is_op_input("__xpu__conv2d", "InputMax")
                          ->AsInput();
      }
      conv_3 = OpNode("conv_3", "__xpu__conv2d")->AsIntermediate();
      conv_out_3 = VarNode("conv_out_3")
                       ->assert_is_op_output("__xpu__conv2d", "Output")
                       ->assert_is_op_input("__xpu__conv2d", "Branch")
                       ->AsIntermediate();
      conv_max_3 = VarNode("conv_max_3")
                       ->assert_is_op_output("__xpu__conv2d", "OutputMax")
                       ->AsIntermediate();
    }

    auto* conv_filter_2 = VarNode("conv_filter_2")
                              ->assert_is_op_input("__xpu__conv2d", "Filter")
                              ->AsInput();
    auto* conv_bias_2 = VarNode("conv_bias_2")
                            ->assert_is_op_input("__xpu__conv2d", "Bias")
                            ->AsInput();
    auto* conv_2 = OpNode("conv_2", "__xpu__conv2d")->AsIntermediate();
    auto* conv_out_2 = VarNode("conv_out_2")
                           ->assert_is_op_output("__xpu__conv2d", "Output")
                           ->AsOutput();
    auto* conv_max_2 = VarNode("conv_max_2")
                           ->assert_is_op_output("__xpu__conv2d", "OutputMax")
                           ->AsIntermediate();

    std::vector<PMNode*> gn_silu1_input{input_1, gn_scale_1, gn_bias_1};
    std::vector<PMNode*> gn_silu2_input{conv_out_1, gn_scale_2, gn_bias_2};
    std::vector<PMNode*> fc_input{silu_out, fc_weight, fc_bias};
    std::vector<PMNode*> fc_output{fc_out, fc_max};
    std::vector<PMNode*> conv1_input{
        gn_silu_out_1, conv_filter_1, conv_bias_1, unsqueeze_out};
    std::vector<PMNode*> conv1_output{conv_out_1, conv_max_1};
    std::vector<PMNode*> conv2_input{gn_silu_out_2, conv_filter_2, conv_bias_2};
    std::vector<PMNode*> conv2_output{conv_out_2, conv_max_2};
    if (conv_fix_) {
      conv2_input.push_back(conv_out_3);
      std::vector<PMNode*> conv3_input{input_1, conv_filter_3, conv_bias_3};
      if (input_max_) {
        conv3_input.push_back(input_max_3);
      }
      std::vector<PMNode*> conv3_output{conv_out_3, conv_max_3};
      conv3_input >> *conv_3 >> conv3_output;
    }
    *input_2 >> *silu >> *silu_out;
    fc_input >> *fc >> fc_output;
    *fc_out >> *unsqueeze >> *unsqueeze_out;
    *unsqueeze >> *unsqueeze_out_shape;
    gn_silu1_input >> *gn_silu_1 >> *gn_silu_out_1;
    conv1_input >> *conv_1 >> conv1_output;
    gn_silu2_input >> *gn_silu_2 >> *gn_silu_out_2;
    conv2_input >> *conv_2 >> conv2_output;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__unet_resblock");
    std::vector<std::string> fc_weight_names = {
        matched.at("fc_weight")->arg()->name,
    };
    std::vector<std::string> conv_filter_names = {
        matched.at("conv_filter_1")->arg()->name,
        matched.at("conv_filter_2")->arg()->name,
    };
    if (conv_fix_) {
      conv_filter_names.push_back(matched.at("conv_filter_3")->arg()->name);
    }
    std::vector<std::string> fc_weight_maxptr_names;
    for (size_t i = 0; i < fc_weight_names.size(); i++) {
      fc_weight_maxptr_names.push_back(fc_weight_names[i] + "_max");
    }
    std::vector<std::string> conv_filter_maxptr_names;
    for (size_t i = 0; i < conv_filter_names.size(); i++) {
      conv_filter_maxptr_names.push_back(conv_filter_names[i] + "_max");
    }
    if (input_max_) {
      op_desc.SetInput("InputMax", {matched.at("input_max_3")->arg()->name});
    } else {
      op_desc.SetInput("InputMax", {});
    }
    op_desc.SetInput("Input1", {matched.at("input_1")->arg()->name});
    op_desc.SetInput("Input2", {matched.at("input_2")->arg()->name});
    op_desc.SetInput("FCWeight", fc_weight_names);
    op_desc.SetInput("ConvFilter", conv_filter_names);
    op_desc.SetInput("FCBias",
                     {
                         matched.at("fc_bias")->arg()->name,
                     });
    if (conv_fix_) {
      op_desc.SetInput("ConvBias",
                       {
                           matched.at("conv_bias_1")->arg()->name,
                           matched.at("conv_bias_2")->arg()->name,
                           matched.at("conv_bias_3")->arg()->name,
                       });
    } else {
      op_desc.SetInput("ConvBias",
                       {
                           matched.at("conv_bias_1")->arg()->name,
                           matched.at("conv_bias_2")->arg()->name,
                       });
    }
    op_desc.SetInput("GNScale",
                     {
                         matched.at("gn_scale_1")->arg()->name,
                         matched.at("gn_scale_2")->arg()->name,
                     });
    op_desc.SetInput("GNBias",
                     {
                         matched.at("gn_bias_1")->arg()->name,
                         matched.at("gn_bias_2")->arg()->name,
                     });
    op_desc.SetOutput("Output", {matched.at("conv_out_2")->arg()->name});
    op_desc.SetAttr<std::vector<std::string>>("FCWeightMax",
                                              fc_weight_maxptr_names);
    op_desc.SetAttr<std::vector<std::string>>("ConvFilterMax",
                                              conv_filter_maxptr_names);
    auto* scope = matched.at("gn_silu_2")->stmt()->op()->scope();
    update_weight(scope, fc_weight_names, fc_weight_maxptr_names, true);
    update_weight(scope, conv_filter_names, conv_filter_maxptr_names, false);

    std::vector<std::vector<int>> strides;
    std::vector<std::vector<int>> paddings;
    std::vector<std::vector<int>> dilations;
    std::vector<std::vector<int>> filter_dims;
    std::vector<int> groups;
    std::vector<std::string> conv_vec = {"conv_1", "conv_2"};
    if (conv_fix_) {
      conv_vec.emplace_back("conv_3");
    }
    for (auto pm_name : conv_vec) {
      auto* conv_op_info = matched.at(pm_name)->stmt()->op_info();
      auto strides_tmp = conv_op_info->GetAttr<std::vector<int>>("strides");
      strides.emplace_back(std::move(strides_tmp));
      auto paddings_tmp = conv_op_info->GetAttr<std::vector<int>>("paddings");
      paddings.emplace_back(std::move(paddings_tmp));
      auto dilations_tmp = conv_op_info->GetAttr<std::vector<int>>("dilations");
      dilations.emplace_back(std::move(dilations_tmp));
      std::vector<int> groups_tmp =
          conv_op_info->GetAttr<std::vector<int>>("groups");
      groups.push_back(groups_tmp[0]);
      auto filter_dims_tmp =
          conv_op_info->GetAttr<std::vector<int>>("filter_dims");
      filter_dims.emplace_back(std::move(filter_dims_tmp));
    }
    op_desc.SetAttr<std::vector<int>>("Groups", groups);
    op_desc.SetAttr<std::vector<int>>("Strides", vec2DTo1D_int(strides));
    op_desc.SetAttr<std::vector<int>>("Paddings", vec2DTo1D_int(paddings));
    op_desc.SetAttr<std::vector<int>>("Dilations", vec2DTo1D_int(dilations));
    op_desc.SetAttr<std::vector<int>>("FilterDims", vec2DTo1D_int(filter_dims));

    std::vector<int> gn_groups;
    std::vector<float> gn_eps;
    for (auto pm_name : {"gn_silu_1", "gn_silu_2"}) {
      auto* gnsilu_op_info = matched.at(pm_name)->stmt()->op_info();
      auto gn_group = gnsilu_op_info->GetAttr<int>("groups");
      gn_groups.emplace_back(gn_group);
      auto eps = gnsilu_op_info->GetAttr<float>("epsilon");
      gn_eps.emplace_back(eps);
    }
    op_desc.SetAttr<std::vector<int>>("GNGroups", gn_groups);
    op_desc.SetAttr<std::vector<float>>("GNEps", gn_eps);
    op_desc.SetAttr<bool>("ConvFix", conv_fix_);

    auto resblock_op = LiteOpRegistry::Global().Create(op_desc.Type());
    resblock_op->Attach(op_desc, scope);
    resblock_op->SetValidPlaces(
        matched.at("gn_silu_2")->stmt()->op()->valid_places());
    auto kernels = resblock_op->CreateKernels(resblock_op->valid_places());
    matched.at("gn_silu_2")->stmt()->SetOp(resblock_op);
    matched.at("gn_silu_2")->stmt()->SetKernels(std::move(kernels));

    std::vector<std::string> froms = {
        "input_1",
        "input_2",
        "gn_scale_1",
        "gn_bias_1",
        "fc_bias",
        "fc_weight",
        "conv_filter_1",
        "conv_bias_1",
        "conv_filter_2",
        "conv_bias_2",
    };
    if (conv_fix_) {
      froms.emplace_back("conv_filter_3");
      froms.emplace_back("conv_bias_3");
      if (input_max_) {
        froms.emplace_back("input_max_3");
      }
    }
    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), matched.at("gn_silu_2"));
    }
    IR_OP_VAR_LINK(matched.at("gn_silu_2"), matched.at("conv_out_2"));
  }

 private:
  bool conv_fix_;
  bool input_max_;
  void update_weight(Scope* scope,
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

class XPUUnetResBlockfusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    for (auto conv_fix : {false, true}) {
      if (conv_fix == true) {
        for (auto input_max : {false, true}) {
          fusion::UnetResBlockfuser fuser(conv_fix, input_max);
          fuser(graph.get());
        }
      } else {
        fusion::UnetResBlockfuser fuser(conv_fix, false);
        fuser(graph.get());
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__unet_resblock_fuse_pass,
                  paddle::lite::mir::XPUUnetResBlockfusePass)
    .BindTargets({TARGET(kXPU)});
