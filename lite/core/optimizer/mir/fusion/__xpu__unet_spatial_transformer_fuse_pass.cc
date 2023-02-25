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

class UnetSpatialTransformerfuser : public FuseBase {
public:
  void BuildPattern() override {
    auto* input = VarNode("input")->assert_is_op_input("group_norm", "X")
                                  ->assert_is_op_input("__xpu__conv2d", "Branch")
                                  ->AsInput();

    // image to sequence
    auto* gn_scale = VarNode("gn_scale")
                            ->assert_is_op_input("group_norm", "Scale")
                            ->AsInput();
    auto* gn_bias = VarNode("gn_bias")
                           ->assert_is_op_input("group_norm", "Bias")
                           ->AsInput();
    auto* gn = OpNode("gn", "group_norm")->AsIntermediate();
    auto* gn_out = VarNode("gn_out")
                          ->assert_is_op_output("group_norm", "Y")
                          ->assert_is_op_input("__xpu__conv2d", "Input")
                          ->AsIntermediate();
    auto* gn_mean = VarNode("gn_mean")
                           ->assert_is_op_output("group_norm", "Mean")
                           ->AsIntermediate();
    auto* gn_var = VarNode("gn_var")
                          ->assert_is_op_output("group_norm", "Variance")
                          ->AsIntermediate();

    auto* pre_xpu_conv2d = OpNode("pre__xpu__conv2d", "__xpu__conv2d")->AsIntermediate();
    auto* pre_xpu_conv2d_bias = VarNode("pre__xpu__conv2d_bias")
                           ->assert_is_op_input("__xpu__conv2d", "Bias")
                           ->AsInput();
    auto* pre_xpu_conv2d_filter = VarNode("pre__xpu__conv2d_filter")
                           ->assert_is_op_input("__xpu__conv2d", "Filter")
                           ->AsInput();
    auto* pre_xpu_conv2d_output = VarNode("pre__xpu__conv2d_output")->AsIntermediate()
                            ->assert_is_op_input("transpose2", "X")
                            ->assert_is_op_output("__xpu__conv2d", "Output");
    auto* pre_xpu_conv2d_output_max = VarNode("pre__xpu__conv2d_output_max")->AsIntermediate()
                            ->assert_is_op_output("__xpu__conv2d", "OutputMax");
    
    auto* transpose2 = OpNode("transpose2", "transpose2")->AsIntermediate();
    auto* transpose2_output = VarNode("transpose2_output")->AsIntermediate()
                            ->assert_is_op_output("transpose2", "Out")
                            ->assert_is_op_input("flatten_contiguous_range", "X");
    auto* transpose2_output_xshape = VarNode("transpose2_output_xshape")->AsIntermediate()
                            ->assert_is_op_output("transpose2", "XShape");

    auto* flatten = OpNode("flatten_contiguous_range", "flatten_contiguous_range")->AsIntermediate();
    auto* flatten_output =  VarNode("flatten_output")->AsIntermediate()
                            ->assert_is_op_output("flatten_contiguous_range", "Out")
                            ->assert_is_op_input("__xpu__unet_mhsa", "Input")
                            ->assert_is_op_input("elementwise_add", "Y");
    auto* flatten_output_xshape =  VarNode("flatten_output_xshape")->AsIntermediate()
                            ->assert_is_op_output("flatten_contiguous_range", "XShape");

    // __xpu__unet_mhsa
    auto* __xpu__unet_mhsa = OpNode("__xpu__unet_mhsa", "__xpu__unet_mhsa")->AsIntermediate();
    auto* __xpu__unet_mhsa_fcbias = VarNode("__xpu__unet_mhsa_fcbias")
                           ->assert_is_op_input("__xpu__unet_mhsa", "FCBias")
                           ->AsInput();
    auto* __xpu__unet_mhsa_lnbias = VarNode("__xpu__unet_mhsa_lnbias")
                           ->assert_is_op_input("__xpu__unet_mhsa", "LNBias")
                           ->AsInput();
    auto* __xpu__unet_mhsa_lnscale = VarNode("__xpu__unet_mhsa_lnscale")
                           ->assert_is_op_input("__xpu__unet_mhsa", "LNScale")
                           ->AsInput();
    auto* __xpu__unet_mhsa_fcweight0 = VarNode("__xpu__unet_mhsa_fcweight0")
                           ->assert_is_op_nth_input("__xpu__unet_mhsa", "FCWeight", 0)
                           ->AsInput();
    auto* __xpu__unet_mhsa_fcweight1 = VarNode("__xpu__unet_mhsa_fcweight1")
                           ->assert_is_op_nth_input("__xpu__unet_mhsa", "FCWeight", 1)
                           ->AsInput();
    auto* __xpu__unet_mhsa_fcweight2 = VarNode("__xpu__unet_mhsa_fcweight2")
                           ->assert_is_op_nth_input("__xpu__unet_mhsa", "FCWeight", 2)
                           ->AsInput();
    auto* __xpu__unet_mhsa_fcweight3 = VarNode("__xpu__unet_mhsa_fcweight3")
                           ->assert_is_op_nth_input("__xpu__unet_mhsa", "FCWeight", 3)
                           ->AsInput();
    auto* __xpu__unet_mhsa_output = VarNode("__xpu__unet_mhsa_output")->AsIntermediate()
                                    ->assert_is_op_output("__xpu__unet_mhsa", "Output")
                                    ->assert_is_op_input("elementwise_add", "X");
    auto* residual_add = OpNode("elementwise_add", "elementwise_add")->AsIntermediate();
    auto* residual_add_output = VarNode("residual_add_output")->AsIntermediate()
                                ->assert_is_op_output("elementwise_add", "Out")
                                ->assert_is_op_input("elementwise_add", "Y")
                                ->assert_is_op_input("__xpu__unet_mhca", "Input");

    // __xpu__unet_mhca
    auto* __xpu__unet_mhca = OpNode("__xpu__unet_mhca", "__xpu__unet_mhca")->AsIntermediate();
    auto* __xpu__unet_mhca_embedding = VarNode("__xpu__unet_mhca_embedding")
                           ->assert_is_op_input("__xpu__unet_mhca", "Embedding")
                           ->AsInput();
    auto* __xpu__unet_mhca_fcbias = VarNode("__xpu__unet_mhca_fcbias")
                           ->assert_is_op_input("__xpu__unet_mhca", "FCBias")
                           ->AsInput();
    auto* __xpu__unet_mhca_lnbias = VarNode("__xpu__unet_mhca_lnbias")
                           ->assert_is_op_input("__xpu__unet_mhca", "LNBias")
                           ->AsInput();
    auto* __xpu__unet_mhca_lnscale = VarNode("__xpu__unet_mhca_lnscale")
                           ->assert_is_op_input("__xpu__unet_mhca", "LNScale")
                           ->AsInput();
    auto* __xpu__unet_mhca_fcweight0 = VarNode("__xpu__unet_mhca_fcweight0")
                           ->assert_is_op_nth_input("__xpu__unet_mhca", "FCWeight", 0)
                           ->AsInput();
    auto* __xpu__unet_mhca_fcweight1 = VarNode("__xpu__unet_mhca_fcweight1")
                           ->assert_is_op_nth_input("__xpu__unet_mhca", "FCWeight", 1)
                           ->AsInput();
    auto* __xpu__unet_mhca_fcweight2 = VarNode("__xpu__unet_mhca_fcweight2")
                           ->assert_is_op_nth_input("__xpu__unet_mhca", "FCWeight", 2)
                           ->AsInput();
    auto* __xpu__unet_mhca_fcweight3 = VarNode("__xpu__unet_mhca_fcweight3")
                           ->assert_is_op_nth_input("__xpu__unet_mhca", "FCWeight", 3)
                           ->AsInput();
    auto* __xpu__unet_mhca_output = VarNode("__xpu__unet_mhca_output")->AsIntermediate()
                                    ->assert_is_op_output("__xpu__unet_mhca", "Output")
                                    ->assert_is_op_input("elementwise_add", "X");
    auto* residual_add2 = OpNode("elementwise_add2", "elementwise_add")->AsIntermediate();
    auto* residual_add2_output = VarNode("residual2_add_output")->AsIntermediate()
                                ->assert_is_op_output("elementwise_add", "Out")
                                ->assert_is_op_input("elementwise_add", "Y")
                                ->assert_is_op_input("__xpu__unet_geglu", "Input");

    // geglu
    auto* __xpu__unet_geglu = OpNode("__xpu__unet_geglu", "__xpu__unet_geglu")->AsIntermediate();
    auto* __xpu__unet_geglu_fcbias0 = VarNode("__xpu__unet_geglu_fcbias0")
                           ->assert_is_op_nth_input("__xpu__unet_geglu", "FCBias", 0)
                           ->AsInput();
    auto* __xpu__unet_geglu_fcbias1 = VarNode("__xpu__unet_geglu_fcbias1")
                           ->assert_is_op_nth_input("__xpu__unet_geglu", "FCBias", 1)
                           ->AsInput();
    auto* __xpu__unet_geglu_lnbias = VarNode("__xpu__unet_geglu_lnbias")
                           ->assert_is_op_input("__xpu__unet_geglu", "LNBias")
                           ->AsInput();
    auto* __xpu__unet_geglu_lnscale = VarNode("__xpu__unet_geglu_lnscale")
                           ->assert_is_op_input("__xpu__unet_geglu", "LNScale")
                           ->AsInput();
    auto* __xpu__unet_geglu_fcweight0 = VarNode("__xpu__unet_geglu_fcweight0")
                           ->assert_is_op_nth_input("__xpu__unet_geglu", "FCWeight", 0)
                           ->AsInput();
    auto* __xpu__unet_geglu_fcweight1 = VarNode("__xpu__unet_geglu_fcweight1")
                           ->assert_is_op_nth_input("__xpu__unet_geglu", "FCWeight", 1)
                           ->AsInput();
    auto* __xpu__unet_geglu_output = VarNode("__xpu__unet_geglu_output")->AsIntermediate()
                                    ->assert_is_op_output("__xpu__unet_geglu", "Output")
                                    ->assert_is_op_input("elementwise_add", "X");
    auto* residual_add3 = OpNode("elementwise_add3", "elementwise_add")->AsIntermediate();
    auto* residual_add3_output = VarNode("residual3_add_output")->AsIntermediate()
                                ->assert_is_op_output("elementwise_add", "Out")
                                ->assert_is_op_input("reshape2", "X");


    // sequence to image
    auto* reshape = OpNode("reshape2", "reshape2")->AsIntermediate();
    auto* reshape_output = VarNode("reshape_output")->AsIntermediate()
                            ->assert_is_op_input("transpose2", "X")
                            ->assert_is_op_output("reshape2", "Out");
    auto* reshape_output_xshape = VarNode("reshape_output_xshape")->AsIntermediate()
                            ->assert_is_op_output("reshape2", "XShape");
    auto* transpose2_2 = OpNode("transpose2_2", "transpose2")->AsIntermediate();
    auto* transpose2_2_output = VarNode("transpose2_2_output")->AsIntermediate()
                            ->assert_is_op_input("__xpu__conv2d", "Input")
                            ->assert_is_op_output("transpose2", "Out");
    auto* transpose2_2_output_xshape = VarNode("transpose2_2_output_xshape")->AsIntermediate()
                            ->assert_is_op_output("transpose2", "XShape");
    auto* post_xpu_conv2d = OpNode("post__xpu__conv2d", "__xpu__conv2d")->AsIntermediate();
    auto* post_xpu_conv2d_bias = VarNode("post__xpu__conv2d_bias")
                           ->assert_is_op_input("__xpu__conv2d", "Bias")
                           ->AsInput();
    auto* post_xpu_conv2d_filter = VarNode("post__xpu__conv2d_filter")
                           ->assert_is_op_input("__xpu__conv2d", "Filter")
                           ->AsInput();
    auto* post_xpu_conv2d_output = VarNode("post__xpu__conv2d_output")->AsOutput()
                            ->assert_is_op_output("__xpu__conv2d", "Output");
    auto* post_xpu_conv2d_outputmax = VarNode("post__xpu__conv2d_output_max")->AsIntermediate()
                            ->assert_is_op_output("__xpu__conv2d", "OutputMax");

    std::vector<PMNode*> gn_input{input, gn_bias, gn_scale};
    std::vector<PMNode*> gn_output{gn_out, gn_mean, gn_var};
    gn_input >> *gn >> gn_output;
    std::vector<PMNode*> pre_conv2d_input{gn_out, pre_xpu_conv2d_bias, pre_xpu_conv2d_filter};
    std::vector<PMNode*> pre_conv2d_output{pre_xpu_conv2d_output, pre_xpu_conv2d_output_max};
    pre_conv2d_input >> *pre_xpu_conv2d >> pre_conv2d_output;
    *pre_xpu_conv2d_output >> *transpose2 >> *transpose2_output >> *flatten >> *flatten_output;
    *transpose2 >> *transpose2_output_xshape;
    *flatten >> *flatten_output_xshape;

    std::vector<PMNode*> mhsa_input{flatten_output, __xpu__unet_mhsa_fcbias, __xpu__unet_mhsa_fcweight0, __xpu__unet_mhsa_fcweight1, __xpu__unet_mhsa_fcweight2, __xpu__unet_mhsa_fcweight3, __xpu__unet_mhsa_lnbias, __xpu__unet_mhsa_lnscale};
    mhsa_input >> *__xpu__unet_mhsa >> *__xpu__unet_mhsa_output >> *residual_add >> *residual_add_output;
    *flatten_output >> *residual_add;

    std::vector<PMNode*> mhca_input{residual_add_output, __xpu__unet_mhca_embedding, __xpu__unet_mhca_fcbias, __xpu__unet_mhca_lnbias, __xpu__unet_mhca_lnscale, __xpu__unet_mhca_fcweight0, __xpu__unet_mhca_fcweight1, __xpu__unet_mhca_fcweight2, __xpu__unet_mhca_fcweight3};
    mhca_input >> *__xpu__unet_mhca >> *__xpu__unet_mhca_output >> *residual_add2 >> *residual_add2_output;
    *residual_add_output >> *residual_add2;

    std::vector<PMNode*> geglu_input{residual_add2_output, __xpu__unet_geglu_fcbias0, __xpu__unet_geglu_fcbias1, __xpu__unet_geglu_lnbias, __xpu__unet_geglu_lnscale, __xpu__unet_geglu_fcweight0, __xpu__unet_geglu_fcweight1};
    geglu_input >> *__xpu__unet_geglu >> *__xpu__unet_geglu_output >> *residual_add3 >> *residual_add3_output;
    *residual_add2_output >> *residual_add3;

    *residual_add3_output >> *reshape >> *reshape_output >> *transpose2_2 >> *transpose2_2_output;
    *reshape >> *reshape_output_xshape;
    *transpose2_2 >> *transpose2_2_output_xshape;

    std::vector<PMNode*> post_conv2d_input{transpose2_2_output, post_xpu_conv2d_bias, input, post_xpu_conv2d_filter};
    std::vector<PMNode*> post_conv2d_output{post_xpu_conv2d_output, post_xpu_conv2d_outputmax};
    post_conv2d_input >> *post_xpu_conv2d >> post_conv2d_output;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    // OpDesc
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__unet_spatial_transformer");
    auto* mhsa_op_info = matched.at("__xpu__unet_mhsa")->stmt()->op_info();
    auto* mhca_op_info = matched.at("__xpu__unet_mhca")->stmt()->op_info();
    auto* geglu_op_info = matched.at("__xpu__unet_geglu")->stmt()->op_info();

    std::vector<std::string> fc_weight_names;
    for (const auto& name : mhsa_op_info->Input("FCWeight")) {
      fc_weight_names.push_back(name);
    }
    for (const auto& name : mhca_op_info->Input("FCWeight")) {
      fc_weight_names.push_back(name);
    }
    for (const auto& name : geglu_op_info->Input("FCWeight")) {
      fc_weight_names.push_back(name);
    }
    CHECK_EQ(fc_weight_names.size(), 10);

    std::vector<std::string> fc_weight_maxptr_names;
    for (const auto& name : mhsa_op_info->GetAttr<std::vector<std::string>>("FCWeightMax")) {
      fc_weight_maxptr_names.push_back(name);
    }
    for (const auto& name : mhca_op_info->GetAttr<std::vector<std::string>>("FCWeightMax")) {
      fc_weight_maxptr_names.push_back(name);
    }
    for (const auto& name : geglu_op_info->GetAttr<std::vector<std::string>>("FCWeightMax")) {
      fc_weight_maxptr_names.push_back(name);
    }
    CHECK_EQ(fc_weight_maxptr_names.size(), 10);

    std::vector<std::string> ln_scale_names;
    for (const auto& name : mhsa_op_info->Input("LNScale")) {
      ln_scale_names.push_back(name);
    }
    for (const auto& name : mhca_op_info->Input("LNScale")) {
      ln_scale_names.push_back(name);
    }
    for (const auto& name : geglu_op_info->Input("LNScale")) {
      ln_scale_names.push_back(name);
    }
    std::vector<std::string> ln_bias_names;
    for (const auto& name : mhsa_op_info->Input("LNBias")) {
      ln_bias_names.push_back(name);
    }
    for (const auto& name : mhca_op_info->Input("LNBias")) {
      ln_bias_names.push_back(name);
    }
    for (const auto& name : geglu_op_info->Input("LNBias")) {
      ln_bias_names.push_back(name);
    }
    std::vector<std::string> fc_bias_names;
    for (const auto& name : mhsa_op_info->Input("FCBias")) {
      fc_bias_names.push_back(name);
    }
    for (const auto& name : mhca_op_info->Input("FCBias")) {
      fc_bias_names.push_back(name);
    }
    for (const auto& name : geglu_op_info->Input("FCBias")) {
      fc_bias_names.push_back(name);
    }

    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetInput("Embedding", mhca_op_info->Input("Embedding"));
    op_desc.SetInput("FCWeight", fc_weight_names);
    op_desc.SetInput("FCBias", fc_bias_names);
    op_desc.SetInput("LNScale", ln_scale_names);
    op_desc.SetInput("LNBias", ln_bias_names);
    op_desc.SetInput("ConvWeight", {
      matched.at("pre__xpu__conv2d_filter")->arg()->name,
      matched.at("post__xpu__conv2d_filter")->arg()->name
    });
    op_desc.SetInput("ConvBias", {
      matched.at("pre__xpu__conv2d_bias")->arg()->name,
      matched.at("post__xpu__conv2d_bias")->arg()->name
    });
    op_desc.SetInput("GNScale", {
      matched.at("gn_scale")->arg()->name
    });
    op_desc.SetInput("GNBias", {
      matched.at("gn_bias")->arg()->name
    });
    op_desc.SetOutput("Output", {matched.at("post__xpu__conv2d_output")->arg()->name});
    op_desc.SetAttr<std::vector<std::string>>("FCWeightMax", fc_weight_maxptr_names);

    op_desc.SetAttr<int>("head_num", mhsa_op_info->GetAttr<int>("head_num"));
    op_desc.SetAttr<int>("size_per_head", mhsa_op_info->GetAttr<int>("size_per_head"));
    op_desc.SetAttr<int>("hidden_dim", mhsa_op_info->GetAttr<int>("hidden_dim"));
    op_desc.SetAttr<int>("embedding_dim", mhca_op_info->GetAttr<int>("embedding_dim"));
    op_desc.SetAttr<int>("gelu_dim", geglu_op_info->GetAttr<int>("gelu_dim"));

    auto spatial_transformer_op = LiteOpRegistry::Global().Create(op_desc.Type());
    auto* scope = matched.at("gn")->stmt()->op()->scope();
    spatial_transformer_op->Attach(op_desc, scope);
    spatial_transformer_op->SetValidPlaces(matched.at("gn")->stmt()->op()->valid_places());
    auto kernels = spatial_transformer_op->CreateKernels(spatial_transformer_op->valid_places());
    auto* new_op_node = graph->GraphCreateInstructNode(spatial_transformer_op, spatial_transformer_op->valid_places());

    std::vector<std::string> froms = {
        "input",
        "gn_scale",
        "gn_bias",
        "pre__xpu__conv2d_bias",
        "pre__xpu__conv2d_filter",
        "__xpu__unet_mhsa_fcbias",
        "__xpu__unet_mhsa_lnbias",
        "__xpu__unet_mhsa_lnscale",
        "__xpu__unet_mhsa_fcweight0",
        "__xpu__unet_mhsa_fcweight1",
        "__xpu__unet_mhsa_fcweight2",
        "__xpu__unet_mhsa_fcweight3",
        "__xpu__unet_mhca_embedding",
        "__xpu__unet_mhca_fcbias",
        "__xpu__unet_mhca_lnbias",
        "__xpu__unet_mhca_lnscale",
        "__xpu__unet_mhca_fcweight0",
        "__xpu__unet_mhca_fcweight1",
        "__xpu__unet_mhca_fcweight2",
        "__xpu__unet_mhca_fcweight3",
        "__xpu__unet_geglu_fcbias0",
        "__xpu__unet_geglu_fcbias1",
        "__xpu__unet_geglu_lnbias",
        "__xpu__unet_geglu_lnscale",
        "__xpu__unet_geglu_fcweight0",
        "__xpu__unet_geglu_fcweight1",
        "post__xpu__conv2d_bias",
        "post__xpu__conv2d_filter"
    };

    for (auto& from : froms) {
       IR_NODE_LINK_TO(matched.at(from), new_op_node);
    }

    IR_OP_VAR_LINK(new_op_node, matched.at("post__xpu__conv2d_output"));

  }
};

}  // namespace fusion

class XPUUnetSpatialTransformerfusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    fusion::UnetSpatialTransformerfuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__unet_spatial_transformer_fuse_pass,
                  paddle::lite::mir::XPUUnetSpatialTransformerfusePass)
    .BindTargets({TARGET(kXPU)});
