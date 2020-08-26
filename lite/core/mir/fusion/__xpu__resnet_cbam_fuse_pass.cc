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
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/xpu_pattern_matcher_high_api.h"
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class XPUResNetCbamBlock0Fuser : public FuseBase {
 public:
  XPUResNetCbamBlock0Fuser() {}

  void BuildPattern() override {
    auto* input =
        VarNode("input")->assert_is_op_input("conv2d", "Input")->AsInput();

    auto* left_conv1_weight = VarNode("left_conv1_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv1 = OpNode("left_conv1", "conv2d");
    auto* left_conv1_out = VarNode("left_conv1_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    auto* left_bn1_scale = VarNode("left_bn1_scale")
                               ->assert_is_op_input("batch_norm", "Scale")
                               ->AsIntermediate();
    auto* left_bn1_bias = VarNode("left_bn1_bias")
                              ->assert_is_op_input("batch_norm", "Bias")
                              ->AsInput();
    auto* left_bn1_mean = VarNode("left_bn1_mean")
                              ->assert_is_op_input("batch_norm", "Mean")
                              ->AsIntermediate();
    auto* left_bn1_var = VarNode("left_bn1_variance")
                             ->assert_is_op_input("batch_norm", "Variance")
                             ->AsIntermediate();
    auto* left_bn1 = OpNode("left_bn1", "batch_norm")->AsIntermediate();
    auto* left_bn1_out = VarNode("left_bn1_out")
                             ->assert_is_op_output("batch_norm", "Y")
                             ->assert_is_op_input("relu", "X")
                             ->AsIntermediate();
    auto* left_bn1_mean_out = VarNode("left_bn1_mean_out")
                                  ->assert_is_op_output("batch_norm", "MeanOut")
                                  ->AsIntermediate();
    auto* left_bn1_var_out =
        VarNode("left_bn1_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* left_bn1_saved_mean =
        VarNode("left_bn1_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* left_bn1_saved_var =
        VarNode("left_bn1_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();
    auto* left_relu1 = OpNode("left_relu1", "relu")->AsIntermediate();
    auto* left_relu1_out = VarNode("left_relu1_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_input("conv2d", "Input")
                               ->AsIntermediate();

    auto* left_conv2_weight = VarNode("left_conv2_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv2 = OpNode("left_conv2", "conv2d")->AsIntermediate();
    auto* left_conv2_out = VarNode("left_conv2_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    auto* left_bn2_scale = VarNode("left_bn2_scale")
                               ->assert_is_op_input("batch_norm", "Scale")
                               ->AsIntermediate();
    auto* left_bn2_bias = VarNode("left_bn2_bias")
                              ->assert_is_op_input("batch_norm", "Bias")
                              ->AsInput();
    auto* left_bn2_mean = VarNode("left_bn2_mean")
                              ->assert_is_op_input("batch_norm", "Mean")
                              ->AsIntermediate();
    auto* left_bn2_var = VarNode("left_bn2_variance")
                             ->assert_is_op_input("batch_norm", "Variance")
                             ->AsIntermediate();
    auto* left_bn2 = OpNode("left_bn2", "batch_norm")->AsIntermediate();
    auto* left_bn2_out = VarNode("left_bn2_out")
                             ->assert_is_op_output("batch_norm", "Y")
                             ->assert_is_op_input("relu", "X")
                             ->AsIntermediate();
    auto* left_bn2_mean_out = VarNode("left_bn2_mean_out")
                                  ->assert_is_op_output("batch_norm", "MeanOut")
                                  ->AsIntermediate();
    auto* left_bn2_var_out =
        VarNode("left_bn2_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* left_bn2_saved_mean =
        VarNode("left_bn2_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* left_bn2_saved_var =
        VarNode("left_bn2_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();
    auto* left_relu2 = OpNode("left_relu2", "relu")->AsIntermediate();
    auto* left_relu2_out = VarNode("left_relu2_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_input("conv2d", "Input")
                               ->AsIntermediate();

    auto* left_conv3_weight = VarNode("left_conv3_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv3 = OpNode("left_conv3", "conv2d")->AsIntermediate();
    auto* left_conv3_out = VarNode("left_conv3_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    auto* left_bn3_scale = VarNode("left_bn3_scale")
                               ->assert_is_op_input("batch_norm", "Scale")
                               ->AsIntermediate();
    auto* left_bn3_bias = VarNode("left_bn3_bias")
                              ->assert_is_op_input("batch_norm", "Bias")
                              ->AsInput();
    auto* left_bn3_mean = VarNode("left_bn3_mean")
                              ->assert_is_op_input("batch_norm", "Mean")
                              ->AsIntermediate();
    auto* left_bn3_var = VarNode("left_bn3_variance")
                             ->assert_is_op_input("batch_norm", "Variance")
                             ->AsIntermediate();
    auto* left_bn3 = OpNode("left_bn3", "batch_norm")->AsIntermediate();
    auto* left_bn3_out = VarNode("left_bn3_out")
                             ->assert_is_op_output("batch_norm", "Y")
                             ->AsIntermediate();
    auto* left_bn3_mean_out = VarNode("left_bn3_mean_out")
                                  ->assert_is_op_output("batch_norm", "MeanOut")
                                  ->AsIntermediate();
    auto* left_bn3_var_out =
        VarNode("left_bn3_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* left_bn3_saved_mean =
        VarNode("left_bn3_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* left_bn3_saved_var =
        VarNode("left_bn3_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();

    // cbam specific
    auto* reduce_mean = OpNode("reduce_mean", "reduce_mean")->AsIntermediate();
    auto* reduce_mean_out = VarNode("reduce_mean_out")
                                ->assert_is_op_output("reduce_mean", "Out")
                                ->assert_is_op_input("concat")
                                ->AsIntermediate();
    auto* reduce_max = OpNode("reduce_max", "reduce_max")->AsIntermediate();
    auto* reduce_max_out = VarNode("reduce_max_out")
                               ->assert_is_op_output("reduce_max", "Out")
                               ->assert_is_op_input("concat")
                               ->AsIntermediate();
    auto* concat = OpNode("concat", "concat")->AsIntermediate();
    auto* concat_out = VarNode("concat_out")
                           ->assert_is_op_output("concat", "Out")
                           ->assert_is_op_input("conv2d", "Input")
                           ->AsIntermediate();
    auto* left_conv4_weight = VarNode("left_conv4_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv4 = OpNode("left_conv4", "conv2d")->AsIntermediate();
    auto* left_conv4_out = VarNode("left_conv4_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("sigmoid", "X")
                               ->AsIntermediate();
    auto* sigmoid = OpNode("sigmoid", "sigmoid")->AsIntermediate();
    auto* sigmoid_out = VarNode("sigmoid_out")
                            ->assert_is_op_output("sigmoid", "Out")
                            ->assert_is_op_input("elementwise_mul")
                            ->AsIntermediate();
    auto* reshape = OpNode("reshape", "reshape2")->AsIntermediate();
    auto* reshape_out = VarNode("reshape_out")
                            ->assert_is_op_output("reshape2", "Out")
                            ->assert_is_op_input("elementwise_mul")
                            ->AsIntermediate();
    auto* reshape_xshape = VarNode("reshape_xshape")
                               ->assert_is_op_output("reshape2", "XShape")
                               ->AsIntermediate();
    auto* eltwise_mul =
        OpNode("eltwise_mul", "elementwise_mul")->AsIntermediate();
    auto* eltwise_mul_out = VarNode("eltwise_mul_out")
                                ->assert_is_op_output("elementwise_mul", "Out")
                                ->assert_is_op_input("elementwise_add")
                                ->AsIntermediate();

    auto* right_conv1_weight = VarNode("right_conv1_weight")
                                   ->assert_is_op_input("conv2d", "Filter")
                                   ->AsInput();
    auto* right_conv1 = OpNode("right_conv1", "conv2d")->AsIntermediate();
    auto* right_conv1_out = VarNode("right_conv1_out")
                                ->assert_is_op_output("conv2d", "Output")
                                ->assert_is_op_input("batch_norm", "X")
                                ->AsIntermediate();
    auto* right_bn1_scale = VarNode("right_bn1_scale")
                                ->assert_is_op_input("batch_norm", "Scale")
                                ->AsIntermediate();
    auto* right_bn1_bias = VarNode("right_bn1_bias")
                               ->assert_is_op_input("batch_norm", "Bias")
                               ->AsInput();
    auto* right_bn1_mean = VarNode("right_bn1_mean")
                               ->assert_is_op_input("batch_norm", "Mean")
                               ->AsIntermediate();
    auto* right_bn1_var = VarNode("right_bn1_variance")
                              ->assert_is_op_input("batch_norm", "Variance")
                              ->AsIntermediate();
    auto* right_bn1 = OpNode("right_bn1", "batch_norm")->AsIntermediate();
    auto* right_bn1_out = VarNode("right_bn1_out")
                              ->assert_is_op_output("batch_norm", "Y")
                              ->assert_is_op_input("elementwise_add")
                              ->AsIntermediate();
    auto* right_bn1_mean_out =
        VarNode("right_bn1_mean_out")
            ->assert_is_op_output("batch_norm", "MeanOut")
            ->AsIntermediate();
    auto* right_bn1_var_out =
        VarNode("right_bn1_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* right_bn1_saved_mean =
        VarNode("right_bn1_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* right_bn1_saved_var =
        VarNode("right_bn1_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();

    auto* add = OpNode("add", "elementwise_add")->AsIntermediate();
    auto* add_out = VarNode("add_out")
                        ->assert_is_op_output("elementwise_add", "Out")
                        ->assert_is_op_input("relu", "X")
                        ->AsIntermediate();
    auto* relu = OpNode("relu", "relu")->AsIntermediate();
    auto* relu_out =
        VarNode("relu_out")->assert_is_op_output("relu", "Out")->AsOutput();

    *input >> *left_conv1 >> *left_conv1_out >> *left_bn1 >> *left_bn1_out >>
        *left_relu1 >> *left_relu1_out >> *left_conv2 >> *left_conv2_out >>
        *left_bn2 >> *left_bn2_out >> *left_relu2 >> *left_relu2_out >>
        *left_conv3 >> *left_conv3_out >> *left_bn3 >>
        *left_bn3_out /* >> *add*/;

    *left_bn3_out >> *reduce_mean >> *reduce_mean_out >> *concat;
    *left_bn3_out >> *reduce_max >> *reduce_max_out >> *concat;
    *concat >> *concat_out >> *left_conv4 >> *left_conv4_out >> *sigmoid >>
        *sigmoid_out >> *eltwise_mul;
    *left_conv4_weight >> *left_conv4;
    *left_bn3_out >> *reshape >> *reshape_out >> *eltwise_mul;
    *reshape >> *reshape_xshape;
    *eltwise_mul >> *eltwise_mul_out >> *add;

    *left_conv1_weight >> *left_conv1;
    *left_bn1_scale >> *left_bn1;
    *left_bn1_bias >> *left_bn1;
    *left_bn1_mean >> *left_bn1;
    *left_bn1_var >> *left_bn1;
    *left_bn1 >> *left_bn1_mean_out;
    *left_bn1 >> *left_bn1_var_out;
    *left_bn1 >> *left_bn1_saved_mean;
    *left_bn1 >> *left_bn1_saved_var;

    *left_conv2_weight >> *left_conv2;
    *left_bn2_scale >> *left_bn2;
    *left_bn2_bias >> *left_bn2;
    *left_bn2_mean >> *left_bn2;
    *left_bn2_var >> *left_bn2;
    *left_bn2 >> *left_bn2_mean_out;
    *left_bn2 >> *left_bn2_var_out;
    *left_bn2 >> *left_bn2_saved_mean;
    *left_bn2 >> *left_bn2_saved_var;

    *left_conv3_weight >> *left_conv3;
    *left_bn3_scale >> *left_bn3;
    *left_bn3_bias >> *left_bn3;
    *left_bn3_mean >> *left_bn3;
    *left_bn3_var >> *left_bn3;
    *left_bn3 >> *left_bn3_mean_out;
    *left_bn3 >> *left_bn3_var_out;
    *left_bn3 >> *left_bn3_saved_mean;
    *left_bn3 >> *left_bn3_saved_var;

    *input >> *right_conv1 >> *right_conv1_out >> *right_bn1 >>
        *right_bn1_out >> *add;

    *right_conv1_weight >> *right_conv1;
    *right_bn1_scale >> *right_bn1;
    *right_bn1_bias >> *right_bn1;
    *right_bn1_mean >> *right_bn1;
    *right_bn1_var >> *right_bn1;
    *right_bn1 >> *right_bn1_mean_out;
    *right_bn1 >> *right_bn1_var_out;
    *right_bn1 >> *right_bn1_saved_mean;
    *right_bn1 >> *right_bn1_saved_var;

    *add >> *add_out >> *relu >> *relu_out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("resnet_cbam_block0");
    op_desc.SetInput("Inputs", {matched.at("input")->arg()->name});
    op_desc.SetInput("Filter",
                     {
                         matched.at("left_conv1_weight")->arg()->name,
                         matched.at("left_conv2_weight")->arg()->name,
                         matched.at("left_conv3_weight")->arg()->name,
                         matched.at("left_conv4_weight")->arg()->name,
                         matched.at("right_conv1_weight")->arg()->name,
                     });
    op_desc.SetInput("Scale",
                     {
                         matched.at("left_bn1_scale")->arg()->name,
                         matched.at("left_bn2_scale")->arg()->name,
                         matched.at("left_bn3_scale")->arg()->name,
                         "placeholder_sa_conv",
                         matched.at("right_bn1_scale")->arg()->name,
                     });
    op_desc.SetInput("Bias",
                     {
                         matched.at("left_bn1_bias")->arg()->name,
                         matched.at("left_bn2_bias")->arg()->name,
                         matched.at("left_bn3_bias")->arg()->name,
                         "placeholder_sa_conv",
                         matched.at("right_bn1_bias")->arg()->name,
                     });
    op_desc.SetInput("Mean",
                     {
                         matched.at("left_bn1_mean")->arg()->name,
                         matched.at("left_bn2_mean")->arg()->name,
                         matched.at("left_bn3_mean")->arg()->name,
                         "placeholder_sa_conv",
                         matched.at("right_bn1_mean")->arg()->name,
                     });
    op_desc.SetInput("Var",
                     {
                         matched.at("left_bn1_variance")->arg()->name,
                         matched.at("left_bn2_variance")->arg()->name,
                         matched.at("left_bn3_variance")->arg()->name,
                         "placeholder_sa_conv",
                         matched.at("right_bn1_variance")->arg()->name,
                     });
    op_desc.SetOutput("Outputs", {matched.at("relu_out")->arg()->name});
    // XXX: keep these to fool SubgraphOp::AttachImpl()
    op_desc.SetAttr<int>("sub_block", 0);
    op_desc.SetAttr<std::vector<std::string>>("input_data_names", {});
    op_desc.SetAttr<std::vector<std::string>>("output_data_names", {});

    auto block0_stmt = matched.at("left_conv1")->stmt();
    // block0_stmt->ResetOp(op_desc, graph->valid_places());
    auto fake_subgraph_op = LiteOpRegistry::Global().Create("subgraph");
    auto sub_program_desc = std::make_shared<cpp::ProgramDesc>();
    sub_program_desc->AddBlock<cpp::BlockDesc>();
    static_cast<operators::SubgraphOp*>(fake_subgraph_op.get())
        ->SetProgramDesc(sub_program_desc);
    fake_subgraph_op->Attach(op_desc, block0_stmt->op()->scope());
    fake_subgraph_op->SetValidPlaces(block0_stmt->op()->valid_places());
    block0_stmt->SetOp(fake_subgraph_op);

    std::vector<std::string> froms = {
        "left_conv2_weight",
        "left_conv3_weight",
        "left_conv4_weight",
        "right_conv1_weight",
        "left_bn1_bias",
        "left_bn2_bias",
        "left_bn3_bias",
        "right_bn1_bias",
    };
    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), matched.at("left_conv1"));
    }
    IR_OP_VAR_LINK(matched.at("left_conv1"), matched.at("relu_out"));
  }
};

class XPUResNetCbamBlock1Fuser : public FuseBase {
 public:
  XPUResNetCbamBlock1Fuser() {}

  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("conv2d", "Input")
                      ->assert_is_op_input("elementwise_add")
                      ->AsInput();

    auto* right_conv1_weight = VarNode("right_conv1_weight")
                                   ->assert_is_op_input("conv2d", "Filter")
                                   ->AsInput();
    auto* right_conv1 = OpNode("right_conv1", "conv2d");
    auto* right_conv1_out = VarNode("right_conv1_out")
                                ->assert_is_op_output("conv2d", "Output")
                                ->assert_is_op_input("batch_norm", "X")
                                ->AsIntermediate();
    auto* right_bn1_scale = VarNode("right_bn1_scale")
                                ->assert_is_op_input("batch_norm", "Scale")
                                ->AsIntermediate();
    auto* right_bn1_bias = VarNode("right_bn1_bias")
                               ->assert_is_op_input("batch_norm", "Bias")
                               ->AsInput();
    auto* right_bn1_mean = VarNode("right_bn1_mean")
                               ->assert_is_op_input("batch_norm", "Mean")
                               ->AsIntermediate();
    auto* right_bn1_var = VarNode("right_bn1_variance")
                              ->assert_is_op_input("batch_norm", "Variance")
                              ->AsIntermediate();
    auto* right_bn1 = OpNode("right_bn1", "batch_norm")->AsIntermediate();
    auto* right_bn1_out = VarNode("right_bn1_out")
                              ->assert_is_op_output("batch_norm", "Y")
                              ->assert_is_op_input("relu", "X")
                              ->AsIntermediate();
    auto* right_bn1_mean_out =
        VarNode("right_bn1_mean_out")
            ->assert_is_op_output("batch_norm", "MeanOut")
            ->AsIntermediate();
    auto* right_bn1_var_out =
        VarNode("right_bn1_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* right_bn1_saved_mean =
        VarNode("right_bn1_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* right_bn1_saved_var =
        VarNode("right_bn1_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();
    auto* right_relu1 = OpNode("right_relu1", "relu")->AsIntermediate();
    auto* right_relu1_out = VarNode("right_relu1_out")
                                ->assert_is_op_output("relu", "Out")
                                ->assert_is_op_input("conv2d", "Input")
                                ->AsIntermediate();

    auto* right_conv2_weight = VarNode("right_conv2_weight")
                                   ->assert_is_op_input("conv2d", "Filter")
                                   ->AsInput();
    auto* right_conv2 = OpNode("right_conv2", "conv2d")->AsIntermediate();
    auto* right_conv2_out = VarNode("right_conv2_out")
                                ->assert_is_op_output("conv2d", "Output")
                                ->assert_is_op_input("batch_norm", "X")
                                ->AsIntermediate();
    auto* right_bn2_scale = VarNode("right_bn2_scale")
                                ->assert_is_op_input("batch_norm", "Scale")
                                ->AsIntermediate();
    auto* right_bn2_bias = VarNode("right_bn2_bias")
                               ->assert_is_op_input("batch_norm", "Bias")
                               ->AsInput();
    auto* right_bn2_mean = VarNode("right_bn2_mean")
                               ->assert_is_op_input("batch_norm", "Mean")
                               ->AsIntermediate();
    auto* right_bn2_var = VarNode("right_bn2_variance")
                              ->assert_is_op_input("batch_norm", "Variance")
                              ->AsIntermediate();
    auto* right_bn2 = OpNode("right_bn2", "batch_norm")->AsIntermediate();
    auto* right_bn2_out = VarNode("right_bn2_out")
                              ->assert_is_op_output("batch_norm", "Y")
                              ->assert_is_op_input("relu", "X")
                              ->AsIntermediate();
    auto* right_bn2_mean_out =
        VarNode("right_bn2_mean_out")
            ->assert_is_op_output("batch_norm", "MeanOut")
            ->AsIntermediate();
    auto* right_bn2_var_out =
        VarNode("right_bn2_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* right_bn2_saved_mean =
        VarNode("right_bn2_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* right_bn2_saved_var =
        VarNode("right_bn2_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();
    auto* right_relu2 = OpNode("right_relu2", "relu")->AsIntermediate();
    auto* right_relu2_out = VarNode("right_relu2_out")
                                ->assert_is_op_output("relu", "Out")
                                ->assert_is_op_input("conv2d", "Input")
                                ->AsIntermediate();

    auto* right_conv3_weight = VarNode("right_conv3_weight")
                                   ->assert_is_op_input("conv2d", "Filter")
                                   ->AsInput();
    auto* right_conv3 = OpNode("right_conv3", "conv2d")->AsIntermediate();
    auto* right_conv3_out = VarNode("right_conv3_out")
                                ->assert_is_op_output("conv2d", "Output")
                                ->assert_is_op_input("batch_norm", "X")
                                ->AsIntermediate();
    auto* right_bn3_scale = VarNode("right_bn3_scale")
                                ->assert_is_op_input("batch_norm", "Scale")
                                ->AsIntermediate();
    auto* right_bn3_bias = VarNode("right_bn3_bias")
                               ->assert_is_op_input("batch_norm", "Bias")
                               ->AsInput();
    auto* right_bn3_mean = VarNode("right_bn3_mean")
                               ->assert_is_op_input("batch_norm", "Mean")
                               ->AsIntermediate();
    auto* right_bn3_var = VarNode("right_bn3_variance")
                              ->assert_is_op_input("batch_norm", "Variance")
                              ->AsIntermediate();
    auto* right_bn3 = OpNode("right_bn3", "batch_norm")->AsIntermediate();
    auto* right_bn3_out = VarNode("right_bn3_out")
                              ->assert_is_op_output("batch_norm", "Y")
                              ->AsIntermediate();
    auto* right_bn3_mean_out =
        VarNode("right_bn3_mean_out")
            ->assert_is_op_output("batch_norm", "MeanOut")
            ->AsIntermediate();
    auto* right_bn3_var_out =
        VarNode("right_bn3_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* right_bn3_saved_mean =
        VarNode("right_bn3_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* right_bn3_saved_var =
        VarNode("right_bn3_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();

    // cbam specific
    auto* reduce_mean = OpNode("reduce_mean", "reduce_mean")->AsIntermediate();
    auto* reduce_mean_out = VarNode("reduce_mean_out")
                                ->assert_is_op_output("reduce_mean", "Out")
                                ->assert_is_op_input("concat")
                                ->AsIntermediate();
    auto* reduce_max = OpNode("reduce_max", "reduce_max")->AsIntermediate();
    auto* reduce_max_out = VarNode("reduce_max_out")
                               ->assert_is_op_output("reduce_max", "Out")
                               ->assert_is_op_input("concat")
                               ->AsIntermediate();
    auto* concat = OpNode("concat", "concat")->AsIntermediate();
    auto* concat_out = VarNode("concat_out")
                           ->assert_is_op_output("concat", "Out")
                           ->assert_is_op_input("conv2d", "Input")
                           ->AsIntermediate();
    auto* right_conv4_weight = VarNode("right_conv4_weight")
                                   ->assert_is_op_input("conv2d", "Filter")
                                   ->AsInput();
    auto* right_conv4 = OpNode("right_conv4", "conv2d")->AsIntermediate();
    auto* right_conv4_out = VarNode("right_conv4_out")
                                ->assert_is_op_output("conv2d", "Output")
                                ->assert_is_op_input("sigmoid", "X")
                                ->AsIntermediate();
    auto* sigmoid = OpNode("sigmoid", "sigmoid")->AsIntermediate();
    auto* sigmoid_out = VarNode("sigmoid_out")
                            ->assert_is_op_output("sigmoid", "Out")
                            ->assert_is_op_input("elementwise_mul")
                            ->AsIntermediate();
    auto* reshape = OpNode("reshape", "reshape2")->AsIntermediate();
    auto* reshape_out = VarNode("reshape_out")
                            ->assert_is_op_output("reshape2", "Out")
                            ->assert_is_op_input("elementwise_mul")
                            ->AsIntermediate();
    auto* reshape_xshape = VarNode("reshape_xshape")
                               ->assert_is_op_output("reshape2", "XShape")
                               ->AsIntermediate();
    auto* eltwise_mul =
        OpNode("eltwise_mul", "elementwise_mul")->AsIntermediate();
    auto* eltwise_mul_out = VarNode("eltwise_mul_out")
                                ->assert_is_op_output("elementwise_mul", "Out")
                                ->assert_is_op_input("elementwise_add")
                                ->AsIntermediate();

    auto* add = OpNode("add", "elementwise_add")->AsIntermediate();
    auto* add_out = VarNode("add_out")
                        ->assert_is_op_output("elementwise_add", "Out")
                        ->assert_is_op_input("relu", "X")
                        ->AsIntermediate();
    auto* relu = OpNode("relu", "relu")->AsIntermediate();
    auto* relu_out =
        VarNode("relu_out")->assert_is_op_output("relu", "Out")->AsOutput();

    *input >> *right_conv1 >> *right_conv1_out >> *right_bn1 >>
        *right_bn1_out >> *right_relu1 >> *right_relu1_out >> *right_conv2 >>
        *right_conv2_out >> *right_bn2 >> *right_bn2_out >> *right_relu2 >>
        *right_relu2_out >> *right_conv3 >> *right_conv3_out >> *right_bn3 >>
        *right_bn3_out /* >> *add*/;

    *right_bn3_out >> *reduce_mean >> *reduce_mean_out >> *concat;
    *right_bn3_out >> *reduce_max >> *reduce_max_out >> *concat;
    *concat >> *concat_out >> *right_conv4 >> *right_conv4_out >> *sigmoid >>
        *sigmoid_out >> *eltwise_mul;
    *right_conv4_weight >> *right_conv4;
    *right_bn3_out >> *reshape >> *reshape_out >> *eltwise_mul;
    *reshape >> *reshape_xshape;
    *eltwise_mul >> *eltwise_mul_out >> *add;

    *right_conv1_weight >> *right_conv1;
    *right_bn1_scale >> *right_bn1;
    *right_bn1_bias >> *right_bn1;
    *right_bn1_mean >> *right_bn1;
    *right_bn1_var >> *right_bn1;
    *right_bn1 >> *right_bn1_mean_out;
    *right_bn1 >> *right_bn1_var_out;
    *right_bn1 >> *right_bn1_saved_mean;
    *right_bn1 >> *right_bn1_saved_var;

    *right_conv2_weight >> *right_conv2;
    *right_bn2_scale >> *right_bn2;
    *right_bn2_bias >> *right_bn2;
    *right_bn2_mean >> *right_bn2;
    *right_bn2_var >> *right_bn2;
    *right_bn2 >> *right_bn2_mean_out;
    *right_bn2 >> *right_bn2_var_out;
    *right_bn2 >> *right_bn2_saved_mean;
    *right_bn2 >> *right_bn2_saved_var;

    *right_conv3_weight >> *right_conv3;
    *right_bn3_scale >> *right_bn3;
    *right_bn3_bias >> *right_bn3;
    *right_bn3_mean >> *right_bn3;
    *right_bn3_var >> *right_bn3;
    *right_bn3 >> *right_bn3_mean_out;
    *right_bn3 >> *right_bn3_var_out;
    *right_bn3 >> *right_bn3_saved_mean;
    *right_bn3 >> *right_bn3_saved_var;

    *input >> *add;

    *add >> *add_out >> *relu >> *relu_out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("resnet_cbam_block1");
    op_desc.SetInput("Inputs", {matched.at("input")->arg()->name});
    op_desc.SetInput("Filter",
                     {
                         matched.at("right_conv1_weight")->arg()->name,
                         matched.at("right_conv2_weight")->arg()->name,
                         matched.at("right_conv3_weight")->arg()->name,
                         matched.at("right_conv4_weight")->arg()->name,
                     });
    op_desc.SetInput("Scale",
                     {
                         matched.at("right_bn1_scale")->arg()->name,
                         matched.at("right_bn2_scale")->arg()->name,
                         matched.at("right_bn3_scale")->arg()->name,
                         "placeholder_sa_conv",
                     });
    op_desc.SetInput("Bias",
                     {
                         matched.at("right_bn1_bias")->arg()->name,
                         matched.at("right_bn2_bias")->arg()->name,
                         matched.at("right_bn3_bias")->arg()->name,
                         "placeholder_sa_conv",
                     });
    op_desc.SetInput("Mean",
                     {
                         matched.at("right_bn1_mean")->arg()->name,
                         matched.at("right_bn2_mean")->arg()->name,
                         matched.at("right_bn3_mean")->arg()->name,
                         "placeholder_sa_conv",
                     });
    op_desc.SetInput("Var",
                     {
                         matched.at("right_bn1_variance")->arg()->name,
                         matched.at("right_bn2_variance")->arg()->name,
                         matched.at("right_bn3_variance")->arg()->name,
                         "placeholder_sa_conv",
                     });
    op_desc.SetOutput("Outputs", {matched.at("relu_out")->arg()->name});
    // XXX: keep these to fool SubgraphOp::AttachImpl()
    op_desc.SetAttr<int>("sub_block", 0);
    op_desc.SetAttr<std::vector<std::string>>("input_data_names", {});
    op_desc.SetAttr<std::vector<std::string>>("output_data_names", {});

    auto block1_stmt = matched.at("right_conv1")->stmt();
    auto fake_subgraph_op = LiteOpRegistry::Global().Create("subgraph");
    auto sub_program_desc = std::make_shared<cpp::ProgramDesc>();
    sub_program_desc->AddBlock<cpp::BlockDesc>();
    static_cast<operators::SubgraphOp*>(fake_subgraph_op.get())
        ->SetProgramDesc(sub_program_desc);
    fake_subgraph_op->Attach(op_desc, block1_stmt->op()->scope());
    fake_subgraph_op->SetValidPlaces(block1_stmt->op()->valid_places());
    block1_stmt->SetOp(fake_subgraph_op);

    std::vector<std::string> froms = {
        "right_conv2_weight",
        "right_conv3_weight",
        "right_conv4_weight",
        "right_bn1_bias",
        "right_bn2_bias",
        "right_bn3_bias",
    };
    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), matched.at("right_conv1"));
    }
    IR_OP_VAR_LINK(matched.at("right_conv1"), matched.at("relu_out"));
  }
};

class XPUResNetCbamBlock2Fuser : public FuseBase {
 public:
  XPUResNetCbamBlock2Fuser() {}

  void BuildPattern() override {
    auto* input = VarNode("input")->assert_is_op_input("clip", "X")->AsInput();

    auto* clip = OpNode("clip", "clip");
    auto* clip_out = VarNode("clip_out")
                         ->assert_is_op_output("clip", "Out")
                         ->assert_is_op_input("elementwise_pow")
                         ->AsIntermediate();
    auto* eltwise_y = VarNode("eltwise_y")
                          ->assert_is_op_input("elementwise_pow")
                          ->assert_is_op_input("elementwise_div")
                          ->AsIntermediate();
    auto* eltwise_pow =
        OpNode("eltwise_pow", "elementwise_pow")->AsIntermediate();
    auto* eltwise_pow_out = VarNode("eltwise_pow_out")
                                ->assert_is_op_output("elementwise_pow", "Out")
                                ->assert_is_op_input("pad2d", "X")
                                ->AsIntermediate();
    auto* pad2d = OpNode("pad2d", "pad2d")->AsIntermediate();
    auto* pad2d_out = VarNode("pad2d_out")
                          ->assert_is_op_output("pad2d", "Out")
                          ->assert_is_op_input("pool2d", "X")
                          ->AsIntermediate();
    auto* pool2d = OpNode("pool2d", "pool2d")->AsIntermediate();
    auto* pool2d_out = VarNode("pool2d_out")
                           ->assert_is_op_output("pool2d", "Out")
                           ->assert_is_op_input("elementwise_pow")
                           ->AsIntermediate();

    auto* fill_const = OpNode("fill_const", "fill_constant")->AsIntermediate();
    auto* fill_const_out = VarNode("fill_const_out")
                               ->assert_is_op_output("fill_constant", "Out")
                               ->assert_is_op_input("elementwise_div")
                               ->AsIntermediate();
    auto* eltwise_div =
        OpNode("eltwise_div", "elementwise_div")->AsIntermediate();
    auto* eltwise_div_out = VarNode("eltwise_div_out")
                                ->assert_is_op_output("elementwise_div", "Out")
                                ->assert_is_op_input("elementwise_pow")
                                ->AsIntermediate();

    auto* eltwise_pow2 =
        OpNode("eltwise_pow2", "elementwise_pow")->AsIntermediate();
    auto* eltwise_pow2_out = VarNode("eltwise_pow2_out")
                                 ->assert_is_op_output("elementwise_pow", "Out")
                                 ->AsIntermediate();

    auto* shape = OpNode("shape", "shape")->AsIntermediate();
    auto* shape_out = VarNode("shape_out")
                          ->assert_is_op_output("shape", "Out")
                          ->assert_is_op_input("gather")
                          ->AsIntermediate();
    auto* fill_const2 =
        OpNode("fill_const2", "fill_constant")->AsIntermediate();
    auto* fill_const2_out = VarNode("fill_const2_out")
                                ->assert_is_op_output("fill_constant", "Out")
                                ->assert_is_op_input("gather")
                                ->AsIntermediate();
    auto* gather = OpNode("gather", "gather")->AsIntermediate();
    auto* gather_out = VarNode("gather_out")
                           ->assert_is_op_output("gather", "Out")
                           ->assert_is_op_input("assign", "X")
                           ->AsIntermediate();
    auto* assign = OpNode("assign", "assign")->AsIntermediate();
    auto* assign_out = VarNode("assign_out")
                           ->assert_is_op_output("assign", "Out")
                           ->assert_is_op_input("concat")
                           ->AsIntermediate();

    auto* fill_const3 =
        OpNode("fill_const3", "fill_constant")->AsIntermediate();
    auto* fill_const3_out = VarNode("fill_const3_out")
                                ->assert_is_op_output("fill_constant", "Out")
                                ->assert_is_op_input("assign")
                                ->AsIntermediate();
    auto* assign2 = OpNode("assign2", "assign")->AsIntermediate();
    auto* assign2_out = VarNode("assign2_out")
                            ->assert_is_op_output("assign", "Out")
                            ->assert_is_op_input("concat")
                            ->AsIntermediate();

    auto* concat = OpNode("concat", "concat")->AsIntermediate();
    auto* concat_out = VarNode("concat_out")
                           ->assert_is_op_output("concat", "Out")
                           ->assert_is_op_input("cast", "X")
                           ->AsIntermediate();
    auto* cast = OpNode("cast", "cast")->AsIntermediate();
    auto* cast_out = VarNode("cast_out")
                         ->assert_is_op_output("cast", "Out")
                         ->assert_is_op_input("reshape2", "Shape")
                         ->AsIntermediate();

    auto* reshape2 = OpNode("reshape2", "reshape2")->AsIntermediate();
    auto* reshape2_out = VarNode("reshape2_out")
                             ->assert_is_op_output("reshape2", "Out")
                             ->assert_is_op_input("matmul", "X")
                             ->AsIntermediate();
    auto* reshape2_xshape = VarNode("reshape2_xshape")
                                ->assert_is_op_output("reshape2", "XShape")
                                ->AsIntermediate();
    auto* matmul_y =
        VarNode("matmul_y")->assert_is_op_input("matmul", "Y")->AsInput();
    auto* matmul = OpNode("matmul", "matmul")->AsIntermediate();
    auto* matmul_out = VarNode("matmul_out")
                           ->assert_is_op_output("matmul", "Out")
                           ->assert_is_op_input("elementwise_add")
                           ->AsIntermediate();
    auto* eltwise_add_y = VarNode("eltwise_add_y")
                              ->assert_is_op_input("elementwise_add")
                              ->AsInput();
    auto* eltwise_add =
        OpNode("eltwise_add", "elementwise_add")->AsIntermediate();
    auto* eltwise_add_out = VarNode("eltwise_add_out")
                                ->assert_is_op_output("elementwise_add", "Out")
                                ->AsIntermediate();

    auto* norm = OpNode("norm", "norm")->AsIntermediate();
    auto* norm_out = VarNode("norm_out")
                         ->assert_is_op_output("norm", "Out")
                         ->assert_is_op_input("elementwise_add")
                         ->AsIntermediate();
    auto* norm_norm = VarNode("norm_norm")
                          ->assert_is_op_output("norm", "Norm")
                          ->AsIntermediate();
    auto* fill_const4 =
        OpNode("fill_const4", "fill_constant")->AsIntermediate();
    auto* fill_const4_out = VarNode("fill_const4_out")
                                ->assert_is_op_output("fill_constant", "Out")
                                ->assert_is_op_input("elementwise_add")
                                ->AsIntermediate();
    auto* eltwise_add2 =
        OpNode("eltwise_add2", "elementwise_add")->AsIntermediate();
    auto* eltwise_add2_out = VarNode("eltwise_add2_out")
                                 ->assert_is_op_output("elementwise_add", "Out")
                                 ->assert_is_op_input("elementwise_mul")
                                 ->AsIntermediate();
    auto* fill_const5 =
        OpNode("fill_const5", "fill_constant")->AsIntermediate();
    auto* fill_const5_out = VarNode("fill_const5_out")
                                ->assert_is_op_output("fill_constant", "Out")
                                ->assert_is_op_input("elementwise_mul")
                                ->AsIntermediate();
    auto* eltwise_mul =
        OpNode("eltwise_mul", "elementwise_mul")->AsIntermediate();
    auto* eltwise_mul_out = VarNode("eltwise_mul_out")
                                ->assert_is_op_output("elementwise_mul", "Out")
                                ->assert_is_op_input("elementwise_div")
                                ->AsIntermediate();

    auto* eltwise_div2 =
        OpNode("eltwise_div2", "elementwise_div")->AsIntermediate();
    auto* eltwise_div2_out = VarNode("eltwise_div2_out")
                                 ->assert_is_op_output("elementwise_div", "Out")
                                 ->AsOutput();

    *input >> *clip >> *clip_out >> *eltwise_pow >> *eltwise_pow_out >>
        *pad2d >> *pad2d_out >> *pool2d >> *pool2d_out >> *eltwise_pow2;
    *eltwise_y >> *eltwise_pow;

    *fill_const >> *fill_const_out >> *eltwise_div >> *eltwise_div_out >>
        *eltwise_pow2;
    *eltwise_y >> *eltwise_div;

    *eltwise_pow2 >> *eltwise_pow2_out >> *shape >> *shape_out >> *gather >>
        *gather_out >> *assign >> *assign_out >> *concat >> *concat_out >>
        *cast >> *cast_out >> *reshape2;
    *fill_const2 >> *fill_const2_out >> *gather;
    *fill_const3 >> *fill_const3_out >> *assign2 >> *assign2_out >> *concat;
    *eltwise_pow2_out >> *reshape2;

    *reshape2 >> *reshape2_out >> *matmul >> *matmul_out >> *eltwise_add >>
        *eltwise_add_out;
    *reshape2 >> *reshape2_xshape;
    *matmul_y >> *matmul;
    *eltwise_add_y >> *eltwise_add;

    *eltwise_add_out >> *norm >> *norm_out >> *eltwise_add2 >>
        *eltwise_add2_out >> *eltwise_mul >> *eltwise_mul_out >>
        *eltwise_div2 >> *eltwise_div2_out;
    *norm >> *norm_norm;
    *fill_const4 >> *fill_const4_out >> *eltwise_add2;
    *fill_const5 >> *fill_const5_out >> *eltwise_mul;
    *eltwise_add_out >> *eltwise_div2;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("resnet_cbam_block2");
    op_desc.SetInput("Inputs", {matched.at("input")->arg()->name});
    op_desc.SetInput("Filter", {matched.at("matmul_y")->arg()->name});
    op_desc.SetInput("Scale", {"placeholder_last_fc"});
    op_desc.SetInput("Bias", {matched.at("eltwise_add_y")->arg()->name});
    op_desc.SetInput("Mean", {"placeholder_last_fc"});
    op_desc.SetInput("Var", {"placeholder_last_fc"});
    op_desc.SetOutput("Outputs", {matched.at("eltwise_div2_out")->arg()->name});
    // XXX: keep these to fool SubgraphOp::AttachImpl()
    op_desc.SetAttr<int>("sub_block", 0);
    op_desc.SetAttr<std::vector<std::string>>("input_data_names", {});
    op_desc.SetAttr<std::vector<std::string>>("output_data_names", {});

    // extra traits to distill
    auto block2_stmt = matched.at("clip")->stmt();
    auto* scope = block2_stmt->op()->scope();
    auto pow_tensor_name = matched.at("eltwise_y")->arg()->name;
    auto* pow_tensor = scope->FindTensor(pow_tensor_name);
    float pool_p = pow_tensor->data<float>()[0];
    op_desc.SetAttr<float>("pool_p", pool_p);
    auto* matmul_op_info = matched.at("matmul")->stmt()->op_info();
    CHECK(matmul_op_info->GetAttr<bool>("transpose_Y") == true)
        << "Y of last fc must have been transposed";

    auto fake_subgraph_op = LiteOpRegistry::Global().Create("subgraph");
    auto sub_program_desc = std::make_shared<cpp::ProgramDesc>();
    sub_program_desc->AddBlock<cpp::BlockDesc>();
    static_cast<operators::SubgraphOp*>(fake_subgraph_op.get())
        ->SetProgramDesc(sub_program_desc);
    fake_subgraph_op->Attach(op_desc, scope);
    fake_subgraph_op->SetValidPlaces(block2_stmt->op()->valid_places());
    block2_stmt->SetOp(fake_subgraph_op);

    std::vector<std::string> froms = {
        "matmul_y", "eltwise_add_y",
    };
    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), matched.at("clip"));
    }
    IR_OP_VAR_LINK(matched.at("clip"), matched.at("eltwise_div2_out"));
  }
};

class XPUResNetCbamFuser : public xpu::XPUFuseBase {
 public:
  XPUResNetCbamFuser() {}

  void BuildPattern() override {
    auto* input =
        VarNode("input")->assert_is_op_input("conv2d", "Input")->AsInput();

    auto* top_conv_weight = VarNode("top_conv_weight")
                                ->assert_is_op_input("conv2d", "Filter")
                                ->AsInput();
    auto* top_conv = OpNode("top_conv", "conv2d");
    auto* top_conv_out = VarNode("top_conv_out")
                             ->assert_is_op_output("conv2d", "Output")
                             ->assert_is_op_input("batch_norm", "X")
                             ->AsIntermediate();
    auto* top_bn_scale = VarNode("top_bn_scale")
                             ->assert_is_op_input("batch_norm", "Scale")
                             ->AsIntermediate();
    auto* top_bn_bias = VarNode("top_bn_bias")
                            ->assert_is_op_input("batch_norm", "Bias")
                            ->AsInput();
    auto* top_bn_mean = VarNode("top_bn_mean")
                            ->assert_is_op_input("batch_norm", "Mean")
                            ->AsIntermediate();
    auto* top_bn_var = VarNode("top_bn_variance")
                           ->assert_is_op_input("batch_norm", "Variance")
                           ->AsIntermediate();
    auto* top_bn = OpNode("top_bn", "batch_norm")->AsIntermediate();
    auto* top_bn_out = VarNode("top_bn_out")
                           ->assert_is_op_output("batch_norm", "Y")
                           ->assert_is_op_input("relu", "X")
                           ->AsIntermediate();
    auto* top_bn_mean_out = VarNode("top_bn_mean_out")
                                ->assert_is_op_output("batch_norm", "MeanOut")
                                ->AsIntermediate();
    auto* top_bn_var_out =
        VarNode("top_bn_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* top_bn_saved_mean =
        VarNode("top_bn_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* top_bn_saved_var =
        VarNode("top_bn_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();
    auto* top_relu = OpNode("top_relu", "relu")->AsIntermediate();
    auto* top_relu_out = VarNode("top_relu_out")
                             ->assert_is_op_output("relu", "Out")
                             ->assert_is_op_input("pool2d", "X")
                             ->AsIntermediate();
    auto* top_pool = OpNode("top_pool", "pool2d")->AsIntermediate();
    auto* top_pool_out =
        VarNode("top_pool_out")
            ->assert_is_op_output("pool2d", "Out")
            ->assert_is_op_input("resnet_cbam_block0", "Inputs")
            ->AsIntermediate();

    // args are left out
    auto* resnet_block0_1 =
        OpNode("resnet_block0_1", "resnet_cbam_block0")->AsIntermediate();
    auto* resnet_block0_1_out =
        VarNode("resnet_block0_1_out")
            ->assert_is_op_output("resnet_cbam_block0", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_1_1 =
        OpNode("resnet_block1_1_1", "resnet_cbam_block1")->AsIntermediate();
    auto* resnet_block1_1_1_out =
        VarNode("resnet_block1_1_1_out")
            ->assert_is_op_output("resnet_cbam_block1", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_1_2 =
        OpNode("resnet_block1_1_2", "resnet_cbam_block1")->AsIntermediate();
    auto* resnet_block1_1_2_out =
        VarNode("resnet_block1_1_2_out")
            ->assert_is_op_output("resnet_cbam_block1", "Outputs")
            ->AsIntermediate();

    auto* resnet_block0_2 =
        OpNode("resnet_block0_2", "resnet_cbam_block0")->AsIntermediate();
    auto* resnet_block0_2_out =
        VarNode("resnet_block0_2_out")
            ->assert_is_op_output("resnet_cbam_block0", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_2_1 =
        OpNode("resnet_block1_2_1", "resnet_cbam_block1")->AsIntermediate();
    auto* resnet_block1_2_1_out =
        VarNode("resnet_block1_2_1_out")
            ->assert_is_op_output("resnet_cbam_block1", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_2_2 =
        OpNode("resnet_block1_2_2", "resnet_cbam_block1")->AsIntermediate();
    auto* resnet_block1_2_2_out =
        VarNode("resnet_block1_2_2_out")
            ->assert_is_op_output("resnet_cbam_block1", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_2_3 =
        OpNode("resnet_block1_2_3", "resnet_cbam_block1")->AsIntermediate();
    auto* resnet_block1_2_3_out =
        VarNode("resnet_block1_2_3_out")
            ->assert_is_op_output("resnet_cbam_block1", "Outputs")
            ->AsIntermediate();

    auto* resnet_block0_3 =
        OpNode("resnet_block0_3", "resnet_cbam_block0")->AsIntermediate();
    auto* resnet_block0_3_out =
        VarNode("resnet_block0_3_out")
            ->assert_is_op_output("resnet_cbam_block0", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_3_1 =
        OpNode("resnet_block1_3_1", "resnet_cbam_block1")->AsIntermediate();
    auto* resnet_block1_3_1_out =
        VarNode("resnet_block1_3_1_out")
            ->assert_is_op_output("resnet_cbam_block1", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_3_2 =
        OpNode("resnet_block1_3_2", "resnet_cbam_block1")->AsIntermediate();
    auto* resnet_block1_3_2_out =
        VarNode("resnet_block1_3_2_out")
            ->assert_is_op_output("resnet_cbam_block1", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_3_3 =
        OpNode("resnet_block1_3_3", "resnet_cbam_block1")->AsIntermediate();
    auto* resnet_block1_3_3_out =
        VarNode("resnet_block1_3_3_out")
            ->assert_is_op_output("resnet_cbam_block1", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_3_4 =
        OpNode("resnet_block1_3_4", "resnet_cbam_block1")->AsIntermediate();
    auto* resnet_block1_3_4_out =
        VarNode("resnet_block1_3_4_out")
            ->assert_is_op_output("resnet_cbam_block1", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_3_5 =
        OpNode("resnet_block1_3_5", "resnet_cbam_block1")->AsIntermediate();
    auto* resnet_block1_3_5_out =
        VarNode("resnet_block1_3_5_out")
            ->assert_is_op_output("resnet_cbam_block1", "Outputs")
            ->AsIntermediate();

    auto* resnet_block0_4 =
        OpNode("resnet_block0_4", "resnet_cbam_block0")->AsIntermediate();
    auto* resnet_block0_4_out =
        VarNode("resnet_block0_4_out")
            ->assert_is_op_output("resnet_cbam_block0", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_4_1 =
        OpNode("resnet_block1_4_1", "resnet_cbam_block1")->AsIntermediate();
    auto* resnet_block1_4_1_out =
        VarNode("resnet_block1_4_1_out")
            ->assert_is_op_output("resnet_cbam_block1", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_4_2 =
        OpNode("resnet_block1_4_2", "resnet_cbam_block1")->AsIntermediate();
    auto* resnet_block1_4_2_out =
        VarNode("resnet_block1_4_2_out")
            ->assert_is_op_output("resnet_cbam_block1", "Outputs")
            ->AsIntermediate();

    auto* resnet_block2 =
        OpNode("resnet_block2", "resnet_cbam_block2")->AsIntermediate();
    auto* resnet_block2_out =
        VarNode("resnet_block2_out")
            ->assert_is_op_output("resnet_cbam_block2", "Outputs")
            ->AsOutput();

    *input >> *top_conv >> *top_conv_out >> *top_bn >> *top_bn_out >>
        *top_relu >> *top_relu_out >> *top_pool >> *top_pool_out >>
        *resnet_block0_1 >> *resnet_block0_1_out >> *resnet_block1_1_1 >>
        *resnet_block1_1_1_out >> *resnet_block1_1_2 >>
        *resnet_block1_1_2_out >> *resnet_block0_2 >> *resnet_block0_2_out >>
        *resnet_block1_2_1 >> *resnet_block1_2_1_out >> *resnet_block1_2_2 >>
        *resnet_block1_2_2_out >> *resnet_block1_2_3 >>
        *resnet_block1_2_3_out >> *resnet_block0_3 >> *resnet_block0_3_out >>
        *resnet_block1_3_1 >> *resnet_block1_3_1_out >> *resnet_block1_3_2 >>
        *resnet_block1_3_2_out >> *resnet_block1_3_3 >>
        *resnet_block1_3_3_out >> *resnet_block1_3_4 >>
        *resnet_block1_3_4_out >> *resnet_block1_3_5 >>
        *resnet_block1_3_5_out >> *resnet_block0_4 >> *resnet_block0_4_out >>
        *resnet_block1_4_1 >> *resnet_block1_4_1_out >> *resnet_block1_4_2 >>
        *resnet_block1_4_2_out >> *resnet_block2 >> *resnet_block2_out;

    *top_conv_weight >> *top_conv;
    *top_bn_scale >> *top_bn;
    *top_bn_bias >> *top_bn;
    *top_bn_mean >> *top_bn;
    *top_bn_var >> *top_bn;
    *top_bn >> *top_bn_mean_out;
    *top_bn >> *top_bn_var_out;
    *top_bn >> *top_bn_saved_mean;
    *top_bn >> *top_bn_saved_var;
  }

  void handle_placeholder_sa_conv(SSAGraph* graph,
                                  const key2nodes_t& matched,
                                  paddle::lite::Scope* scope,
                                  const std::string& filter_name,
                                  std::vector<std::string>* max_filter_name) {
    auto* filter_t = scope->FindMutableTensor(filter_name);
    int filter_len = filter_t->numel();
    float* filter_on_host = filter_t->mutable_data<float>();

    float max_f =
        paddle::lite::xpu::math::FindMaxAbs(filter_on_host, filter_len);
    std::unique_ptr<int16_t[]> filter_int16(new int16_t[filter_len]);
    paddle::lite::xpu::math::ConvertFP32ToInt16(
        filter_on_host, filter_int16.get(), max_f, filter_len);
    memcpy(filter_on_host, filter_int16.get(), filter_len * sizeof(int16_t));

    // create new arg in graph and scope
    std::string max_name = filter_name + "_max";
    max_filter_name->push_back(max_name);
    auto* max_filter_node = graph->NewArgumentNode(max_name);
    max_filter_node->arg()->is_weight = true;
    max_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    DirectedLink(max_filter_node, matched.at("top_conv"));
    auto* max_filter_t = scope->NewTensor(max_name);
    max_filter_t->Resize({4});
    float* max_ptr = max_filter_t->mutable_data<float>();
    max_ptr[0] = max_f;
    max_ptr[1] = max_f;
    max_ptr[2] = max_f;
    max_ptr[3] = max_f;
  }

  void handle_placeholder_last_fc(SSAGraph* graph,
                                  const key2nodes_t& matched,
                                  paddle::lite::Scope* scope,
                                  const std::string& filter_name,
                                  std::vector<std::string>* max_filter_name) {
    auto* filter_t = scope->FindMutableTensor(filter_name);
    auto filter_dims = filter_t->dims();
    int filter_len = filter_t->numel();
    float* filter_on_host = filter_t->mutable_data<float>();

    // XXX(miaotianxiang): Y has already been transposed in model...
    float max_f =
        paddle::lite::xpu::math::FindMaxAbs(filter_on_host, filter_len);
    std::unique_ptr<int16_t[]> filter_int16(new int16_t[filter_len]);
    paddle::lite::xpu::math::ConvertFP32ToInt16(
        filter_on_host, filter_int16.get(), max_f, filter_len);
    memcpy(filter_on_host, filter_int16.get(), filter_len * sizeof(int16_t));

    // create new arg in graph and scope
    std::string max_name = filter_name + "_max";
    max_filter_name->push_back(max_name);
    auto* max_filter_node = graph->NewArgumentNode(max_name);
    max_filter_node->arg()->is_weight = true;
    max_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    DirectedLink(max_filter_node, matched.at("top_conv"));
    auto* max_filter_t = scope->NewTensor(max_name);
    max_filter_t->Resize({4});
    float* max_ptr = max_filter_t->mutable_data<float>();
    max_ptr[0] = max_f;
    max_ptr[1] = max_f;
    max_ptr[2] = max_f;
    max_ptr[3] = max_f;
  }

  void InsertNewNode(SSAGraph* graph,
                     const key2nodes_t& matched,
                     const std::vector<Node*>& extra_input_vars) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__resnet_cbam");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    std::vector<std::string> filter_name = {
        matched.at("top_conv_weight")->arg()->name};
    std::vector<std::string> scale_name = {
        matched.at("top_bn_scale")->arg()->name};
    std::vector<std::string> bias_name = {
        matched.at("top_bn_bias")->arg()->name};
    std::vector<std::string> mean_name = {
        matched.at("top_bn_mean")->arg()->name};
    std::vector<std::string> var_name = {
        matched.at("top_bn_variance")->arg()->name};
    std::vector<std::string> max_filter_name;
    std::vector<std::string> resnet_block_vec = {
        "resnet_block0_1",
        "resnet_block1_1_1",
        "resnet_block1_1_2",
        "resnet_block0_2",
        "resnet_block1_2_1",
        "resnet_block1_2_2",
        "resnet_block1_2_3",
        "resnet_block0_3",
        "resnet_block1_3_1",
        "resnet_block1_3_2",
        "resnet_block1_3_3",
        "resnet_block1_3_4",
        "resnet_block1_3_5",
        "resnet_block0_4",
        "resnet_block1_4_1",
        "resnet_block1_4_2",
        "resnet_block2",
    };
    for (auto& block : resnet_block_vec) {
      auto* block_op_info = matched.at(block)->stmt()->op_info();
      auto block_filter_name = block_op_info->Input("Filter");
      std::copy(block_filter_name.begin(),
                block_filter_name.end(),
                std::back_inserter(filter_name));
      auto block_scale_name = block_op_info->Input("Scale");
      std::copy(block_scale_name.begin(),
                block_scale_name.end(),
                std::back_inserter(scale_name));
      auto block_bias_name = block_op_info->Input("Bias");
      std::copy(block_bias_name.begin(),
                block_bias_name.end(),
                std::back_inserter(bias_name));
      auto block_mean_name = block_op_info->Input("Mean");
      std::copy(block_mean_name.begin(),
                block_mean_name.end(),
                std::back_inserter(mean_name));
      auto block_var_name = block_op_info->Input("Var");
      std::copy(block_var_name.begin(),
                block_var_name.end(),
                std::back_inserter(var_name));
    }

    auto* resnet_cbam_stmt = matched.at("top_conv")->stmt();
    auto* scope = resnet_cbam_stmt->op()->scope();
    for (size_t i = 0; i < filter_name.size(); ++i) {
      if (scale_name[i] == "placeholder_sa_conv") {
        handle_placeholder_sa_conv(
            graph, matched, scope, filter_name[i], &max_filter_name);
        continue;
      } else if (scale_name[i] == "placeholder_last_fc") {
        handle_placeholder_last_fc(
            graph, matched, scope, filter_name[i], &max_filter_name);
        continue;
      }

      auto* filter_t = scope->FindMutableTensor(filter_name[i]);
      auto* scale_t = scope->FindMutableTensor(scale_name[i]);
      auto* bias_t = scope->FindMutableTensor(bias_name[i]);
      auto* mean_t = scope->FindMutableTensor(mean_name[i]);
      auto* var_t = scope->FindMutableTensor(var_name[i]);

      int mean_len = mean_t->numel();
      int filter_len = filter_t->numel();
      int filter_stride = filter_len / mean_len;

      float* filter_on_host = filter_t->mutable_data<float>();
      float* scale_on_host = scale_t->mutable_data<float>();
      float* bias_on_host = bias_t->mutable_data<float>();
      float* mean_on_host = mean_t->mutable_data<float>();
      float* var_on_host = var_t->mutable_data<float>();

      // Perform preprocess
      for (int i = 0; i < mean_len; ++i) {
        scale_on_host[i] = scale_on_host[i] / sqrtf(var_on_host[i] + 0.00001f);
      }
      for (int i = 0; i < mean_len; ++i) {
        for (int j = 0; j < filter_stride; ++j) {
          filter_on_host[i * filter_stride + j] *= scale_on_host[i];
        }
      }
      for (int i = 0; i < mean_len; ++i) {
        bias_on_host[i] += -mean_on_host[i] * scale_on_host[i];
      }

      float max_f =
          paddle::lite::xpu::math::FindMaxAbs(filter_on_host, filter_len);
      std::unique_ptr<int16_t[]> filter_int16(new int16_t[filter_len]);
      paddle::lite::xpu::math::ConvertFP32ToInt16(
          filter_on_host, filter_int16.get(), max_f, filter_len);
      memcpy(filter_on_host, filter_int16.get(), filter_len * sizeof(int16_t));

      // create new arg in graph and scope
      std::string max_name = filter_name[i] + "_max";
      max_filter_name.push_back(max_name);
      auto* max_filter_node = graph->NewArgumentNode(max_name);
      max_filter_node->arg()->is_weight = true;
      max_filter_node->arg()->type = LiteType::GetTensorTy(
          TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
      DirectedLink(max_filter_node, matched.at("top_conv"));
      auto* max_filter_t = scope->NewTensor(max_name);
      max_filter_t->Resize({4});
      float* max_ptr = max_filter_t->mutable_data<float>();
      max_ptr[0] = max_f;
      max_ptr[1] = max_f;
      max_ptr[2] = max_f;
      max_ptr[3] = max_f;
    }
    op_desc.SetInput("Filter", filter_name);
    op_desc.SetInput("Bias", bias_name);
    op_desc.SetInput("MaxFilter", max_filter_name);
    op_desc.SetOutput("Output", {matched.at("resnet_block2_out")->arg()->name});
    op_desc.SetAttr<int>("xpu", 1);
    auto* block2_op_info = matched.at("resnet_block2")->stmt()->op_info();
    op_desc.SetAttr<float>("pool_p", block2_op_info->GetAttr<float>("pool_p"));

    auto resnet_cbam_op = LiteOpRegistry::Global().Create(op_desc.Type());
    resnet_cbam_op->Attach(op_desc, scope);
    resnet_cbam_op->SetValidPlaces(resnet_cbam_stmt->op()->valid_places());
    auto kernels =
        resnet_cbam_op->CreateKernels(resnet_cbam_op->valid_places());
    resnet_cbam_stmt->SetOp(resnet_cbam_op);
    resnet_cbam_stmt->SetKernels(std::move(kernels));

    IR_NODE_LINK_TO(matched.at("top_bn_bias"), matched.at("top_conv"));
    for (auto* node : extra_input_vars) {
      IR_NODE_LINK_TO(node, matched.at("top_conv"));
    }
    IR_OP_VAR_LINK(matched.at("top_conv"), matched.at("resnet_block2_out"));
  }
};

}  // namespace fusion

class XPUResNetCbamFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;

    bool changed = false;
    SSAGraph backup;
    backup.CloneFrom(*graph);

    fusion::XPUResNetCbamBlock0Fuser block0_fuser;
    changed |= block0_fuser(graph.get());
    fusion::XPUResNetCbamBlock1Fuser block1_fuser;
    changed |= block1_fuser(graph.get());
    fusion::XPUResNetCbamBlock2Fuser block2_fuser;
    changed |= block2_fuser(graph.get());
    fusion::XPUResNetCbamFuser resnet_fuser;
    size_t n_matches = resnet_fuser(graph.get());

    if (changed && !n_matches) {
      // Restore graph from backuped one if no whole ResNetCbam graph was found
      graph->CloneFrom(backup);
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__resnet_cbam_fuse_pass,
                  paddle::lite::mir::XPUResNetCbamFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__resnet_cbam");
