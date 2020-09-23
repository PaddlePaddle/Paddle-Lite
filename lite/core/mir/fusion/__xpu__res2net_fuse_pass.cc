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

class XPURes2NetBlock0Fuser : public FuseBase {
 public:
  XPURes2NetBlock0Fuser() {}

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
                               ->assert_is_op_input("split", "X")
                               ->AsIntermediate();
    // split
    auto* split = OpNode("split", "split")
                      ->assert_op_attr_satisfied<int>(
                          "axis", [](const int& attr) { return attr == 1; })
                      ->assert_op_attr_satisfied<int>(
                          "num", [](const int attr) { return attr == 4; })
                      ->AsIntermediate();
    auto* split_out_0 = VarNode("split_out_0")
                            ->assert_is_op_nth_output("split", "Out", 0)
                            ->assert_is_op_input("conv2d", "Input")
                            ->AsIntermediate();
    auto* split_out_1 = VarNode("split_out_1")
                            ->assert_is_op_nth_output("split", "Out", 1)
                            ->assert_is_op_input("elementwise_add", "X")
                            ->AsIntermediate();
    auto* split_out_2 = VarNode("split_out_2")
                            ->assert_is_op_nth_output("split", "Out", 2)
                            ->assert_is_op_input("elementwise_add", "X")
                            ->AsIntermediate();
    auto* split_out_3 = VarNode("split_out_3")
                            ->assert_is_op_nth_output("split", "Out", 3)
                            ->assert_is_op_nth_input("concat", "X", 3)
                            ->AsIntermediate();
    // concat
    auto* concat = OpNode("concat", "concat")->AsIntermediate();
    auto* concat_out = VarNode("concat_out")
                           ->assert_is_op_output("concat", "Out")
                           ->assert_is_op_input("conv2d", "Input")
                           ->AsIntermediate();
    // conv
    auto* left_conv2_weight = VarNode("left_conv2_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv2 = OpNode("left_conv2", "conv2d")->AsIntermediate();
    auto* left_conv2_out = VarNode("left_conv2_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
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
    // relu
    auto* left_relu2 = OpNode("left_relu2", "relu")->AsIntermediate();
    auto* left_relu2_out = VarNode("left_relu2_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_input("elementwise_add", "Y")
                               ->assert_is_op_nth_input("concat", "X", 0)
                               ->AsIntermediate();
    // add
    auto* add1 = OpNode("add1", "elementwise_add")->AsIntermediate();
    auto* add1_out = VarNode("add1_out")
                         ->assert_is_op_input("conv2d", "Input")
                         ->AsIntermediate();
    // conv
    auto* left_conv3_weight = VarNode("left_conv3_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv3 = OpNode("left_conv3", "conv2d")->AsIntermediate();
    auto* left_conv3_out = VarNode("left_conv3_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
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
                             ->assert_is_op_input("relu", "X")
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
    // relu
    auto* left_relu3 = OpNode("left_relu3", "relu")->AsIntermediate();
    auto* left_relu3_out = VarNode("left_relu3_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_nth_input("concat", "X", 1)
                               ->AsIntermediate();
    // add
    auto* add2 = OpNode("add2", "elementwise_add")->AsIntermediate();
    auto* add2_out = VarNode("add2_out")
                         ->assert_is_op_input("conv2d", "Input")
                         ->AsIntermediate();
    // conv
    auto* left_conv4_weight = VarNode("left_conv4_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv4 = OpNode("left_conv4", "conv2d")->AsIntermediate();
    auto* left_conv4_out = VarNode("left_conv4_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
    auto* left_bn4_scale = VarNode("left_bn4_scale")
                               ->assert_is_op_input("batch_norm", "Scale")
                               ->AsIntermediate();
    auto* left_bn4_bias = VarNode("left_bn4_bias")
                              ->assert_is_op_input("batch_norm", "Bias")
                              ->AsInput();
    auto* left_bn4_mean = VarNode("left_bn4_mean")
                              ->assert_is_op_input("batch_norm", "Mean")
                              ->AsIntermediate();
    auto* left_bn4_var = VarNode("left_bn4_variance")
                             ->assert_is_op_input("batch_norm", "Variance")
                             ->AsIntermediate();
    auto* left_bn4 = OpNode("left_bn4", "batch_norm")->AsIntermediate();
    auto* left_bn4_out = VarNode("left_bn4_out")
                             ->assert_is_op_output("batch_norm", "Y")
                             ->assert_is_op_input("relu", "X")
                             ->AsIntermediate();
    auto* left_bn4_mean_out = VarNode("left_bn4_mean_out")
                                  ->assert_is_op_output("batch_norm", "MeanOut")
                                  ->AsIntermediate();
    auto* left_bn4_var_out =
        VarNode("left_bn4_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* left_bn4_saved_mean =
        VarNode("left_bn4_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* left_bn4_saved_var =
        VarNode("left_bn4_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();
    // relu
    auto* left_relu4 = OpNode("left_relu4", "relu")->AsIntermediate();
    auto* left_relu4_out = VarNode("left_relu4_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_nth_input("concat", "X", 2)
                               ->AsIntermediate();
    // conv
    auto* left_conv5_weight = VarNode("left_conv5_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv5 = OpNode("left_conv5", "conv2d")->AsIntermediate();
    auto* left_conv5_out = VarNode("left_conv5_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
    auto* left_bn5_scale = VarNode("left_bn5_scale")
                               ->assert_is_op_input("batch_norm", "Scale")
                               ->AsIntermediate();
    auto* left_bn5_bias = VarNode("left_bn5_bias")
                              ->assert_is_op_input("batch_norm", "Bias")
                              ->AsInput();
    auto* left_bn5_mean = VarNode("left_bn5_mean")
                              ->assert_is_op_input("batch_norm", "Mean")
                              ->AsIntermediate();
    auto* left_bn5_var = VarNode("left_bn5_variance")
                             ->assert_is_op_input("batch_norm", "Variance")
                             ->AsIntermediate();
    auto* left_bn5 = OpNode("left_bn5", "batch_norm")->AsIntermediate();
    auto* left_bn5_out = VarNode("left_bn5_out")
                             ->assert_is_op_output("batch_norm", "Y")
                             ->assert_is_op_input("elementwise_add", "Y")
                             ->AsIntermediate();
    auto* left_bn5_mean_out = VarNode("left_bn5_mean_out")
                                  ->assert_is_op_output("batch_norm", "MeanOut")
                                  ->AsIntermediate();
    auto* left_bn5_var_out =
        VarNode("left_bn5_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* left_bn5_saved_mean =
        VarNode("left_bn5_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* left_bn5_saved_var =
        VarNode("left_bn5_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();
    // conv
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
                              ->assert_is_op_input("elementwise_add", "X")
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
        *left_relu1 >> *left_relu1_out >> *split;

    *split >> *split_out_3 >> *concat;
    *split >> *split_out_0;
    *split >> *split_out_1 >> *add1;
    *split >> *split_out_2 >> *add2;

    *split_out_0 >> *left_conv2 >> *left_conv2_out >> *left_bn2 >>
        *left_bn2_out >> *left_relu2 >> *left_relu2_out >> *concat;
    *left_relu2_out >> *add1 >> *add1_out >> *left_conv3 >> *left_conv3_out >>
        *left_bn3 >> *left_bn3_out >> *left_relu3 >> *left_relu3_out >> *concat;
    *left_relu3_out >> *add2 >> *add2_out >> *left_conv4 >> *left_conv4_out >>
        *left_bn4 >> *left_bn4_out >> *left_relu4 >> *left_relu4_out >> *concat;

    *concat >> *concat_out >> *left_conv5 >> *left_conv5_out >> *left_bn5 >>
        *left_bn5_out >> *add;

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

    *left_conv4_weight >> *left_conv4;
    *left_bn4_scale >> *left_bn4;
    *left_bn4_bias >> *left_bn4;
    *left_bn4_mean >> *left_bn4;
    *left_bn4_var >> *left_bn4;
    *left_bn4 >> *left_bn4_mean_out;
    *left_bn4 >> *left_bn4_var_out;
    *left_bn4 >> *left_bn4_saved_mean;
    *left_bn4 >> *left_bn4_saved_var;

    *left_conv5_weight >> *left_conv5;
    *left_bn5_scale >> *left_bn5;
    *left_bn5_bias >> *left_bn5;
    *left_bn5_mean >> *left_bn5;
    *left_bn5_var >> *left_bn5;
    *left_bn5 >> *left_bn5_mean_out;
    *left_bn5 >> *left_bn5_var_out;
    *left_bn5 >> *left_bn5_saved_mean;
    *left_bn5 >> *left_bn5_saved_var;

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
    op_desc.SetType("res2net_block0");
    op_desc.SetInput("Inputs", {matched.at("input")->arg()->name});
    op_desc.SetInput("Filter",
                     {
                         matched.at("left_conv1_weight")->arg()->name,
                         matched.at("left_conv2_weight")->arg()->name,
                         matched.at("left_conv3_weight")->arg()->name,
                         matched.at("left_conv4_weight")->arg()->name,
                         matched.at("left_conv5_weight")->arg()->name,
                         matched.at("right_conv1_weight")->arg()->name,
                     });
    op_desc.SetInput("Scale",
                     {
                         matched.at("left_bn1_scale")->arg()->name,
                         matched.at("left_bn2_scale")->arg()->name,
                         matched.at("left_bn3_scale")->arg()->name,
                         matched.at("left_bn4_scale")->arg()->name,
                         matched.at("left_bn5_scale")->arg()->name,
                         matched.at("right_bn1_scale")->arg()->name,
                     });
    op_desc.SetInput("Bias",
                     {
                         matched.at("left_bn1_bias")->arg()->name,
                         matched.at("left_bn2_bias")->arg()->name,
                         matched.at("left_bn3_bias")->arg()->name,
                         matched.at("left_bn4_bias")->arg()->name,
                         matched.at("left_bn5_bias")->arg()->name,
                         matched.at("right_bn1_bias")->arg()->name,
                     });
    op_desc.SetInput("Mean",
                     {
                         matched.at("left_bn1_mean")->arg()->name,
                         matched.at("left_bn2_mean")->arg()->name,
                         matched.at("left_bn3_mean")->arg()->name,
                         matched.at("left_bn4_mean")->arg()->name,
                         matched.at("left_bn5_mean")->arg()->name,
                         matched.at("right_bn1_mean")->arg()->name,
                     });
    op_desc.SetInput("Var",
                     {
                         matched.at("left_bn1_variance")->arg()->name,
                         matched.at("left_bn2_variance")->arg()->name,
                         matched.at("left_bn3_variance")->arg()->name,
                         matched.at("left_bn4_variance")->arg()->name,
                         matched.at("left_bn5_variance")->arg()->name,
                         matched.at("right_bn1_variance")->arg()->name,
                     });

    op_desc.SetOutput("Outputs", {matched.at("relu_out")->arg()->name});
    // keep these to fool SubgraphOp::AttachImpl()
    op_desc.SetAttr<int>("sub_block", 0);
    op_desc.SetAttr<std::vector<std::string>>("input_data_names", {});
    op_desc.SetAttr<std::vector<std::string>>("output_data_names", {});

    auto block0_stmt = matched.at("left_conv1")->stmt();
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
        "left_conv5_weight",
        "right_conv1_weight",
        "left_bn1_bias",
        "left_bn2_bias",
        "left_bn3_bias",
        "left_bn4_bias",
        "left_bn5_bias",
        "right_bn1_bias",
    };
    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), matched.at("left_conv1"));
    }
    IR_OP_VAR_LINK(matched.at("left_conv1"), matched.at("relu_out"));
  }
};

class XPURes2NetBlock1Fuser : public FuseBase {
 public:
  XPURes2NetBlock1Fuser() {}

  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("conv2d", "Input")
                      ->assert_is_op_input("pool2d", "X")
                      ->AsInput();
    // conv
    auto* left_conv1_weight = VarNode("left_conv1_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv1 = OpNode("left_conv1", "conv2d");
    auto* left_conv1_out = VarNode("left_conv1_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
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
    // relu
    auto* left_relu1 = OpNode("left_relu1", "relu")->AsIntermediate();
    auto* left_relu1_out = VarNode("left_relu1_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_input("split", "X")
                               ->AsIntermediate();
    // split
    auto* split = OpNode("split", "split")
                      ->assert_op_attr_satisfied<int>(
                          "axis", [](const int& attr) { return attr == 1; })
                      ->assert_op_attr_satisfied<int>(
                          "num", [](const int attr) { return attr == 4; })
                      ->AsIntermediate();
    auto* split_out_0 = VarNode("split_out_0")
                            ->assert_is_op_nth_output("split", "Out", 0)
                            ->assert_is_op_input("conv2d", "Input")
                            ->AsIntermediate();
    auto* split_out_1 = VarNode("split_out_1")
                            ->assert_is_op_nth_output("split", "Out", 1)
                            ->assert_is_op_input("conv2d", "Input")
                            ->AsIntermediate();
    auto* split_out_2 = VarNode("split_out_2")
                            ->assert_is_op_nth_output("split", "Out", 2)
                            ->assert_is_op_input("conv2d", "Input")
                            ->AsIntermediate();
    auto* split_out_3 = VarNode("split_out_3")
                            ->assert_is_op_nth_output("split", "Out", 3)
                            ->assert_is_op_input("pool2d", "X")
                            ->AsIntermediate();
    // concat
    auto* concat = OpNode("concat", "concat")->AsIntermediate();
    auto* concat_out = VarNode("concat_out")
                           ->assert_is_op_output("concat", "Out")
                           ->assert_is_op_input("conv2d", "Input")
                           ->AsIntermediate();
    // conv
    auto* left_conv2_weight = VarNode("left_conv2_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv2 = OpNode("left_conv2", "conv2d")->AsIntermediate();
    auto* left_conv2_out = VarNode("left_conv2_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
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
    // relu
    auto* left_relu2 = OpNode("left_relu2", "relu")->AsIntermediate();
    auto* left_relu2_out = VarNode("left_relu2_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_nth_input("concat", "X", 0)
                               ->AsIntermediate();
    // conv
    auto* left_conv3_weight = VarNode("left_conv3_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv3 = OpNode("left_conv3", "conv2d")->AsIntermediate();
    auto* left_conv3_out = VarNode("left_conv3_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
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
                             ->assert_is_op_input("relu", "X")
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
    // relu
    auto* left_relu3 = OpNode("left_relu3", "relu")->AsIntermediate();
    auto* left_relu3_out = VarNode("left_relu3_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_nth_input("concat", "X", 1)
                               ->AsIntermediate();
    // conv
    auto* left_conv4_weight = VarNode("left_conv4_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv4 = OpNode("left_conv4", "conv2d")->AsIntermediate();
    auto* left_conv4_out = VarNode("left_conv4_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
    auto* left_bn4_scale = VarNode("left_bn4_scale")
                               ->assert_is_op_input("batch_norm", "Scale")
                               ->AsIntermediate();
    auto* left_bn4_bias = VarNode("left_bn4_bias")
                              ->assert_is_op_input("batch_norm", "Bias")
                              ->AsInput();
    auto* left_bn4_mean = VarNode("left_bn4_mean")
                              ->assert_is_op_input("batch_norm", "Mean")
                              ->AsIntermediate();
    auto* left_bn4_var = VarNode("left_bn4_variance")
                             ->assert_is_op_input("batch_norm", "Variance")
                             ->AsIntermediate();
    auto* left_bn4 = OpNode("left_bn4", "batch_norm")->AsIntermediate();
    auto* left_bn4_out = VarNode("left_bn4_out")
                             ->assert_is_op_output("batch_norm", "Y")
                             ->assert_is_op_input("relu", "X")
                             ->AsIntermediate();
    auto* left_bn4_mean_out = VarNode("left_bn4_mean_out")
                                  ->assert_is_op_output("batch_norm", "MeanOut")
                                  ->AsIntermediate();
    auto* left_bn4_var_out =
        VarNode("left_bn4_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* left_bn4_saved_mean =
        VarNode("left_bn4_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* left_bn4_saved_var =
        VarNode("left_bn4_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();
    // relu
    auto* left_relu4 = OpNode("left_relu4", "relu")->AsIntermediate();
    auto* left_relu4_out = VarNode("left_relu4_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_nth_input("concat", "X", 2)
                               ->AsIntermediate();

    // pool
    auto* left_pool = OpNode("left_pool", "pool2d")->AsIntermediate();
    auto* left_pool_out = VarNode("left_pool_out")
                              ->assert_is_op_output("pool2d", "Out")
                              ->assert_is_op_nth_input("concat", "X", 3)
                              ->AsIntermediate();

    // conv
    auto* left_conv5_weight = VarNode("left_conv5_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv5 = OpNode("left_conv5", "conv2d")->AsIntermediate();
    auto* left_conv5_out = VarNode("left_conv5_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
    auto* left_bn5_scale = VarNode("left_bn5_scale")
                               ->assert_is_op_input("batch_norm", "Scale")
                               ->AsIntermediate();
    auto* left_bn5_bias = VarNode("left_bn5_bias")
                              ->assert_is_op_input("batch_norm", "Bias")
                              ->AsInput();
    auto* left_bn5_mean = VarNode("left_bn5_mean")
                              ->assert_is_op_input("batch_norm", "Mean")
                              ->AsIntermediate();
    auto* left_bn5_var = VarNode("left_bn5_variance")
                             ->assert_is_op_input("batch_norm", "Variance")
                             ->AsIntermediate();
    auto* left_bn5 = OpNode("left_bn5", "batch_norm")->AsIntermediate();
    auto* left_bn5_out = VarNode("left_bn5_out")
                             ->assert_is_op_output("batch_norm", "Y")
                             ->assert_is_op_input("elementwise_add", "Y")
                             ->AsIntermediate();
    auto* left_bn5_mean_out = VarNode("left_bn5_mean_out")
                                  ->assert_is_op_output("batch_norm", "MeanOut")
                                  ->AsIntermediate();
    auto* left_bn5_var_out =
        VarNode("left_bn5_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* left_bn5_saved_mean =
        VarNode("left_bn5_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* left_bn5_saved_var =
        VarNode("left_bn5_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();

    auto* right_pool = OpNode("right_pool", "pool2d")->AsIntermediate();
    auto* right_pool_out = VarNode("right_pool_out")
                               ->assert_is_op_output("pool2d", "Out")
                               ->assert_is_op_input("conv2d", "Input")
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
                              ->assert_is_op_input("elementwise_add", "X")
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
        *left_relu1 >> *left_relu1_out >> *split;
    *split >> *split_out_0 >> *left_conv2 >> *left_conv2_out >> *left_bn2 >>
        *left_bn2_out >> *left_relu2 >> *left_relu2_out >> *concat;
    *split >> *split_out_1 >> *left_conv3 >> *left_conv3_out >> *left_bn3 >>
        *left_bn3_out >> *left_relu3 >> *left_relu3_out >> *concat;
    *split >> *split_out_2 >> *left_conv4 >> *left_conv4_out >> *left_bn4 >>
        *left_bn4_out >> *left_relu4 >> *left_relu4_out >> *concat;
    *split >> *split_out_3 >> *left_pool >> *left_pool_out >> *concat;
    *concat >> *concat_out >> *left_conv5 >> *left_conv5_out >> *left_bn5 >>
        *left_bn5_out >> *add;

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

    *left_conv4_weight >> *left_conv4;
    *left_bn4_scale >> *left_bn4;
    *left_bn4_bias >> *left_bn4;
    *left_bn4_mean >> *left_bn4;
    *left_bn4_var >> *left_bn4;
    *left_bn4 >> *left_bn4_mean_out;
    *left_bn4 >> *left_bn4_var_out;
    *left_bn4 >> *left_bn4_saved_mean;
    *left_bn4 >> *left_bn4_saved_var;

    *left_conv5_weight >> *left_conv5;
    *left_bn5_scale >> *left_bn5;
    *left_bn5_bias >> *left_bn5;
    *left_bn5_mean >> *left_bn5;
    *left_bn5_var >> *left_bn5;
    *left_bn5 >> *left_bn5_mean_out;
    *left_bn5 >> *left_bn5_var_out;
    *left_bn5 >> *left_bn5_saved_mean;
    *left_bn5 >> *left_bn5_saved_var;

    *input >> *right_pool >> *right_pool_out >> *right_conv1 >>
        *right_conv1_out >> *right_bn1 >> *right_bn1_out >> *add;

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
    op_desc.SetType("res2net_block1");
    op_desc.SetInput("Inputs", {matched.at("input")->arg()->name});
    op_desc.SetInput("Filter",
                     {
                         matched.at("left_conv1_weight")->arg()->name,
                         matched.at("left_conv2_weight")->arg()->name,
                         matched.at("left_conv3_weight")->arg()->name,
                         matched.at("left_conv4_weight")->arg()->name,
                         matched.at("left_conv5_weight")->arg()->name,
                         matched.at("right_conv1_weight")->arg()->name,
                     });
    op_desc.SetInput("Scale",
                     {
                         matched.at("left_bn1_scale")->arg()->name,
                         matched.at("left_bn2_scale")->arg()->name,
                         matched.at("left_bn3_scale")->arg()->name,
                         matched.at("left_bn4_scale")->arg()->name,
                         matched.at("left_bn5_scale")->arg()->name,
                         matched.at("right_bn1_scale")->arg()->name,
                     });
    op_desc.SetInput("Bias",
                     {
                         matched.at("left_bn1_bias")->arg()->name,
                         matched.at("left_bn2_bias")->arg()->name,
                         matched.at("left_bn3_bias")->arg()->name,
                         matched.at("left_bn4_bias")->arg()->name,
                         matched.at("left_bn5_bias")->arg()->name,
                         matched.at("right_bn1_bias")->arg()->name,
                     });
    op_desc.SetInput("Mean",
                     {
                         matched.at("left_bn1_mean")->arg()->name,
                         matched.at("left_bn2_mean")->arg()->name,
                         matched.at("left_bn3_mean")->arg()->name,
                         matched.at("left_bn4_mean")->arg()->name,
                         matched.at("left_bn5_mean")->arg()->name,
                         matched.at("right_bn1_mean")->arg()->name,
                     });
    op_desc.SetInput("Var",
                     {
                         matched.at("left_bn1_variance")->arg()->name,
                         matched.at("left_bn2_variance")->arg()->name,
                         matched.at("left_bn3_variance")->arg()->name,
                         matched.at("left_bn4_variance")->arg()->name,
                         matched.at("left_bn5_variance")->arg()->name,
                         matched.at("right_bn1_variance")->arg()->name,
                     });
    op_desc.SetOutput("Outputs", {matched.at("relu_out")->arg()->name});
    // keep these to fool SubgraphOp::AttachImpl()
    op_desc.SetAttr<int>("sub_block", 0);
    op_desc.SetAttr<std::vector<std::string>>("input_data_names", {});
    op_desc.SetAttr<std::vector<std::string>>("output_data_names", {});

    auto block0_stmt = matched.at("left_conv1")->stmt();
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
        "left_conv5_weight",
        "right_conv1_weight",
        "left_bn1_bias",
        "left_bn2_bias",
        "left_bn3_bias",
        "left_bn4_bias",
        "left_bn5_bias",
        "right_bn1_bias",
    };
    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), matched.at("left_conv1"));
    }
    IR_OP_VAR_LINK(matched.at("left_conv1"), matched.at("relu_out"));
  }
};

class XPURes2NetBlock2Fuser : public FuseBase {
 public:
  XPURes2NetBlock2Fuser() {}

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
                               ->assert_is_op_input("split", "X")
                               ->AsIntermediate();
    // split
    auto* split = OpNode("split", "split")
                      ->assert_op_attr_satisfied<int>(
                          "axis", [](const int& attr) { return attr == 1; })
                      ->assert_op_attr_satisfied<int>(
                          "num", [](const int attr) { return attr == 4; })
                      ->AsIntermediate();
    auto* split_out_0 = VarNode("split_out_0")
                            ->assert_is_op_nth_output("split", "Out", 0)
                            ->assert_is_op_input("conv2d", "Input")
                            ->AsIntermediate();
    auto* split_out_1 = VarNode("split_out_1")
                            ->assert_is_op_nth_output("split", "Out", 1)
                            ->assert_is_op_input("elementwise_add", "X")
                            ->AsIntermediate();
    auto* split_out_2 = VarNode("split_out_2")
                            ->assert_is_op_nth_output("split", "Out", 2)
                            ->assert_is_op_input("elementwise_add", "X")
                            ->AsIntermediate();
    auto* split_out_3 = VarNode("split_out_3")
                            ->assert_is_op_nth_output("split", "Out", 3)
                            ->assert_is_op_nth_input("concat", "X", 3)
                            ->AsIntermediate();
    // concat
    auto* concat = OpNode("concat", "concat")->AsIntermediate();
    auto* concat_out = VarNode("concat_out")
                           ->assert_is_op_output("concat", "Out")
                           ->assert_is_op_input("conv2d", "Input")
                           ->AsIntermediate();
    // conv
    auto* left_conv2_weight = VarNode("left_conv2_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv2 = OpNode("left_conv2", "conv2d")->AsIntermediate();
    auto* left_conv2_out = VarNode("left_conv2_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
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
    // relu
    auto* left_relu2 = OpNode("left_relu2", "relu")->AsIntermediate();
    auto* left_relu2_out = VarNode("left_relu2_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_input("elementwise_add", "Y")
                               ->assert_is_op_nth_input("concat", "X", 0)
                               ->AsIntermediate();
    // add
    auto* add1 = OpNode("add1", "elementwise_add")->AsIntermediate();
    auto* add1_out = VarNode("add1_out")
                         ->assert_is_op_input("conv2d", "Input")
                         ->AsIntermediate();
    // conv
    auto* left_conv3_weight = VarNode("left_conv3_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv3 = OpNode("left_conv3", "conv2d")->AsIntermediate();
    auto* left_conv3_out = VarNode("left_conv3_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
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
                             ->assert_is_op_input("relu", "X")
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
    // relu
    auto* left_relu3 = OpNode("left_relu3", "relu")->AsIntermediate();
    auto* left_relu3_out = VarNode("left_relu3_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_nth_input("concat", "X", 1)
                               ->AsIntermediate();
    // add
    auto* add2 = OpNode("add2", "elementwise_add")->AsIntermediate();
    auto* add2_out = VarNode("add2_out")
                         ->assert_is_op_input("conv2d", "Input")
                         ->AsIntermediate();
    // conv
    auto* left_conv4_weight = VarNode("left_conv4_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv4 = OpNode("left_conv4", "conv2d")->AsIntermediate();
    auto* left_conv4_out = VarNode("left_conv4_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
    auto* left_bn4_scale = VarNode("left_bn4_scale")
                               ->assert_is_op_input("batch_norm", "Scale")
                               ->AsIntermediate();
    auto* left_bn4_bias = VarNode("left_bn4_bias")
                              ->assert_is_op_input("batch_norm", "Bias")
                              ->AsInput();
    auto* left_bn4_mean = VarNode("left_bn4_mean")
                              ->assert_is_op_input("batch_norm", "Mean")
                              ->AsIntermediate();
    auto* left_bn4_var = VarNode("left_bn4_variance")
                             ->assert_is_op_input("batch_norm", "Variance")
                             ->AsIntermediate();
    auto* left_bn4 = OpNode("left_bn4", "batch_norm")->AsIntermediate();
    auto* left_bn4_out = VarNode("left_bn4_out")
                             ->assert_is_op_output("batch_norm", "Y")
                             ->assert_is_op_input("relu", "X")
                             ->AsIntermediate();
    auto* left_bn4_mean_out = VarNode("left_bn4_mean_out")
                                  ->assert_is_op_output("batch_norm", "MeanOut")
                                  ->AsIntermediate();
    auto* left_bn4_var_out =
        VarNode("left_bn4_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* left_bn4_saved_mean =
        VarNode("left_bn4_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* left_bn4_saved_var =
        VarNode("left_bn4_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();
    // relu
    auto* left_relu4 = OpNode("left_relu4", "relu")->AsIntermediate();
    auto* left_relu4_out = VarNode("left_relu4_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_nth_input("concat", "X", 2)
                               ->AsIntermediate();
    // conv
    auto* left_conv5_weight = VarNode("left_conv5_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* left_conv5 = OpNode("left_conv5", "conv2d")->AsIntermediate();
    auto* left_conv5_out = VarNode("left_conv5_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    // bn
    auto* left_bn5_scale = VarNode("left_bn5_scale")
                               ->assert_is_op_input("batch_norm", "Scale")
                               ->AsIntermediate();
    auto* left_bn5_bias = VarNode("left_bn5_bias")
                              ->assert_is_op_input("batch_norm", "Bias")
                              ->AsInput();
    auto* left_bn5_mean = VarNode("left_bn5_mean")
                              ->assert_is_op_input("batch_norm", "Mean")
                              ->AsIntermediate();
    auto* left_bn5_var = VarNode("left_bn5_variance")
                             ->assert_is_op_input("batch_norm", "Variance")
                             ->AsIntermediate();
    auto* left_bn5 = OpNode("left_bn5", "batch_norm")->AsIntermediate();
    auto* left_bn5_out = VarNode("left_bn5_out")
                             ->assert_is_op_output("batch_norm", "Y")
                             ->assert_is_op_input("elementwise_add", "Y")
                             ->AsIntermediate();
    auto* left_bn5_mean_out = VarNode("left_bn5_mean_out")
                                  ->assert_is_op_output("batch_norm", "MeanOut")
                                  ->AsIntermediate();
    auto* left_bn5_var_out =
        VarNode("left_bn5_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* left_bn5_saved_mean =
        VarNode("left_bn5_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* left_bn5_saved_var =
        VarNode("left_bn5_saved_var")
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
        *left_relu1 >> *left_relu1_out >> *split;
    *split >> *split_out_3 >> *concat;
    *split >> *split_out_0;
    *split >> *split_out_1 >> *add1;
    *split >> *split_out_2 >> *add2;

    *split_out_0 >> *left_conv2 >> *left_conv2_out >> *left_bn2 >>
        *left_bn2_out >> *left_relu2 >> *left_relu2_out >> *concat;
    *left_relu2_out >> *add1 >> *add1_out >> *left_conv3 >> *left_conv3_out >>
        *left_bn3 >> *left_bn3_out >> *left_relu3 >> *left_relu3_out >> *concat;
    *left_relu3_out >> *add2 >> *add2_out >> *left_conv4 >> *left_conv4_out >>
        *left_bn4 >> *left_bn4_out >> *left_relu4 >> *left_relu4_out >> *concat;

    *concat >> *concat_out >> *left_conv5 >> *left_conv5_out >> *left_bn5 >>
        *left_bn5_out >> *add;

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

    *left_conv4_weight >> *left_conv4;
    *left_bn4_scale >> *left_bn4;
    *left_bn4_bias >> *left_bn4;
    *left_bn4_mean >> *left_bn4;
    *left_bn4_var >> *left_bn4;
    *left_bn4 >> *left_bn4_mean_out;
    *left_bn4 >> *left_bn4_var_out;
    *left_bn4 >> *left_bn4_saved_mean;
    *left_bn4 >> *left_bn4_saved_var;

    *left_conv5_weight >> *left_conv5;
    *left_bn5_scale >> *left_bn5;
    *left_bn5_bias >> *left_bn5;
    *left_bn5_mean >> *left_bn5;
    *left_bn5_var >> *left_bn5;
    *left_bn5 >> *left_bn5_mean_out;
    *left_bn5 >> *left_bn5_var_out;
    *left_bn5 >> *left_bn5_saved_mean;
    *left_bn5 >> *left_bn5_saved_var;

    *input >> *add >> *add_out >> *relu >> *relu_out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("res2net_block2");
    op_desc.SetInput("Inputs", {matched.at("input")->arg()->name});
    op_desc.SetInput("Filter",
                     {
                         matched.at("left_conv1_weight")->arg()->name,
                         matched.at("left_conv2_weight")->arg()->name,
                         matched.at("left_conv3_weight")->arg()->name,
                         matched.at("left_conv4_weight")->arg()->name,
                         matched.at("left_conv5_weight")->arg()->name,
                     });
    op_desc.SetInput("Scale",
                     {
                         matched.at("left_bn1_scale")->arg()->name,
                         matched.at("left_bn2_scale")->arg()->name,
                         matched.at("left_bn3_scale")->arg()->name,
                         matched.at("left_bn4_scale")->arg()->name,
                         matched.at("left_bn5_scale")->arg()->name,
                     });
    op_desc.SetInput("Bias",
                     {
                         matched.at("left_bn1_bias")->arg()->name,
                         matched.at("left_bn2_bias")->arg()->name,
                         matched.at("left_bn3_bias")->arg()->name,
                         matched.at("left_bn4_bias")->arg()->name,
                         matched.at("left_bn5_bias")->arg()->name,
                     });
    op_desc.SetInput("Mean",
                     {
                         matched.at("left_bn1_mean")->arg()->name,
                         matched.at("left_bn2_mean")->arg()->name,
                         matched.at("left_bn3_mean")->arg()->name,
                         matched.at("left_bn4_mean")->arg()->name,
                         matched.at("left_bn5_mean")->arg()->name,
                     });
    op_desc.SetInput("Var",
                     {
                         matched.at("left_bn1_variance")->arg()->name,
                         matched.at("left_bn2_variance")->arg()->name,
                         matched.at("left_bn3_variance")->arg()->name,
                         matched.at("left_bn4_variance")->arg()->name,
                         matched.at("left_bn5_variance")->arg()->name,
                     });

    op_desc.SetOutput("Outputs", {matched.at("relu_out")->arg()->name});
    // keep these to fool SubgraphOp::AttachImpl()
    op_desc.SetAttr<int>("sub_block", 0);
    op_desc.SetAttr<std::vector<std::string>>("input_data_names", {});
    op_desc.SetAttr<std::vector<std::string>>("output_data_names", {});

    auto block0_stmt = matched.at("left_conv1")->stmt();
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
        "left_conv5_weight",
        "left_bn1_bias",
        "left_bn2_bias",
        "left_bn3_bias",
        "left_bn4_bias",
        "left_bn5_bias",
    };
    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), matched.at("left_conv1"));
    }
    IR_OP_VAR_LINK(matched.at("left_conv1"), matched.at("relu_out"));
  }
};

class XPURes2Net50Fuser : public xpu::XPUFuseBase {
 public:
  XPURes2Net50Fuser() {}

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
                             ->assert_is_op_input("conv2d", "Input")
                             ->AsIntermediate();

    auto* second_conv_weight = VarNode("second_conv_weight")
                                   ->assert_is_op_input("conv2d", "Filter")
                                   ->AsInput();
    auto* second_conv = OpNode("second_conv", "conv2d")->AsIntermediate();
    auto* second_conv_out = VarNode("second_conv_out")
                                ->assert_is_op_output("conv2d", "Output")
                                ->assert_is_op_input("batch_norm", "X")
                                ->AsIntermediate();
    auto* second_bn_scale = VarNode("second_bn_scale")
                                ->assert_is_op_input("batch_norm", "Scale")
                                ->AsIntermediate();
    auto* second_bn_bias = VarNode("second_bn_bias")
                               ->assert_is_op_input("batch_norm", "Bias")
                               ->AsInput();
    auto* second_bn_mean = VarNode("second_bn_mean")
                               ->assert_is_op_input("batch_norm", "Mean")
                               ->AsIntermediate();
    auto* second_bn_var = VarNode("second_bn_variance")
                              ->assert_is_op_input("batch_norm", "Variance")
                              ->AsIntermediate();
    auto* second_bn = OpNode("second_bn", "batch_norm")->AsIntermediate();
    auto* second_bn_out = VarNode("second_bn_out")
                              ->assert_is_op_output("batch_norm", "Y")
                              ->assert_is_op_input("relu", "X")
                              ->AsIntermediate();
    auto* second_bn_mean_out =
        VarNode("second_bn_mean_out")
            ->assert_is_op_output("batch_norm", "MeanOut")
            ->AsIntermediate();
    auto* second_bn_var_out =
        VarNode("second_bn_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* second_bn_saved_mean =
        VarNode("second_bn_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* second_bn_saved_var =
        VarNode("second_bn_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();
    auto* second_relu = OpNode("second_relu", "relu")->AsIntermediate();
    auto* second_relu_out = VarNode("second_relu_out")
                                ->assert_is_op_output("relu", "Out")
                                ->assert_is_op_input("conv2d", "Input")
                                ->AsIntermediate();

    auto* third_conv_weight = VarNode("third_conv_weight")
                                  ->assert_is_op_input("conv2d", "Filter")
                                  ->AsInput();
    auto* third_conv = OpNode("third_conv", "conv2d")->AsIntermediate();
    auto* third_conv_out = VarNode("third_conv_out")
                               ->assert_is_op_output("conv2d", "Output")
                               ->assert_is_op_input("batch_norm", "X")
                               ->AsIntermediate();
    auto* third_bn_scale = VarNode("third_bn_scale")
                               ->assert_is_op_input("batch_norm", "Scale")
                               ->AsIntermediate();
    auto* third_bn_bias = VarNode("third_bn_bias")
                              ->assert_is_op_input("batch_norm", "Bias")
                              ->AsInput();
    auto* third_bn_mean = VarNode("third_bn_mean")
                              ->assert_is_op_input("batch_norm", "Mean")
                              ->AsIntermediate();
    auto* third_bn_var = VarNode("third_bn_variance")
                             ->assert_is_op_input("batch_norm", "Variance")
                             ->AsIntermediate();
    auto* third_bn = OpNode("third_bn", "batch_norm")->AsIntermediate();
    auto* third_bn_out = VarNode("third_bn_out")
                             ->assert_is_op_output("batch_norm", "Y")
                             ->assert_is_op_input("relu", "X")
                             ->AsIntermediate();
    auto* third_bn_mean_out = VarNode("third_bn_mean_out")
                                  ->assert_is_op_output("batch_norm", "MeanOut")
                                  ->AsIntermediate();
    auto* third_bn_var_out =
        VarNode("third_bn_var_out")
            ->assert_is_op_output("batch_norm", "VarianceOut")
            ->AsIntermediate();
    auto* third_bn_saved_mean =
        VarNode("third_bn_saved_mean")
            ->assert_is_op_output("batch_norm", "SavedMean")
            ->AsIntermediate();
    auto* third_bn_saved_var =
        VarNode("third_bn_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();
    auto* third_relu = OpNode("third_relu", "relu")->AsIntermediate();
    auto* third_relu_out = VarNode("third_relu_out")
                               ->assert_is_op_output("relu", "Out")
                               ->assert_is_op_input("pool2d", "X")
                               ->AsIntermediate();

    auto* top_pool = OpNode("top_pool", "pool2d")->AsIntermediate();
    auto* top_pool_out = VarNode("top_pool_out")
                             ->assert_is_op_output("pool2d", "Out")
                             ->assert_is_op_input("res2net_block0", "Inputs")
                             ->AsIntermediate();
    auto* resnet_block0_1 =
        OpNode("resnet_block0_1", "res2net_block0")->AsIntermediate();
    auto* resnet_block0_1_out =
        VarNode("resnet_block0_1_out")
            ->assert_is_op_output("res2net_block0", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_1_1 =
        OpNode("resnet_block1_1_1", "res2net_block2")->AsIntermediate();
    auto* resnet_block1_1_1_out =
        VarNode("resnet_block1_1_1_out")
            ->assert_is_op_output("res2net_block2", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_1_2 =
        OpNode("resnet_block1_1_2", "res2net_block2")->AsIntermediate();
    auto* resnet_block1_1_2_out =
        VarNode("resnet_block1_1_2_out")
            ->assert_is_op_output("res2net_block2", "Outputs")
            ->AsIntermediate();
    auto* resnet_block0_2 =
        OpNode("resnet_block0_2", "res2net_block1")->AsIntermediate();
    auto* resnet_block0_2_out =
        VarNode("resnet_block0_2_out")
            ->assert_is_op_output("res2net_block1", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_2_1 =
        OpNode("resnet_block1_2_1", "res2net_block2")->AsIntermediate();
    auto* resnet_block1_2_1_out =
        VarNode("resnet_block1_2_1_out")
            ->assert_is_op_output("res2net_block2", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_2_2 =
        OpNode("resnet_block1_2_2", "res2net_block2")->AsIntermediate();
    auto* resnet_block1_2_2_out =
        VarNode("resnet_block1_2_2_out")
            ->assert_is_op_output("res2net_block2", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_2_3 =
        OpNode("resnet_block1_2_3", "res2net_block2")->AsIntermediate();
    auto* resnet_block1_2_3_out =
        VarNode("resnet_block1_2_3_out")
            ->assert_is_op_output("res2net_block2", "Outputs")
            ->AsIntermediate();
    auto* resnet_block0_3 =
        OpNode("resnet_block0_3", "res2net_block1")->AsIntermediate();
    auto* resnet_block0_3_out =
        VarNode("resnet_block0_3_out")
            ->assert_is_op_output("res2net_block1", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_3_1 =
        OpNode("resnet_block1_3_1", "res2net_block2")->AsIntermediate();
    auto* resnet_block1_3_1_out =
        VarNode("resnet_block1_3_1_out")
            ->assert_is_op_output("res2net_block2", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_3_2 =
        OpNode("resnet_block1_3_2", "res2net_block2")->AsIntermediate();
    auto* resnet_block1_3_2_out =
        VarNode("resnet_block1_3_2_out")
            ->assert_is_op_output("res2net_block2", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_3_3 =
        OpNode("resnet_block1_3_3", "res2net_block2")->AsIntermediate();
    auto* resnet_block1_3_3_out =
        VarNode("resnet_block1_3_3_out")
            ->assert_is_op_output("res2net_block2", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_3_4 =
        OpNode("resnet_block1_3_4", "res2net_block2")->AsIntermediate();
    auto* resnet_block1_3_4_out =
        VarNode("resnet_block1_3_4_out")
            ->assert_is_op_output("res2net_block2", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_3_5 =
        OpNode("resnet_block1_3_5", "res2net_block2")->AsIntermediate();
    auto* resnet_block1_3_5_out =
        VarNode("resnet_block1_3_5_out")
            ->assert_is_op_output("res2net_block2", "Outputs")
            ->AsIntermediate();
    auto* resnet_block0_4 =
        OpNode("resnet_block0_4", "res2net_block1")->AsIntermediate();
    auto* resnet_block0_4_out =
        VarNode("resnet_block0_4_out")
            ->assert_is_op_output("res2net_block1", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_4_1 =
        OpNode("resnet_block1_4_1", "res2net_block2")->AsIntermediate();
    auto* resnet_block1_4_1_out =
        VarNode("resnet_block1_4_1_out")
            ->assert_is_op_output("res2net_block2", "Outputs")
            ->AsIntermediate();
    auto* resnet_block1_4_2 =
        OpNode("resnet_block1_4_2", "res2net_block2")->AsIntermediate();
    auto* resnet_block1_4_2_out =
        VarNode("resnet_block1_4_2_out")
            ->assert_is_op_output("res2net_block2", "Outputs")
            ->AsIntermediate();
    auto* bottom_pool = OpNode("bottom_pool", "pool2d")->AsIntermediate();
    auto* bottom_pool_out = VarNode("bottom_pool_out")
                                ->assert_is_op_output("pool2d", "Out")
                                ->AsOutput();

    *input >> *top_conv >> *top_conv_out >> *top_bn >> *top_bn_out >>
        *top_relu >> *top_relu_out >> *second_conv >> *second_conv_out >>
        *second_bn >> *second_bn_out >> *second_relu >> *second_relu_out >>
        *third_conv >> *third_conv_out >> *third_bn >> *third_bn_out >>
        *third_relu >> *third_relu_out >> *top_pool >> *top_pool_out >>
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
        *resnet_block1_4_2_out >> *bottom_pool >> *bottom_pool_out;

    *top_conv_weight >> *top_conv;
    *top_bn_scale >> *top_bn;
    *top_bn_bias >> *top_bn;
    *top_bn_mean >> *top_bn;
    *top_bn_var >> *top_bn;
    *top_bn >> *top_bn_mean_out;
    *top_bn >> *top_bn_var_out;
    *top_bn >> *top_bn_saved_mean;
    *top_bn >> *top_bn_saved_var;

    *second_conv_weight >> *second_conv;
    *second_bn_scale >> *second_bn;
    *second_bn_bias >> *second_bn;
    *second_bn_mean >> *second_bn;
    *second_bn_var >> *second_bn;
    *second_bn >> *second_bn_mean_out;
    *second_bn >> *second_bn_var_out;
    *second_bn >> *second_bn_saved_mean;
    *second_bn >> *second_bn_saved_var;

    *third_conv_weight >> *third_conv;
    *third_bn_scale >> *third_bn;
    *third_bn_bias >> *third_bn;
    *third_bn_mean >> *third_bn;
    *third_bn_var >> *third_bn;
    *third_bn >> *third_bn_mean_out;
    *third_bn >> *third_bn_var_out;
    *third_bn >> *third_bn_saved_mean;
    *third_bn >> *third_bn_saved_var;
  }

  void InsertNewNode(SSAGraph* graph,
                     const key2nodes_t& matched,
                     const std::vector<Node*>& extra_input_vars) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__res2net50");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    std::vector<std::string> filter_name = {
        matched.at("top_conv_weight")->arg()->name,
        matched.at("second_conv_weight")->arg()->name,
        matched.at("third_conv_weight")->arg()->name};

    std::vector<std::string> scale_name = {
        matched.at("top_bn_scale")->arg()->name,
        matched.at("second_bn_scale")->arg()->name,
        matched.at("third_bn_scale")->arg()->name};

    std::vector<std::string> bias_name = {
        matched.at("top_bn_bias")->arg()->name,
        matched.at("second_bn_bias")->arg()->name,
        matched.at("third_bn_bias")->arg()->name};

    std::vector<std::string> mean_name = {
        matched.at("top_bn_mean")->arg()->name,
        matched.at("second_bn_mean")->arg()->name,
        matched.at("third_bn_mean")->arg()->name};

    std::vector<std::string> var_name = {
        matched.at("top_bn_variance")->arg()->name,
        matched.at("second_bn_variance")->arg()->name,
        matched.at("third_bn_variance")->arg()->name};

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
    op_desc.SetInput("Filter", filter_name);
    op_desc.SetInput("Bias", bias_name);
    op_desc.SetOutput("Output", {matched.at("bottom_pool_out")->arg()->name});
    op_desc.SetAttr<int>("xpu", 1);

    auto* resnet50_stmt = matched.at("top_conv")->stmt();
    auto* scope = resnet50_stmt->op()->scope();
    for (size_t i = 0; i < filter_name.size(); ++i) {
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
    op_desc.SetInput("MaxFilter", max_filter_name);
    auto resnet50_op = LiteOpRegistry::Global().Create(op_desc.Type());
    resnet50_op->Attach(op_desc, scope);
    resnet50_op->SetValidPlaces(resnet50_stmt->op()->valid_places());
    auto kernels = resnet50_op->CreateKernels(resnet50_op->valid_places());
    resnet50_stmt->SetOp(resnet50_op);
    resnet50_stmt->SetKernels(std::move(kernels));

    IR_NODE_LINK_TO(matched.at("top_bn_bias"), matched.at("top_conv"));
    IR_NODE_LINK_TO(matched.at("second_conv_weight"), matched.at("top_conv"));
    IR_NODE_LINK_TO(matched.at("second_bn_bias"), matched.at("top_conv"));
    IR_NODE_LINK_TO(matched.at("third_conv_weight"), matched.at("top_conv"));
    IR_NODE_LINK_TO(matched.at("third_bn_bias"), matched.at("top_conv"));

    for (auto* node : extra_input_vars) {
      IR_NODE_LINK_TO(node, matched.at("top_conv"));
    }
    IR_OP_VAR_LINK(matched.at("top_conv"), matched.at("bottom_pool_out"));
  }
};

}  // namespace fusion

class XPURes2NetFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    bool changed = false;
    SSAGraph backup;
    backup.CloneFrom(*graph);

    fusion::XPURes2NetBlock0Fuser block0_fuser;
    changed |= block0_fuser(graph.get());

    fusion::XPURes2NetBlock1Fuser block1_fuser;
    changed |= block1_fuser(graph.get());

    fusion::XPURes2NetBlock2Fuser block2_fuser;
    changed |= block2_fuser(graph.get());

    fusion::XPURes2Net50Fuser res2net50_fuser;
    size_t n_matches = res2net50_fuser(graph.get());
    if (changed && !n_matches) {
      // Restore graph from backuped one if no whole ResNet50 graph was found
      graph->CloneFrom(backup);
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__res2net_fuse_pass,
                  paddle::lite::mir::XPURes2NetFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__res2net50");
