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

#include "lite/core/mir/fusion/conv_bn_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ConvBNFuser::BuildPattern() {
  auto* conv_input =
      VarNode("conv_input")->assert_is_op_input(conv_type_, "Input")->AsInput();
  auto* conv_weight = VarNode("conv_weight")
                          ->assert_is_op_input(conv_type_, "Filter")
                          ->AsInput();
  auto* conv = OpNode("conv2d", conv_type_)->assert_is_op(conv_type_);
  auto* conv_out = VarNode("conv_out")
                       ->assert_is_op_output(conv_type_, "Output")
                       ->assert_is_op_input("batch_norm", "X")
                       ->AsIntermediate();

  auto* bn_scale = VarNode("bn_scale")
                       ->assert_is_op_input("batch_norm", "Scale")
                       ->AsIntermediate();
  auto* bn_bias =
      VarNode("bn_bias")->assert_is_op_input("batch_norm", "Bias")->AsInput();
  auto* bn_mean = VarNode("bn_mean")
                      ->assert_is_op_input("batch_norm", "Mean")
                      ->AsIntermediate();
  auto* bn_var = VarNode("bn_variance")
                     ->assert_is_op_input("batch_norm", "Variance")
                     ->AsIntermediate();
  auto* bn =
      OpNode("bn", "batch_norm")->assert_is_op("batch_norm")->AsIntermediate();

  auto* bn_out =
      VarNode("bn_out")->assert_is_op_output("batch_norm", "Y")->AsOutput();
  auto* bn_mean_out = VarNode("bn_mean_out")
                          ->assert_is_op_output("batch_norm", "MeanOut")
                          ->AsIntermediate();
  auto* bn_var_out = VarNode("bn_var_out")
                         ->assert_is_op_output("batch_norm", "VarianceOut")
                         ->AsIntermediate();
  auto* bn_saved_mean = VarNode("bn_saved_mean")
                            ->assert_is_op_output("batch_norm", "SavedMean")
                            ->AsIntermediate();
  auto* bn_saved_var = VarNode("bn_saved_var")
                           ->assert_is_op_output("batch_norm", "SavedVariance")
                           ->AsIntermediate();

  if (has_bias_) {
    auto* conv_bias = VarNode("conv_bias")
                          ->assert_is_op_input(conv_type_, "Bias")
                          ->AsInput()
                          ->AsIntermediate();
    conv->LinksFrom({conv_input, conv_weight, conv_bias}).LinksTo({conv_out});
  } else {
    conv->LinksFrom({conv_input, conv_weight}).LinksTo({conv_out});
  }

  bn->LinksFrom({conv_out, bn_scale, bn_bias, bn_mean, bn_var})
      .LinksTo({bn_out, bn_mean_out, bn_saved_mean, bn_saved_var, bn_var_out});
}

void ConvBNFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  auto conv_instruct = matched.at("conv2d")->stmt();
  auto conv = conv_instruct->op();
  auto* scope = conv->scope();

  auto conv_weight_t = scope->FindVar(matched.at("conv_weight")->arg()->name)
                           ->GetMutable<lite::Tensor>();
  auto conv_weight_dims = conv_weight_t->dims();
  size_t weight_num = conv_weight_t->data_size();

  auto bn_scale_t = scope->FindVar(matched.at("bn_scale")->arg()->name)
                        ->GetMutable<lite::Tensor>();
  size_t bias_size = bn_scale_t->data_size();
  auto bn_scale_d = bn_scale_t->mutable_data<float>();
  CHECK_EQ(bias_size, static_cast<size_t>(conv_weight_dims[0]))
      << "The BN bias's size should be equal to the size of the first "
      << "dim size of the conv weights";

  auto bn_mean_t = scope->FindVar(matched.at("bn_mean")->arg()->name)
                       ->GetMutable<lite::Tensor>();
  auto bn_mean_d = bn_mean_t->mutable_data<float>();

  auto bn_var_t = scope->FindVar(matched.at("bn_variance")->arg()->name)
                      ->GetMutable<lite::Tensor>();
  auto bn_var_d = bn_var_t->mutable_data<float>();

  auto bn_bias_t = scope->FindVar(matched.at("bn_bias")->arg()->name)
                       ->GetMutable<lite::Tensor>();
  auto bn_bias_d = bn_bias_t->mutable_data<float>();
  auto eps = matched.at("bn")->stmt()->op_info()->GetAttr<float>("epsilon");

  auto conv_op_desc = conv_instruct->mutable_op_info();

  bool enable_int8 = conv_op_desc->HasAttr("enable_int8") ? true : false;
  Tensor alpha_tensor, beta_tensor;
  alpha_tensor.CopyDataFrom(*bn_bias_t);
  beta_tensor.CopyDataFrom(*bn_bias_t);
  auto alpha_data = alpha_tensor.mutable_data<float>();
  auto beta_data = beta_tensor.mutable_data<float>();

  int h = bias_size;
  int w = weight_num / bias_size;
  ComputeAlphaAndBeta(
      bn_scale_d, bn_mean_d, bn_var_d, alpha_data, beta_data, eps, h, w);

  if (enable_int8) {
    PADDLE_ENFORCE(conv_op_desc->HasAttr("weight_scale"),
                   "INT8 mode: Conv should has weight_scale attr");
    auto weight_scale =
        conv_op_desc->GetAttr<std::vector<float>>("weight_scale");
    auto conv_weight_d = conv_weight_t->mutable_data<int8_t>();
    for (int i = 0; i < h; i++) {
      weight_scale[i] *= fabsf(alpha_data[i]);
      if (alpha_data[i] < 0.f) {
        auto ptr_row = conv_weight_d + i * w;
        for (int j = 0; j < w; j++) {
          ptr_row[j] *= -1;
        }
      }
    }
    // Interface like this should be abandoned.
    conv_op_desc->SetAttr("weight_scale", weight_scale);
  } else {
    auto conv_weight_d = conv_weight_t->mutable_data<float>();
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        conv_weight_d[i * w + j] *= alpha_data[i];
      }
    }
  }
  for (int i = 0; i < bias_size; i++) {
    bn_bias_d[i] += beta_data[i];
  }
  if (has_bias_) {
    auto conv_bias_var_t = scope->FindVar(matched.at("conv_bias")->arg()->name)
                               ->GetMutable<lite::Tensor>();
    auto conv_bias_var_d = conv_bias_var_t->mutable_data<float>();
    for (int i = 0; i < bias_size; i++) {
      bn_bias_d[i] += conv_bias_var_d[i] * alpha_data[i];
    }
  }
  conv_op_desc->SetType(conv_type_);
  conv_op_desc->SetInput("Input", {matched.at("conv_input")->arg()->name});
  conv_op_desc->SetInput("Filter", {matched.at("conv_weight")->arg()->name});
  conv_op_desc->SetOutput("Output", {matched.at("bn_out")->arg()->name});
  conv_op_desc->SetInput("Bias", {matched.at("bn_bias")->arg()->name});
  auto update_conv_desc = *conv_instruct->mutable_op_info();
  conv_instruct->ResetOp(update_conv_desc, graph->valid_places());

  IR_NODE_LINK_TO(matched.at("bn_bias"), matched.at("conv2d"));
  IR_OP_VAR_LINK(matched.at("conv2d"), matched.at("bn_out"));
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
