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
  // Create op
  auto* conv =
      OpNode("conv2d", conv_type_)->assert_is_op(conv_type_)->AsIntermediate();
  auto* bn =
      OpNode("bn", "batch_norm")->assert_is_op("batch_norm")->AsIntermediate();

  // Create input
  auto* conv_input =
      VarNode("conv_input")->assert_is_op_input(conv_type_, "Input")->AsInput();
  auto* conv_weight = VarNode("conv_weight")
                          ->assert_is_op_input(conv_type_, "Filter")
                          ->AsInput();
  auto* bn_bias = VarNode("bn_bias")
                      ->assert_is_op_input("batch_norm", "Bias")
                      ->AsInput()
                      ->assert_is_persistable_var();

  // Create intermediate
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
  auto* bn_scale = VarNode("bn_scale")
                       ->assert_is_op_input("batch_norm", "Scale")
                       ->AsIntermediate();
  auto* bn_mean = VarNode("bn_mean")
                      ->assert_is_op_input("batch_norm", "Mean")
                      ->AsIntermediate();
  auto* bn_var = VarNode("bn_variance")
                     ->assert_is_op_input("batch_norm", "Variance")
                     ->AsIntermediate();

  auto* conv_out = VarNode("conv_out")
                       ->assert_is_op_output(conv_type_, "Output")
                       ->assert_is_op_input("batch_norm", "X")
                       ->AsIntermediate();
  // Create output
  auto* bn_out =
      VarNode("bn_out")->assert_is_op_output("batch_norm", "Y")->AsOutput();

  // Because `conv_has_bias_` term is not sure existed or not,
  //   the process of old `conv_bias` term deletion is difficult,
  //   define two patterns, to avoid seek and delete old conv_bias term.
  //
  // Store new `conv_bias` term in `bn_bias` term,
  //   because `bn_bias` term is sure existed.
  if (false == conv_has_bias_) {
    conv->LinksFrom({conv_input, conv_weight}).LinksTo({conv_out});
  } else if (true == conv_has_bias_) {
    // conv_op_desc->HasAttr("enable_int8")
    auto* conv_bias = VarNode("conv_bias")
                          ->assert_is_op_input(conv_type_, "Bias")
                          ->AsIntermediate();
    conv->LinksFrom({conv_input, conv_weight, conv_bias}).LinksTo({conv_out});
  } else {  // conv_has_bias(bool) unsupported value
    LOG(FATAL) << "conv_has_bias_(bool) is invalid value";
  }
  bn->LinksFrom({conv_out, bn_scale, bn_bias, bn_mean, bn_var})
      .LinksTo({bn_out, bn_mean_out, bn_saved_mean, bn_saved_var, bn_var_out});
}

void ConvBNFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto new_conv_op = LiteOpRegistry::Global().Create("conv2d");
  auto conv_instruct = matched.at("conv2d")->stmt();
  auto conv_op_desc = conv_instruct->op();
  auto* scope = conv_op_desc->scope();
  auto& valid_places = conv_op_desc->valid_places();

  // bn
  auto bn_scale_t = scope->FindVar(matched.at("bn_scale")->arg()->name)
                        ->GetMutable<lite::Tensor>();
  auto bn_scale_d = bn_scale_t->mutable_data<float>();
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

  // conv
  auto conv_weight_t = scope->FindVar(matched.at("conv_weight")->arg()->name)
                           ->GetMutable<lite::Tensor>();
  CHECK_EQ(static_cast<size_t>(bn_scale_t->data_size()),
           static_cast<size_t>(conv_weight_t->dims()[0]))
      << "The BN bias's size should be equal to the size of the first "
      << "dim size of the conv weights";
  size_t weight_num = conv_weight_t->data_size();

  // comupte BN alpha and beta
  Tensor alpha_tensor, beta_tensor;
  alpha_tensor.CopyDataFrom(*bn_bias_t);
  beta_tensor.CopyDataFrom(*bn_bias_t);
  auto alpha_data = alpha_tensor.mutable_data<float>();
  auto beta_data = beta_tensor.mutable_data<float>();

  int h =
      bn_scale_t
          ->data_size();  // h == bias_size == out channel num of conv weight
  int w = weight_num /
          (bn_scale_t->data_size());  // w = `conv_weight_num` / bias_size = in
                                      // channel num of conv weight

  ComputeAlphaAndBeta(
      bn_scale_d, bn_mean_d, bn_var_d, alpha_data, beta_data, eps, h, w);

  // compute new new weight and bias
  if (conv_op_desc->HasAttr("enable_int8") == true) {
    VLOG(4) << "enable_int8 branch: enable_int8 is true";
    PADDLE_ENFORCE(conv_op_desc->HasAttr("weight_scale"),
                   "INT8 mode: Conv should has weight_scale attr");
    auto weight_scale =
        conv_op_desc->GetAttr<std::vector<float>>("weight_scale");
    for (unsigned int i = 0; i < h; i++) {
      weight_scale[i] *= alpha_data[i];
    }

    // if conv_bias existed, keep value and store in bn_bias
    if (op_desc.HasInput("Bias") && op_desc.Input("Bias").size() > 0) {
      auto bias_var = scope->FindVar(op_desc.Input("Bias").front());
      if (bias_var != nullptr) {
        auto old_conv_bias_t = &(bias_var->Get<lite::Tensor>());
        bn_bias_t.CopyDataFrom(*old_conv_bias_t);
      }
      op_desc.SetInput("Bias",
                       {matched.at("bn_bias")->arg()->name});  // add Bias flag
    }
    // Interface like this should be abandoned.
    conv_op_desc->SetAttr("weight_scale", weight_scale);
    auto update_conv_desc = *conv_instruct->mutable_op_info();
    conv_instruct->ResetOp(update_conv_desc, graph->valid_places());
  } else {
    ///////////////////////////////////////////////////////////////////////////////
    // Compute ConvBNFuser
    //
    //   conv(x) = conv(x) = kx + z = y
    //   bn(y) = ay + b
    //
    // After fusion:
    //
    //   bn(conv(x)) = a(kx + z) + b = akx + az + b
    //
    // Note: h == bias_size == out channel num of conv weight
    //       w = `conv_weight_num` / bias_size = in channel num of conv weight
    ///////////////////////////////////////////////////////////////////////////////
    VLOG(4) << "enable_int8 branch: enable_int8 is false";
    // initialize conv bias value
    Tensor new_conv_bias_tensor;
    new_conv_bias_tensor.Resize(bn_bias_t->dims());
    auto new_conv_bias_d = new_conv_bias_tensor.mutable_data<float>();
    if (op_desc.HasInput("Bias") && op_desc.Input("Bias").size() > 0) {
      auto bias_var = scope->FindVar(op_desc.Input("Bias").front());
      if (bias_var != nullptr) {
        auto old_conv_bias_t = &(bias_var->Get<lite::Tensor>());
        new_conv_bias_tensor.CopyDataFrom(*old_conv_bias_t);
      }
    }

    // compute new conv_weight
    auto conv_weight_d = conv_weight_t->mutable_data<float>();
    for (unsigned int i = 0; i < h; i++) {    // n: conv2d output channels
      for (unsigned int j = 0; j < w; j++) {  // w: conv2d input channels
        conv_weight_d[i * w + j] *= alpha_data[i];
      }
    }

    // compute new conv_bias
    for (unsigned int i = 0; i < bn_scale_t->data_size();
         i++) {  // bias_size == h == conv2d output channls
      new_conv_bias_d[i] =
          alpha_data[i] * new_conv_bias_d[i] + (bn_bias_d[i] + beta_data[i]);
    }

    // store conv_bias_d to `bn_bias` arg node
    bn_bias_t->CopyDataFrom(new_conv_bias_tensor);
    op_desc.SetInput("Bias",
                     {matched.at("bn_bias")->arg()->name});  // add Bias flag
  }

  new_conv_op->Attach(op_desc, scope);
  auto* new_op_node = graph->GraphCreateInstructNode(new_conv_op, valid_places);

  IR_NODE_LINK_TO(matched.at("conv_input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("conv_weight"), new_op_node);
  // consider special case: enable_int8 == true, without conv_bias
  // 4 cases for conv bn fuse pass
  // enable_int8=true, with conv_bias: input need bn_bias
  // enable_int8=true, without conv_bias: input don't need bn_bias
  // enable_int8=false, with conv_bias: input need bn_bias
  // enable_int8=false, without conv_bias: input need bn_bias
  if (enable_int8 == true && (op_desc.HasInput("Bias") == false)) {
    // enable_int8=true, without conv_bias: input don't need bn_bias
  } else {
    IR_NODE_LINK_TO(matched.at("bn_bias"), new_op_node);
  }
  IR_NODE_LINK_TO(new_op_node, matched.at("bn_out"));
}

cpp::OpDesc ConvBNFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc = *matched.at("conv2d")->stmt()->op_info();

  op_desc.SetType(conv_type_);
  op_desc.SetInput("Input", {matched.at("conv_input")->arg()->name});
  op_desc.SetInput("Filter", {matched.at("conv_weight")->arg()->name});
  // `SetInput` `conv_bias` term added in `InsertNewNode` function
  op_desc.SetOutput("Output", {matched.at("bn_out")->arg()->name});

  // Only consider strides, padding, groups, dilations for now
  op_desc.SetAttr("strides", op_desc.GetAttr<std::vector<int>>("strides"));
  op_desc.SetAttr("paddings", op_desc.GetAttr<std::vector<int>>("paddings"));
  op_desc.SetAttr("groups", op_desc.GetAttr<int>("groups"));
  op_desc.SetAttr("dilations", op_desc.GetAttr<std::vector<int>>("dilations"));

  // other params
  std::vector<std::string> input_arg_names = op_desc.InputArgumentNames();
  if (std::find(input_arg_names.begin(),
                input_arg_names.end(),
                "ResidualData") != input_arg_names.end()) {
    op_desc.SetInput("ResidualData", op_desc.Input("ResidualData"));
  }

  // For Int8
  if (op_desc.HasAttr("enable_int8")) {
    op_desc.SetAttr("enable_int8", op_desc.GetAttr<bool>("enable_int8"));
    if (op_desc.HasAttr("input_scale"))
      op_desc.SetAttr("input_scale", op_desc.GetAttr<float>("input_scale"));
    if (op_desc.HasAttr("weight_scale"))
      op_desc.SetAttr("weight_scale",
                      op_desc.GetAttr<std::vector<float>>("weight_scale"));
    if (op_desc.HasAttr("output_scale")) {
      op_desc.SetAttr("output_scale", op_desc.GetAttr<float>("output_scale"));
    }
  }

  // For with_act: ignored, because conv-act fuser pass is behind this pass
  // Other inputs. See operators/conv_op.h

  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
