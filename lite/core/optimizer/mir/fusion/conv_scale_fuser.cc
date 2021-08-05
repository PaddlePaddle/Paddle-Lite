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

#include "lite/core/optimizer/mir/fusion/conv_scale_fuser.h"
#include <memory>
#include <set>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ConvScaleFuser::BuildPattern() {
  // create input nodes
  auto* conv_input =
      VarNode("conv_input")->assert_is_op_input(conv_type_, "Input")->AsInput();
  auto* conv_weight = VarNode("conv_weight")
                          ->assert_is_op_input(conv_type_, "Filter")
                          ->AsInput();

  // create op nodes
  auto* conv = OpNode("conv2d", conv_type_)->assert_is_op(conv_type_);

  auto* scale =
      OpNode("scale", "scale")->assert_is_op("scale")->AsIntermediate();

  // create intermediate nodes
  auto* conv_out = VarNode("conv_out")
                       ->assert_is_op_output(conv_type_, "Output")
                       ->assert_is_op_input("scale", "X")
                       ->AsIntermediate();

  // create output node
  auto* out =
      VarNode("scale_out")->assert_is_op_output("scale", "Out")->AsOutput();

  // create topology.
  if (conv_has_bias_) {
    auto* conv_bias = VarNode("conv_bias")
                          ->assert_is_op_input(conv_type_, "Bias")
                          ->AsIntermediate();
    conv->LinksFrom({conv_input, conv_weight, conv_bias}).LinksTo({conv_out});
  } else {
    LOG(FATAL) << "Unsupported for Conv without bias";
    conv->LinksFrom({conv_input, conv_weight}).LinksTo({conv_out});
  }

  scale->LinksFrom({conv_out}).LinksTo({out});
}

void ConvScaleFuser::InsertNewNode(SSAGraph* graph,
                                   const key2nodes_t& matched) {
  auto conv_instruct = matched.at("conv2d")->stmt();
  auto conv_op_desc = conv_instruct->mutable_op_info();
  auto conv = conv_instruct->op();
  auto* scope = conv->scope();

  // conv
  std::string conv_weight_name = matched.at("conv_weight")->arg()->name;
  auto conv_weight_t =
      scope->FindVar(conv_weight_name)->GetMutable<lite::Tensor>();
  bool enable_int8 = conv_op_desc->HasAttr("enable_int8") ? true : false;
  bool is_int8 =
      enable_int8 ? conv_op_desc->GetAttr<bool>("enable_int8") : false;
  if (is_int8) {
    LOG(FATAL) << "Unsupported for int8 model.";
  }

  // scale
  auto* scale_op_desc = matched.at("scale")->stmt()->op_info();
  float scale_val = 1.f;
  float bias_val = 0.f;
  float alpha_val = 6.f;
  if (scale_op_desc->HasAttr("scale")) {
    scale_val = scale_op_desc->GetAttr<float>("scale");
  }
  if (scale_op_desc->HasAttr("bias")) {
    bias_val = scale_op_desc->GetAttr<float>("bias");
  }
  if (scale_op_desc->HasAttr("alpha")) {
    alpha_val = scale_op_desc->GetAttr<float>("alpha");
  }
  std::string activation_type{""};
  if (scale_op_desc->HasAttr("activation_type")) {
    activation_type = scale_op_desc->GetAttr<std::string>("activation_type");
  }
  conv_op_desc->SetAttr<std::string>("scale_activation_type", activation_type);

  // compute new conv_weight
  auto conv_weight_d = conv_weight_t->mutable_data<float>();
  for (size_t i = 0; i < conv_weight_t->data_size(); i++) {
    conv_weight_d[i] *= scale_val;
  }

  // compute new conv_bias
  std::string conv_bias_var_name{matched.at("conv_bias")->arg()->name};
  if (conv_has_bias_ && conv_op_desc->HasInput("Bias") &&
      conv_op_desc->Input("Bias").size() > 0) {
    auto conv_bias_t =
        scope->FindVar(conv_bias_var_name)->GetMutable<lite::Tensor>();
    auto* conv_bias_d = conv_bias_t->mutable_data<float>();
    for (size_t i = 0; i < conv_bias_t->data_size(); i++) {
      conv_bias_d[i] = conv_bias_d[i] * scale_val + bias_val;
    }
    IR_NODE_LINK_TO(matched.at("conv_bias"), matched.at("conv2d"));
  } else {
    LOG(FATAL) << "Unsupported for Conv without bias";
    /*
    note: Uncomment codes below, because not ready for convert VarNode
          or Tensor to Node.
    auto* conv_bias_t = scope->NewTensor(conv_bias_var_name);
    CHECK(conv_weight_t->dims().size() == 4) << "The dimension of conv weight
    must be 4.";
    conv_bias_t->Resize(conv_weight_t->dims()[0]); // output channel
    auot conv_bias_d = conv_bias_t->data<float>();
    for (size_t i = 0; i < conv_bias_t->data_size(); ++i) {
      conv_bias_d[i] = bias_val;
    }
    // IR_NODE_LINK_TO(matched.at("conv_bias"), matched.at("conv2d"));
    */
  }

  conv_op_desc->SetType(conv_type_);
  conv_op_desc->SetInput("Input", {matched.at("conv_input")->arg()->name});
  conv_op_desc->SetInput("Filter", {matched.at("conv_weight")->arg()->name});
  conv_op_desc->SetOutput("Output", {matched.at("scale_out")->arg()->name});
  conv_op_desc->SetInput("Bias", {conv_bias_var_name});  // conv_bias
  auto update_conv_desc = *conv_instruct->mutable_op_info();
  conv_instruct->ResetOp(update_conv_desc, graph->valid_places());

  IR_OP_VAR_LINK(matched.at("conv2d"), matched.at("scale_out"));
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
