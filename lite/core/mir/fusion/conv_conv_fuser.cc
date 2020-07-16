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

#include "lite/core/mir/fusion/conv_conv_fuser.h"
#include <memory>
#include <set>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ConvConvFuser::BuildPattern() {
  auto* conv_input0 =
      VarNode("conv_input0")->assert_is_op_input(conv_type0_, "Input")->AsInput();
  auto* conv_weight0 = VarNode("conv_weight0")
                          ->assert_is_op_input(conv_type0_, "Filter")
                          ->AsInput();
  auto* conv0 = OpNode("conv2d0", conv_type0_)
                    ->assert_is_op(conv_type0_);
                    // ->assert_op_attr<int>("groups", 1);
  auto* conv_out0 = VarNode("conv_out0")
                       ->assert_is_op_output(conv_type0_, "Output")
                       ->assert_is_op_input(conv_type1_, "Input")
                       ->AsIntermediate();

  auto* conv_weight1 = VarNode("conv_weight1")
                          ->assert_is_op_input(conv_type1_, "Filter")
                          ->AsIntermediate();
  auto* conv1 = OpNode("conv2d1", conv_type1_)
                    ->assert_is_op(conv_type1_)
                    ->assert_op_attr<int>("groups", 1)
                    ->AsIntermediate();

  auto* conv_out1 =
      VarNode("conv_out1")->assert_is_op_output(conv_type1_, "Output")->AsOutput();

  // get  convlution info
  auto weight0 = conv_weight0->GetMutable<lite::Tensor>();
  auto weight1 = conv_weight1->GetMutable<lite::Tensor>();
  auto wei_dim0 = weight0->dims();
  auto wei_dim1 = weight1->dims();

  bool size = (wei_dim0.size() == 4) && (wei_dim0.size() == wei_dim1.size());
  bool ksize_eq = (wei_dim0[2] == wei_dim0[3]) && (wei_dim1[2] == wei_dim1[3]);
  bool ksize1 = (wei_dim0[2] = 1  || wei_dim0[2]==3);
  bool ksize2 = wei_dim1[2] == 1;
  //  only support 1x1/3x3 + 1x1
  if (!(size && ksize_eq && ksize1 && ksize2)){
      return;
  }

  if (conv_has_bias0_) {
    if (conv_has_bias1_) {
        auto* conv_bias0 = VarNode("conv_bias0")
                              ->assert_is_op_input(conv_type0_, "Bias")
                              ->AsIntermediate();
        auto* conv_bias1 = VarNode("conv_bias1")
                              ->assert_is_op_input(conv_type1_, "Bias")
                              ->AsInput();
        conv0->LinksFrom({conv_input0, conv_weight0, conv_bias0}).LinksTo({conv_out0});
        conv1->LinksFrom({conv_out0, conv_weight1, conv_bias1}).LinksTo({conv_out1});
    } else {
        auto* conv_bias0 = VarNode("conv_bias0")
                              ->assert_is_op_input(conv_type0_, "Bias")
                              ->AsInput();
        conv0->LinksFrom({conv_input0, conv_weight0, conv_bias0}).LinksTo({conv_out0});
        conv1->LinksFrom({conv_out0, conv_weight1}).LinksTo({conv_out1});
    }
  } else {
    conv0->LinksFrom({conv_input0, conv_weight0}).LinksTo({conv_out0});
    if (conv_has_bias1_) {
        auto* conv_bias1 = VarNode("conv_bias1")
                              ->assert_is_op_input(conv_type1_, "Bias")
                              ->AsInput();
        conv1->LinksFrom({conv_out0, conv_weight1, conv_bias1}).LinksTo({conv_out1});
    } else {
        conv1->LinksFrom({conv_out0, conv_weight1}).LinksTo({conv_out1});
    }
  }
}

void ConvConvFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  auto conv_instruct = matched.at("conv2d0")->stmt();
  auto conv_op_desc = conv_instruct->mutable_op_info();
  auto conv = conv_instruct->op();
  auto* scope = conv->scope();
  auto conv_op_desc1 = matched.at("conv2d1")->stmt()->mutable_op_info();

  // conv0
  auto weight0_t = scope->FindVar(matched.at("conv_weight0")->arg()->name)
                        ->GetMutable<lite::Tensor>();
  auto weight0_d = weight0_t->mutable_data<float>();

  // conv1
  auto weight1_t = scope->FindVar(matched.at("conv_weight1")->arg()->name)
                        ->GetMutable<lite::Tensor>();
  auto weight1_d = weight1_t->mutable_data<float>();
  auto groups0 = conv_op_desc->GetAttr<int>("groups");
  auto groups1 = conv_op_desc1->GetAttr<int>("groups");
  auto strides1 = conv_op_desc1->GetAttr<std::vector<int>>("strides");
  auto paddings1 = conv_op_desc1->GetAttr<std::vector<int>>("paddings");
  auto dilations1 = conv_op_desc1.GetAttr<std::vector<int>>("dilations");

  bool enable0_int8 = conv_op_desc->HasAttr("enable_int8") ? true : false;
  bool enable1_int8 = conv_op_desc1->HasAttr("enable_int8") ? true : false;
  CHECK_EQ(enable0_int8, enable1_int8) << "The Conv compute type must be same";
  CHECK_EQ(groups1, 1) << "The groups of weight1_dim must be 1";
  CHECK_EQ(weight0_t->dims()[0], weight1_t->dims()[1]) << "weight0_dims[0] == weight1_dim[1]";
  for (int i = 0; i < strides1.size(); i++) {
    CHECK_EQ(strides1[i], 1) << "strides[" << i << "]: " << strides1[i] << " must be 1";
  }
  for (int i = 0; i < paddings1.size(); i++) {
    CHECK_EQ(paddings1[i], 0) << "paddings1[" << i << "]: " << paddings1[i] << " must be 0";
  }
  for (int i = 0; i < dilations1.size(); i++) {
    CHECK_EQ(dilations1[i], 1) << "dilations1[" << i << "]: " << dilations1[i] << " must be 1";
  }
  // comupte new_wight and new bias
  ///////////////////////////////////////////////////////////////////////////////
  // Compute ConvConvFuser
  // Before fusion
  //
  //   conv(x) = conv(x) = kx + z = y
  //   conv(y) = ay + b
  //
  // After fusion:
  //
  //   conv(conv(x)) = a(kx + z) + b = akx + az + b
  //
  //   new_weights = ak
  //   new_bias = az + b
  ///////////////////////////////////////////////////////////////////////////////
  if (enable0_int8) {
    LOG(FATAL) << "it doesn't support";
    return;
  } else {
    // compute new conv_weight
    Tensor weight_tensor;
    ComputeNewWeight(&weight_tensor, weight0_t, weight1_t);
    weight0_t->Resize(weight_tensor.dims());
    auto new_weight_d = weight_tensor.data<float>();
    auto weight_d = weight0_t->mutable_data<float>();
    for (int i = 0; i < weight_tensor.data_size(); i++){
      weight_d[i] = new_weight_d[i];
    }
  }

  // compute new conv_bias
  if (conv_has_bias0_ && conv_op_desc->HasInput("Bias") &&
      conv_op_desc->Input("Bias").size() > 0) {
    auto bias_t0 = scope->FindVar(matched.at("conv_bias0")->arg()->name)
                           ->GetMutable<lite::Tensor>();
    auto bias_d0 = bias_t0->data<float>();
    if (conv_has_bias1_ && conv_op_desc1->HasInput("Bias") &&
      conv_op_desc1->Input("Bias").size() > 0) {
      auto bias_t1 = scope->FindVar(matched.at("conv_bias1")->arg()->name)
                              ->GetMutable<lite::Tensor>();
      auto bias_d1 = bias_t1->data<float>();
      Tensor bias;
      ComputeNewBias(&bias, bias_t0, weight1_t, bias_t1);
      auto bias_d = bias->data<float>();
      for (int i = 0; i < bias_t1->data_size(); i++) {
        bias_d1[i] = bias_d[i];
      }
      conv_op_desc->SetInput("Bias",
                         {matched.at("conv_bias1")->arg()->name});  // conv_bias
      IR_NODE_LINK_TO(matched.at("conv_bias1"), matched.at("conv2d0"));
    } else {
      Tensor bias;
      ComputeNewBias(&bias, bias_t0, weight1_t, nullptr);
      bias_t0.Resize(bias.dims());
      auto bias_d = bias->data<float>();
      auto bias_ptr = bias_t0->mutable_data<float>();
      for (int i = 0; i < bias.data_size(); i++) {
        bias_ptr[i] = bias_d[i];
      }
      conv_op_desc->SetInput("Bias",
                         {matched.at("conv_bias0")->arg()->name});  // conv_bias
    }
  } else {
    if (conv_has_bias1_ && conv_op_desc1->HasInput("Bias") &&
      conv_op_desc1->Input("Bias").size() > 0) {
      conv_op_desc->SetInput("Bias",
                         {matched.at("conv_bias1")->arg()->name});  // conv_bias
      IR_NODE_LINK_TO(matched.at("conv_bias1"), matched.at("conv2d0"));
    }
  }

  conv_op_desc->SetType(conv_type_);
  conv_op_desc->SetInput("Input", {matched.at("conv_input0")->arg()->name});
  conv_op_desc->SetInput("Filter", {matched.at("conv_weight0")->arg()->name});
  conv_op_desc->SetOutput("Output", {matched.at("conv_out1")->arg()->name});

  auto update_conv_desc = *conv_instruct->mutable_op_info();
  conv_instruct->ResetOp(update_conv_desc, graph->valid_places());

  IR_OP_VAR_LINK(matched.at("conv2d0"), matched.at("conv_out1"));
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
