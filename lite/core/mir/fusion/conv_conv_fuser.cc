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

inline void ConvConvFuser::createPattern() {
  auto* conv_input0 = VarNode("conv_input0")
                          ->assert_is_op_input(conv_type0_, "Input")
                          ->AsInput();
  auto* conv_weight0 = VarNode("conv_weight0")
                           ->assert_is_op_input(conv_type0_, "Filter")
                           ->AsInput();
  auto* conv0 = OpNode("conv2d0", conv_type0_)->assert_is_op(conv_type0_);
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

  auto* conv_out1 = VarNode("conv_out1")
                        ->assert_is_op_output(conv_type1_, "Output")
                        ->AsOutput();

  if (conv_has_bias0_) {
    if (conv_has_bias1_) {
      auto* conv_bias0 = VarNode("conv_bias0")
                             ->assert_is_op_input(conv_type0_, "Bias")
                             ->AsIntermediate();
      auto* conv_bias1 = VarNode("conv_bias1")
                             ->assert_is_op_input(conv_type1_, "Bias")
                             ->AsInput();
      conv0->LinksFrom({conv_input0, conv_weight0, conv_bias0})
          .LinksTo({conv_out0});
      conv1->LinksFrom({conv_out0, conv_weight1, conv_bias1})
          .LinksTo({conv_out1});
    } else {
      auto* conv_bias0 = VarNode("conv_bias0")
                             ->assert_is_op_input(conv_type0_, "Bias")
                             ->AsIntermediate();
      conv0->LinksFrom({conv_input0, conv_weight0, conv_bias0})
          .LinksTo({conv_out0});
      conv1->LinksFrom({conv_out0, conv_weight1}).LinksTo({conv_out1});
    }
  } else {
    conv0->LinksFrom({conv_input0, conv_weight0}).LinksTo({conv_out0});
    if (conv_has_bias1_) {
      auto* conv_bias1 = VarNode("conv_bias1")
                             ->assert_is_op_input(conv_type1_, "Bias")
                             ->AsInput();
      conv1->LinksFrom({conv_out0, conv_weight1, conv_bias1})
          .LinksTo({conv_out1});
    } else {
      conv1->LinksFrom({conv_out0, conv_weight1}).LinksTo({conv_out1});
    }
  }
}

void ConvConvFuser::BuildPattern() {
  for (auto& node : graph_->StmtTopologicalOrder()) {
    if (node->IsStmt() &&
        node->AsStmt().picked_kernel().op_type() == conv_type0_) {
      auto* scope = node->stmt()->op()->scope();
      auto conv_op_desc0 = node->stmt()->mutable_op_info();
      // find outlinks of conv2d: in_arg_node
      auto conv2d_outlinks = node->outlinks;
      VLOG(5) << "conv2d_outlinks.size():" << conv2d_outlinks.size();
      if (conv2d_outlinks.size() == 1) {
        auto next_node_tmp = conv2d_outlinks.front();
        if (next_node_tmp->IsArg() && next_node_tmp->outlinks.size() == 1) {
          auto next_node = next_node_tmp->outlinks.front();
          auto conv0_in = node->inlinks;
          auto conv0_wei_name = conv0_in.front();
          VLOG(5) << "next_node->IsStmt(): " << next_node->IsStmt();
          VLOG(5) << ", next op_type:"
                  << next_node->AsStmt().picked_kernel().op_type();
          if (next_node->IsStmt() &&
              next_node->AsStmt().picked_kernel().op_type() == conv_type1_) {
            // find conv->conv pattern
            auto conv1_in = next_node->inlinks;
            auto conv1_wei_name = conv1_in.front();
            auto a = conv0_wei_name->AsArg().name;
            auto b = conv1_wei_name->AsArg().name;
            VLOG(5) << "conv0_wei_name: " << a;
            VLOG(5) << "conv1_wei_name: " << b;
            auto conv_op_desc1 = next_node->stmt()->mutable_op_info();
            auto weight0_dims = scope->FindVar(a)->Get<lite::Tensor>().dims();
            auto weight1_dims = scope->FindVar(b)->Get<lite::Tensor>().dims();
            auto groups0 = conv_op_desc0->GetAttr<int>("groups");
            auto groups1 = conv_op_desc1->GetAttr<int>("groups");
            auto strides1 = conv_op_desc1->GetAttr<std::vector<int>>("strides");
            auto paddings1 =
                conv_op_desc1->GetAttr<std::vector<int>>("paddings");
            auto dilations1 =
                conv_op_desc1->GetAttr<std::vector<int>>("dilations");
            auto ch_out_0 = weight0_dims[0];
            auto ch_in_0 = weight0_dims[1] * groups0;
            auto ch_out_1 = weight1_dims[0];
            auto ch_in_1 = weight1_dims[1] * groups1;
            auto kh = weight1_dims[2];
            auto kw = weight1_dims[3];
            bool enable0_int8 =
                conv_op_desc0->HasAttr("enable_int8") ? true : false;
            bool enable1_int8 =
                conv_op_desc1->HasAttr("enable_int8") ? true : false;
            if (!(kw == 1 && kh == 1)) {
              VLOG(5) << "The kernel size of the second conv must be 1x1";
              continue;
            }
            if (groups0 != 1 || groups1 != 1) {
              VLOG(5) << "The all groups of weight_dim must be 1";
              continue;
            }
            if (ch_out_0 != ch_in_1) {
              VLOG(5) << "channel0_out must be equal channel1_in";
              continue;
            }
            if (enable0_int8 || enable0_int8 != enable1_int8) {
              VLOG(5) << "The Conv-compute type must be same and be false";
              continue;
            }
            // computation: ic0 x (oc1-oc0) < oc0 x oc1
            VLOG(5) << "a: " << (ch_in_0 * (ch_out_1 - ch_out_0)) << " <= "
                    << "b: " << (ch_out_0 * ch_out_1);

            if (ch_in_0 * (ch_out_1 - ch_out_0) > ch_out_0 * ch_out_1) {
              VLOG(5) << "it dose not meet the requirment of conv+conv fusion "
                      << "computation "
                      << "a: " << (ch_in_0 * (ch_out_1 - ch_out_0)) << " <= "
                      << "b: " << (ch_out_0 * ch_out_1);
              continue;
            }
            // create pattern
            VLOG(5) << "matched: " << conv_type0_ << " and " << conv_type1_;
            createPattern();
            return;
          }
        }
      }
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

  // conv1
  auto weight1_t = scope->FindVar(matched.at("conv_weight1")->arg()->name)
                       ->GetMutable<lite::Tensor>();
  bool enable0_int8 = conv_op_desc->HasAttr("enable_int8") ? true : false;
  auto strides1 = conv_op_desc1->GetAttr<std::vector<int>>("strides");
  auto paddings1 = conv_op_desc1->GetAttr<std::vector<int>>("paddings");
  auto dilations1 = conv_op_desc1->GetAttr<std::vector<int>>("dilations");

  for (int i = 0; i < strides1.size(); i++) {
    CHECK_EQ(strides1[i], 1) << "strides[" << i << "]: " << strides1[i]
                             << " must be 1";
  }
  for (int i = 0; i < paddings1.size(); i++) {
    CHECK_EQ(paddings1[i], 0) << "paddings1[" << i << "]: " << paddings1[i]
                              << " must be 0";
  }
  for (int i = 0; i < dilations1.size(); i++) {
    CHECK_EQ(dilations1[i], 1) << "dilations1[" << i << "]: " << dilations1[i]
                               << " must be 1";
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
    LOG(FATAL) << "it doesn't support int8";
  } else {
    // compute new conv_weight
    Tensor weight_tensor;
    auto in_dims = weight0_t->dims();
    auto weight_dims = weight1_t->dims();
    const float* din = weight0_t->data<float>();
    const float* weights = weight1_t->data<float>();
    int oc0 = in_dims[0];
    int ic = in_dims[1];
    int ih = in_dims[2];
    int iw = in_dims[3];
    int oc = weight_dims[0];
    weight_tensor.Resize({oc, ic, ih, iw});
    float* dout = weight_tensor.mutable_data<float>();
    ComputeNewWeight(dout, din, weights, oc0, ic, ih, iw, oc);
    weight0_t->CopyDataFrom(weight_tensor);
  }
  // compute new conv_bias
  if (conv_has_bias0_ && conv_op_desc->HasInput("Bias") &&
      conv_op_desc->Input("Bias").size() > 0) {
    auto bias_t0 = scope->FindVar(matched.at("conv_bias0")->arg()->name)
                       ->GetMutable<lite::Tensor>();
    if (conv_has_bias1_ && conv_op_desc1->HasInput("Bias") &&
        conv_op_desc1->Input("Bias").size() > 0) {
      auto bias_t1 = scope->FindVar(matched.at("conv_bias1")->arg()->name)
                         ->GetMutable<lite::Tensor>();
      Tensor bias;
      bias.CopyDataFrom(*bias_t1);
      auto bias_data = bias.mutable_data<float>();
      ComputeNewBias(bias_data, bias_t0, weight1_t, bias_t1);
      bias_t1->CopyDataFrom(bias);
      conv_op_desc->SetInput(
          "Bias", {matched.at("conv_bias1")->arg()->name});  // conv_bias
      IR_NODE_LINK_TO(matched.at("conv_bias1"), matched.at("conv2d0"));
    } else {
      Tensor bias;
      auto weight_dims = weight1_t->dims();
      bias.Resize({weight_dims[0]});
      auto bias_d = bias.mutable_data<float>();
      ComputeNewBias(bias_d, bias_t0, weight1_t, nullptr);
      bias_t0->CopyDataFrom(bias);
      conv_op_desc->SetInput(
          "Bias", {matched.at("conv_bias0")->arg()->name});  // conv_bias
    }
  } else {
    if (conv_has_bias1_ && conv_op_desc1->HasInput("Bias") &&
        conv_op_desc1->Input("Bias").size() > 0) {
      conv_op_desc->SetInput(
          "Bias", {matched.at("conv_bias1")->arg()->name});  // conv_bias
      IR_NODE_LINK_TO(matched.at("conv_bias1"), matched.at("conv2d0"));
    }
  }
  conv_op_desc->SetType(conv_type0_);
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
