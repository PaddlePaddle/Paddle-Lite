// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {
/* Squeeze and Excitation Block Fusion for SE-ResNet */
/* graph[1]: sub block                               */
/*               in_Input                            */
/*                |     \                            */
/*                |       |                          */
/*                |  Global Pooling                  */
/*                |       |                          */
/*                |       |                          */
/*                |       FC                         */
/*                |       |                          */
/*                |       |                          */
/*                |       FC                         */
/*                \       |                          */
/*                  \     |                          */
/*                    scale                          */
/*                      |                            */
/*                   out_Output                      */

/* DEPERCATED */
// TODO(weihaoji): remove XPUSqueezeExcitationFuser_DEPREC after xpu_fc refactor
class XPUSqueezeExcitationFuser_DEPREC : public FuseBase {
 public:
  explicit XPUSqueezeExcitationFuser_DEPREC(
      const std::string& excitation_act_type1,
      const std::string& excitation_act_type2,
      const std::string& block_act_type,
      bool with_branch) {
    excitation_act_type1_ = excitation_act_type1;
    excitation_act_type2_ = excitation_act_type2;
    block_act_type_ = block_act_type;
    with_branch_ = with_branch;
  }
  void BuildPattern() override {
    PMNode* ew_branch_add = nullptr;
    PMNode* ew_branch_add_in = nullptr;
    PMNode* ew_branch_add_out = nullptr;
    PMNode* block_act = nullptr;
    PMNode* block_act_out = nullptr;

    auto* input = VarNode("input")
                      ->assert_is_op_input("pool2d", "X")
                      ->assert_is_op_input("elementwise_mul", "X")
                      ->AsInput();
    auto* pool = OpNode("pool", "pool2d")
                     ->assert_op_attr<bool>("global_pooling", true)
                     ->assert_op_attr<std::string>("pooling_type", "avg")
                     ->AsIntermediate();
    auto* pool_out = VarNode("pool_out")
                         ->assert_is_op_output("pool2d", "Out")
                         ->assert_is_op_input("mul", "X")
                         ->AsIntermediate();
    auto* mul_1 = OpNode("mul_1", "mul")->AsIntermediate();
    auto* mul_1_w =
        VarNode("mul_1_w")->assert_is_op_input("mul", "Y")->AsIntermediate();
    auto* mul_1_out = VarNode("mul_1_out")
                          ->assert_is_op_output("mul", "Out")
                          ->assert_is_op_input(excitation_act_type1_, "X")
                          ->AsIntermediate();
    auto* act_1 = OpNode("act_1", excitation_act_type1_)->AsIntermediate();
    auto* act_1_out = VarNode("act_1_out")
                          ->assert_is_op_output(excitation_act_type1_, "Out")
                          ->assert_is_op_input("mul", "X")
                          ->AsIntermediate();
    auto* mul_2 = OpNode("mul_2", "mul")->AsIntermediate();
    auto* mul_2_w =
        VarNode("mul_2_w")->assert_is_op_input("mul", "Y")->AsIntermediate();
    auto* mul_2_out = VarNode("mul_2_out")
                          ->assert_is_op_output("mul", "Out")
                          ->assert_is_op_input(excitation_act_type2_, "X")
                          ->AsIntermediate();
    auto* act_2 = OpNode("act_2", excitation_act_type2_)->AsIntermediate();
    auto* act_2_out = VarNode("act_2_out")
                          ->assert_is_op_output(excitation_act_type2_, "Out")
                          ->assert_is_op_input("elementwise_mul", "Y")
                          ->AsIntermediate();
    auto* ew_mul = OpNode("ew_mul", "elementwise_mul")->AsIntermediate();
    auto* ew_mul_out =
        VarNode("ew_mul_out")->assert_is_op_output("elementwise_mul", "Out");
    // branch
    if (with_branch_) {
      ew_branch_add_in = VarNode("ew_branch_add_in")
                             ->assert_is_op_input("elementwise_add", "X")
                             ->AsInput();
      ew_branch_add =
          OpNode("ew_branch_add", "elementwise_add")->AsIntermediate();
      ew_branch_add_out = VarNode("ew_branch_add_out")
                              ->assert_is_op_output("elementwise_add", "Out");
    }
    // act
    if (block_act_type_ != "linear") {
      block_act = OpNode("block_act", block_act_type_)->AsIntermediate();
      block_act_out =
          VarNode("block_act_out")->assert_is_op_output(block_act_type_, "Out");
    }
    // pass
    *input >> *pool >> *pool_out >> *mul_1 >> *mul_1_out >> *act_1 >>
        *act_1_out >> *mul_2 >> *mul_2_out >> *act_2 >> *act_2_out >> *ew_mul;
    *input >> *ew_mul;
    *ew_mul >> *ew_mul_out;

    if (with_branch_) {
      ew_mul_out->assert_is_op_input("elementwise_add", "Y")->AsIntermediate();
      *ew_mul_out >> *ew_branch_add >> *ew_branch_add_out;
      *ew_branch_add_in >> *ew_branch_add;
    } else {
      ew_branch_add_out = ew_mul_out;
    }
    if (block_act_type_ != "linear") {
      ew_branch_add_out->assert_is_op_input(block_act_type_, "X")
          ->AsIntermediate();
      *ew_branch_add_out >> *block_act >> *block_act_out;
    } else {
      block_act_out = ew_branch_add_out;
    }
    block_act_out->AsOutput();
    *mul_1_w >> *mul_1;
    *mul_2_w >> *mul_2;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    auto pool_op = matched.at("pool")->stmt()->op();
    auto* scope = pool_op->scope();
    op_desc.SetType("__xpu__squeeze_excitation_block");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    if (with_branch_) {
      op_desc.SetInput("Branch", {matched.at("ew_branch_add_in")->arg()->name});
    }

    auto mul_1_w_name = matched.at("mul_1_w")->arg()->name;
    auto* mul_1_w_t = scope->FindMutableTensor(mul_1_w_name);
    auto mul_1_w_dims = mul_1_w_t->dims();
    int mul_1_w_len = mul_1_w_t->numel();
    float* mul_1_w_on_host = mul_1_w_t->mutable_data<float>();
    auto mul_2_w_name = matched.at("mul_2_w")->arg()->name;
    auto* mul_2_w_t = scope->FindMutableTensor(mul_2_w_name);
    auto mul_2_w_dims = mul_2_w_t->dims();
    int mul_2_w_len = mul_2_w_t->numel();
    float* mul_2_w_on_host = mul_2_w_t->mutable_data<float>();
    if (mul_1_w_dims[0] != mul_2_w_dims[1] ||
        mul_1_w_dims[1] != mul_2_w_dims[0]) {
      LOG(FATAL) << "Error: Dims of excitation mul1 weight is: "
                 << mul_1_w_dims[0] << ", " << mul_1_w_dims[1]
                 << ", but get dims of excitation mul2 weight is: "
                 << mul_2_w_dims[0] << ", " << mul_2_w_dims[1];
    }
    if (mul_1_w_dims[0] % mul_2_w_dims[1] != 0) {
      LOG(FATAL) << "Error: Reduction ratio of excitation is not and integer.";
    }

    op_desc.SetAttr<std::vector<int>>(
        "filter_dims",
        {static_cast<int>(mul_1_w_dims[0] / mul_1_w_dims[1]),
         static_cast<int>(mul_1_w_dims[0])});
    op_desc.SetAttr<std::vector<int>>("op_type", std::vector<int>{4});
    op_desc.SetAttr<std::vector<int>>("place_x", std::vector<int>{0});
    op_desc.SetAttr<std::vector<int>>("place_y", std::vector<int>{9});
    op_desc.SetAttr<std::vector<int>>("place_z", std::vector<int>{10});
    op_desc.SetAttr<std::vector<int>>("strides", std::vector<int>{});
    op_desc.SetAttr<std::vector<int>>("paddings", std::vector<int>{});
    op_desc.SetAttr<std::vector<int>>("dilations", std::vector<int>{});
    op_desc.SetAttr<std::vector<int>>("groups", std::vector<int>{});
    op_desc.SetAttr<std::vector<int>>("block_lod", std::vector<int>{1});
    op_desc.SetAttr<std::vector<int>>("conv_bias", std::vector<int>{0});
    op_desc.SetAttr<bool>("has_bias", false);

    std::unique_ptr<float[]> encode_filter_float(
        new float[mul_1_w_len + mul_2_w_len]);
    memcpy(encode_filter_float.get(),
           mul_1_w_on_host,
           mul_1_w_len * sizeof(float));
    memcpy(encode_filter_float.get() + mul_1_w_len,
           mul_2_w_on_host,
           mul_2_w_len * sizeof(float));
    std::string new_filter_name = "se_" + mul_1_w_name;
    auto* new_filter_node = graph->NewArgumentNode(new_filter_name);
    new_filter_node->arg()->is_weight = true;
    new_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_filter_t = scope->MutableParent()->NewTensor(new_filter_name);
    new_filter_t->set_precision(paddle::lite_api::PrecisionType::kFloat);
    new_filter_t->set_persistable(true);
    new_filter_t->Resize({mul_1_w_len + mul_2_w_len});
    float* new_filter_ptr = new_filter_t->mutable_data<float>();
    memcpy(new_filter_ptr,
           encode_filter_float.get(),
           (mul_1_w_len + mul_2_w_len) * sizeof(float));
    op_desc.SetInput("Filter", {new_filter_name});

    std::string output_name;
    if (block_act_type_ != "linear") {
      output_name = matched.at("block_act_out")->arg()->name;
    } else if (with_branch_) {
      output_name = matched.at("ew_branch_add_out")->arg()->name;
    } else {
      output_name = matched.at("ew_mul_out")->arg()->name;
    }
    op_desc.SetOutput("Output", {output_name});

    std::string max_output_name = output_name + "_max";
    auto* max_output_node = graph->NewArgumentNode(max_output_name);
    max_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* max_output_tensor = scope->NewTensor(max_output_name);
    max_output_tensor->set_precision(paddle::lite_api::PrecisionType::kFloat);
    max_output_tensor->set_persistable(true);
    op_desc.SetOutput("OutputMax", {max_output_name});

    std::map<std::string, int> act_map{
        {"linear", 0}, {"relu", 1}, {"sigmoid", 2}};

    op_desc.SetAttr<bool>("has_branch", with_branch_);
    op_desc.SetAttr<std::vector<int>>(
        "act_type",
        std::vector<int>{act_map[excitation_act_type1_],
                         act_map[excitation_act_type2_],
                         act_map[block_act_type_]});
    op_desc.SetAttr<std::vector<float>>("act_param",
                                        std::vector<float>{0, 0, 0});

    auto& valid_places = pool_op->valid_places();
    auto se_op = LiteOpRegistry::Global().Create(op_desc.Type());
    se_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(se_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    if (with_branch_) {
      DirectedLink(matched.at("ew_branch_add_in"), new_op_node);
    }
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    if (block_act_type_ != "linear") {
      IR_NODE_LINK_TO(new_op_node, matched.at("block_act_out"));
    } else if (with_branch_) {
      IR_NODE_LINK_TO(new_op_node, matched.at("ew_branch_add_out"));
    } else {
      IR_NODE_LINK_TO(new_op_node, matched.at("ew_mul_out"));
    }
    IR_NODE_LINK_TO(new_op_node, max_output_node);
  }

 private:
  std::string excitation_act_type1_;
  std::string excitation_act_type2_;
  std::string block_act_type_;
  bool with_branch_;
};

class XPUSqueezeExcitationFuser : public FuseBase {
 public:
  explicit XPUSqueezeExcitationFuser(const std::string& op_type,
                                     const std::string& block_act_type,
                                     bool with_branch,
                                     bool with_bias) {
    op_type_ = op_type;
    block_act_type_ = block_act_type;
    with_branch_ = with_branch;
    with_bias_ = with_bias;
  }
  void BuildPattern() override {
    PMNode* ew_branch_add = nullptr;
    PMNode* ew_branch_add_in = nullptr;
    PMNode* ew_branch_add_out = nullptr;
    PMNode* block_act = nullptr;
    PMNode* block_act_out = nullptr;
    PMNode* mul_1_bias = nullptr;
    PMNode* mul_2_bias = nullptr;

    auto* input = VarNode("input")
                      ->assert_is_op_input("pool2d", "X")
                      ->assert_is_op_input("elementwise_mul", "X")
                      ->AsInput();
    auto pool2d_teller = [](const Node* x) -> bool {
      if (x && x->IsStmt()) {
        auto* op_info = x->stmt()->op_info();
        if (op_info->HasAttr("adaptive") &&
            op_info->GetAttr<bool>("adaptive")) {
          if (op_info->GetAttr<std::vector<int>>("ksize")[0] != 1 ||
              op_info->GetAttr<std::vector<int>>("ksize")[1] != 1) {
            return false;
          }
        } else if (op_info->GetAttr<bool>("global_pooling") == false) {
          return false;
        }
      }
      return true;
    };
    auto* pool = OpNode("pool", "pool2d")
                     ->assert_node_satisfied(pool2d_teller)
                     ->assert_op_attr<std::string>("pooling_type", "avg")
                     ->AsIntermediate();
    auto* pool_out = VarNode("pool_out")
                         ->assert_is_op_output("pool2d", "Out")
                         ->assert_is_op_input(op_type_, "Input")
                         ->AsIntermediate();

    auto mul_teller = [](const Node* x) -> bool {
      if (x && x->IsStmt()) {
        auto* op_info = x->stmt()->op_info();
        auto in_c = op_info->GetAttr<std::vector<int>>("filter_dims")[0];
        auto out_c = op_info->GetAttr<std::vector<int>>("filter_dims")[1];
        auto bigger = std::max(in_c, out_c);
        auto smaller = std::min(in_c, out_c);
        if (bigger % smaller != 0) {
          return false;
        }
      }
      return true;
    };
    auto* mul_1 = OpNode("mul_1", op_type_)
                      ->assert_node_satisfied(mul_teller)
                      ->AsIntermediate();
    auto* mul_1_w = VarNode("mul_1_w")
                        ->assert_is_op_input(op_type_, "Filter")
                        ->AsIntermediate();
    auto* mul_1_out = VarNode("mul_1_out")
                          ->assert_is_op_output(op_type_, "Output")
                          ->assert_is_op_input(op_type_, "Input")
                          ->AsIntermediate();
    auto* mul_1_out_max = VarNode("mul_1_out_max")
                              ->assert_is_op_output(op_type_, "OutputMax")
                              ->AsIntermediate();
    auto* mul_2 = OpNode("mul_2", op_type_)
                      ->assert_node_satisfied(mul_teller)
                      ->AsIntermediate();
    auto* mul_2_w = VarNode("mul_2_w")
                        ->assert_is_op_input(op_type_, "Filter")
                        ->AsIntermediate();
    auto* mul_2_out = VarNode("mul_2_out")
                          ->assert_is_op_output(op_type_, "Output")
                          ->assert_is_op_input("elementwise_mul", "Y")
                          ->AsIntermediate();
    auto* mul_2_out_max = VarNode("mul_2_out_max")
                              ->assert_is_op_output(op_type_, "OutputMax")
                              ->AsIntermediate();
    if (with_bias_) {
      mul_1_bias = VarNode("mul_1_bias")
                       ->assert_is_op_input(op_type_, "Bias")
                       ->AsIntermediate();
      mul_2_bias = VarNode("mul_2_bias")
                       ->assert_is_op_input(op_type_, "Bias")
                       ->AsIntermediate();
    }
    auto* ew_mul = OpNode("ew_mul", "elementwise_mul")->AsIntermediate();
    auto* ew_mul_out =
        VarNode("ew_mul_out")->assert_is_op_output("elementwise_mul", "Out");
    // branch
    if (with_branch_) {
      ew_branch_add_in = VarNode("ew_branch_add_in")
                             ->assert_is_op_input("elementwise_add", "X")
                             ->AsInput();
      ew_branch_add =
          OpNode("ew_branch_add", "elementwise_add")->AsIntermediate();
      ew_branch_add_out = VarNode("ew_branch_add_out")
                              ->assert_is_op_output("elementwise_add", "Out");
    }
    // act
    if (block_act_type_ != "linear") {
      block_act = OpNode("block_act", block_act_type_)->AsIntermediate();
      block_act_out =
          VarNode("block_act_out")->assert_is_op_output(block_act_type_, "Out");
    }
    // pass
    *input >> *pool >> *pool_out >> *mul_1 >> *mul_1_out >> *mul_2 >>
        *mul_2_out >> *ew_mul;
    *input >> *ew_mul;
    *ew_mul >> *ew_mul_out;

    if (with_branch_) {
      ew_mul_out->assert_is_op_input("elementwise_add", "Y")->AsIntermediate();
      *ew_mul_out >> *ew_branch_add >> *ew_branch_add_out;
      *ew_branch_add_in >> *ew_branch_add;
    } else {
      ew_branch_add_out = ew_mul_out;
    }
    if (block_act_type_ != "linear") {
      ew_branch_add_out->assert_is_op_input(block_act_type_, "X")
          ->AsIntermediate();
      *ew_branch_add_out >> *block_act >> *block_act_out;
    } else {
      block_act_out = ew_branch_add_out;
    }
    block_act_out->AsOutput();
    *mul_1_w >> *mul_1;
    *mul_2_w >> *mul_2;
    *mul_1 >> *mul_1_out_max;
    *mul_2 >> *mul_2_out_max;
    if (with_bias_) {
      *mul_1_bias >> *mul_1;
      *mul_2_bias >> *mul_2;
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    auto pool_op = matched.at("pool")->stmt()->op();
    auto* scope = pool_op->scope();
    op_desc.SetType("__xpu__squeeze_excitation_block");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    if (with_branch_) {
      op_desc.SetInput("Branch", {matched.at("ew_branch_add_in")->arg()->name});
    }
    auto mul_1_w_name = matched.at("mul_1_w")->arg()->name;
    auto* mul_1_w_t = scope->FindMutableTensor(mul_1_w_name);
    auto mul_1_w_dims = mul_1_w_t->dims();
    int mul_1_w_len = mul_1_w_t->numel();
    float* mul_1_w_on_host = mul_1_w_t->mutable_data<float>();
    auto mul_2_w_name = matched.at("mul_2_w")->arg()->name;
    auto* mul_2_w_t = scope->FindMutableTensor(mul_2_w_name);
    auto mul_2_w_dims = mul_2_w_t->dims();
    int mul_2_w_len = mul_2_w_t->numel();
    float* mul_2_w_on_host = mul_2_w_t->mutable_data<float>();
    if (mul_1_w_dims[0] != mul_2_w_dims[1] ||
        mul_1_w_dims[1] != mul_2_w_dims[0] ||
        mul_1_w_len != mul_1_w_dims[0] * mul_1_w_dims[1]) {
      LOG(FATAL) << "Error: Dims of excitation mul1 weight is: " << mul_1_w_dims
                 << ", but get dims of excitation mul2 weight is: "
                 << mul_2_w_dims;
    }
    std::unique_ptr<float[]> encode_filter_float(
        new float[mul_1_w_len + mul_2_w_len]);
    if (op_type_ == "__xpu__fc") {
      CHECK(mul_1_w_dims[0] % mul_1_w_dims[1] == 0)
          << "Error: Reduction ratio of excitation is not an integer."
          << "Received mul_1_w_dims[0]: " << mul_1_w_dims[0]
          << ", mul_1_w_dims[1]: " << mul_1_w_dims[1];
      op_desc.SetAttr<std::vector<int>>(
          "filter_dims",
          {static_cast<int>(mul_1_w_dims[0] / mul_1_w_dims[1]),
           static_cast<int>(mul_1_w_dims[0])});
      memcpy(encode_filter_float.get(),
             mul_1_w_on_host,
             mul_1_w_len * sizeof(float));
      memcpy(encode_filter_float.get() + mul_1_w_len,
             mul_2_w_on_host,
             mul_2_w_len * sizeof(float));
    } else if (op_type_ == "__xpu__conv2d") {
      CHECK(mul_1_w_dims[1] % mul_1_w_dims[0] == 0)
          << "Error: Reduction ratio of excitation is not an integer."
          << "Received mul_1_w_dims[1]: " << mul_1_w_dims[1]
          << ", mul_1_w_dims[0]: " << mul_1_w_dims[0];
      op_desc.SetAttr<std::vector<int>>(
          "filter_dims",
          {static_cast<int>(mul_1_w_dims[1] / mul_1_w_dims[0]),
           static_cast<int>(mul_1_w_dims[1])});
      paddle::lite::xpu::math::Transpose(mul_1_w_on_host,
                                         encode_filter_float.get(),
                                         mul_1_w_dims[0],
                                         mul_1_w_dims[1]);
      paddle::lite::xpu::math::Transpose(
          mul_2_w_on_host,
          encode_filter_float.get() + mul_1_w_len,
          mul_2_w_dims[0],
          mul_2_w_dims[1]);
    }
    std::string new_filter_name = "se_" + mul_1_w_name;
    auto* new_filter_node = graph->NewArgumentNode(new_filter_name);
    new_filter_node->arg()->is_weight = true;
    new_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_filter_t = scope->MutableParent()->NewTensor(new_filter_name);
    new_filter_t->set_precision(paddle::lite_api::PrecisionType::kFloat);
    new_filter_t->set_persistable(true);
    new_filter_t->Resize({mul_1_w_len + mul_2_w_len});
    float* new_filter_ptr = new_filter_t->mutable_data<float>();
    memcpy(new_filter_ptr,
           encode_filter_float.get(),
           (mul_1_w_len + mul_2_w_len) * sizeof(float));
    op_desc.SetInput("Filter", {new_filter_name});

    std::string new_bias_name = new_filter_name + "_bias";
    auto* new_bias_node = graph->NewArgumentNode(new_bias_name);
    new_bias_node->arg()->is_weight = true;
    new_bias_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_bias_t = scope->MutableParent()->NewTensor(new_bias_name);
    new_bias_t->set_precision(paddle::lite_api::PrecisionType::kFloat);
    new_bias_t->set_persistable(true);
    if (with_bias_) {
      std::vector<std::string> bias_name{matched.at("mul_1_bias")->arg()->name,
                                         matched.at("mul_2_bias")->arg()->name};
      auto* bias0_t = scope->FindMutableTensor(bias_name[0]);
      auto* bias1_t = scope->FindMutableTensor(bias_name[1]);
      int bias0_numel = bias0_t->numel();
      int bias1_numel = bias1_t->numel();
      std::unique_ptr<float[]> encode_bias(
          new float[bias0_numel + bias1_numel]);
      float* bias0_on_host = bias0_t->mutable_data<float>();
      float* bias1_on_host = bias1_t->mutable_data<float>();
      memcpy(encode_bias.get(), bias0_on_host, bias0_numel * sizeof(float));
      memcpy(encode_bias.get() + bias0_numel,
             bias1_on_host,
             bias1_numel * sizeof(float));
      new_bias_t->Resize({bias0_numel + bias1_numel});
      float* new_bias_ptr = new_bias_t->mutable_data<float>();
      memcpy(new_bias_ptr,
             encode_bias.get(),
             (bias0_numel + bias1_numel) * sizeof(float));
      op_desc.SetInput("Bias", {new_bias_name});
    }
    op_desc.SetAttr<bool>("has_bias", with_bias_);
    op_desc.SetAttr<bool>("has_branch", with_branch_);
    std::string output_name;
    if (block_act_type_ != "linear") {
      output_name = matched.at("block_act_out")->arg()->name;
    } else if (with_branch_) {
      output_name = matched.at("ew_branch_add_out")->arg()->name;
    } else {
      output_name = matched.at("ew_mul_out")->arg()->name;
    }
    op_desc.SetOutput("Output", {output_name});
    std::string max_output_name = output_name + "_max";
    auto* max_output_node = graph->NewArgumentNode(max_output_name);
    max_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* max_output_tensor = scope->NewTensor(max_output_name);
    max_output_tensor->set_precision(paddle::lite_api::PrecisionType::kFloat);
    max_output_tensor->set_persistable(true);
    op_desc.SetOutput("OutputMax", {max_output_name});

    op_desc.SetAttr<std::vector<int>>("op_type", std::vector<int>{4});
    op_desc.SetAttr<std::vector<int>>("place_x", std::vector<int>{0});
    op_desc.SetAttr<std::vector<int>>("place_y", std::vector<int>{9});
    op_desc.SetAttr<std::vector<int>>("place_z", std::vector<int>{10});
    op_desc.SetAttr<std::vector<int>>("strides", std::vector<int>{});
    op_desc.SetAttr<std::vector<int>>("paddings", std::vector<int>{});
    op_desc.SetAttr<std::vector<int>>("dilations", std::vector<int>{});
    op_desc.SetAttr<std::vector<int>>("groups", std::vector<int>{});
    op_desc.SetAttr<std::vector<int>>("block_lod", std::vector<int>{1});
    op_desc.SetAttr<std::vector<int>>("conv_bias",
                                      std::vector<int>{with_bias_});

    std::map<std::string, int> act_map{{"linear", 0},
                                       {"relu", 1},
                                       {"sigmoid", 2},
                                       {"tanh", 3},
                                       {"leaky_relu", 5},
                                       {"hard_swish", 14},
                                       {"hard_sigmoid", 15},
                                       {"relu6", 17}};

    float block_act_param_ = 0.0f;
    if (block_act_type_ == "leaky_relu") {
      auto block_act_op_desc = *matched.at("block_act")->stmt()->op_info();
      block_act_param_ = block_act_op_desc.GetAttr<float>("alpha");
    } else if (block_act_type_ == "hard_sigmoid") {
      auto block_act_op_desc = *matched.at("block_act")->stmt()->op_info();
      block_act_param_ = block_act_op_desc.GetAttr<float>("slope");
    }
    if (op_type_ == "__xpu__fc") {
      op_desc.SetAttr<std::vector<int>>(
          "act_type",
          std::vector<int>{
              matched.at("mul_1")->stmt()->op_info()->GetAttr<int>("act_type"),
              matched.at("mul_2")->stmt()->op_info()->GetAttr<int>("act_type"),
              act_map[block_act_type_]});
      op_desc.SetAttr<std::vector<float>>(
          "act_param",
          std::vector<float>{
              matched.at("mul_1")->stmt()->op_info()->GetAttr<float>(
                  "act_param"),
              matched.at("mul_2")->stmt()->op_info()->GetAttr<float>(
                  "act_param"),
              block_act_param_});
    } else {
      op_desc.SetAttr<std::vector<int>>(
          "act_type",
          std::vector<int>{
              matched.at("mul_1")->stmt()->op_info()->GetAttr<std::vector<int>>(
                  "act_type")[0],
              matched.at("mul_2")->stmt()->op_info()->GetAttr<std::vector<int>>(
                  "act_type")[0],
              act_map[block_act_type_]});
      op_desc.SetAttr<std::vector<float>>(
          "act_param",
          std::vector<float>{matched.at("mul_1")
                                 ->stmt()
                                 ->op_info()
                                 ->GetAttr<std::vector<float>>("act_param")[0],
                             matched.at("mul_2")
                                 ->stmt()
                                 ->op_info()
                                 ->GetAttr<std::vector<float>>("act_param")[0],
                             block_act_param_});
    }

    auto& valid_places = pool_op->valid_places();
    auto se_op = LiteOpRegistry::Global().Create(op_desc.Type());
    se_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(se_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    if (with_branch_) {
      DirectedLink(matched.at("ew_branch_add_in"), new_op_node);
    }
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    if (block_act_type_ != "linear") {
      IR_NODE_LINK_TO(new_op_node, matched.at("block_act_out"));
    } else if (with_branch_) {
      IR_NODE_LINK_TO(new_op_node, matched.at("ew_branch_add_out"));
    } else {
      IR_NODE_LINK_TO(new_op_node, matched.at("ew_mul_out"));
    }
    IR_NODE_LINK_TO(new_op_node, max_output_node);
    if (with_bias_) {
      IR_NODE_LINK_TO(new_bias_node, new_op_node);
    }
  }

 private:
  std::string op_type_;
  std::string block_act_type_;
  bool with_branch_;
  bool with_bias_;
};

}  // namespace fusion

class XPUSqueezeExcitationFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    /* DEPERCATED */
    // TODO(weihaoji): remove XPUSqueezeExcitationFuser_DEPREC after xpu_fc
    // refactor
    for (auto with_branch : {true, false}) {
      for (auto excitation_act_type1 : {"relu", "sigmoid"}) {
        for (auto excitation_act_type2 : {"relu", "sigmoid"}) {
          for (auto block_act_type : {"relu", "sigmoid", "linear"}) {
            fusion::XPUSqueezeExcitationFuser_DEPREC fuser(excitation_act_type1,
                                                           excitation_act_type2,
                                                           block_act_type,
                                                           with_branch);
            fuser(graph.get());
          }
        }
      }
    }
    for (auto with_branch : {true, false}) {
      for (auto with_bias : {true, false}) {
        for (auto op_type : {/*"__xpu__fc",*/ "__xpu__conv2d"}) {
          for (auto act_type : {"relu",
                                "sigmoid",
                                "tanh",
                                "leaky_relu",
                                "hard_swish",
                                "hard_sigmoid",
                                "relu6",
                                "linear"}) {
            fusion::XPUSqueezeExcitationFuser fuser(
                op_type, act_type, with_branch, with_bias);
            fuser(graph.get());
          }
        }
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__squeeze_excitation_fuse_pass,
                  paddle::lite::mir::XPUSqueezeExcitationFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__squeeze_excitation_block");
