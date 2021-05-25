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
#include <set>
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/core/context.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/type_precision_cast_pass.h"  // For UpdateInputs()
#include "lite/core/mir/xpu_pattern_matcher_high_api.h"
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace mir {

namespace fusion {

class XPUSingleEncoderFuser : public FuseBase {
 public:
  explicit XPUSingleEncoderFuser(const std::string& act_type = "gelu",
                                 const std::string& input_pos = "Y",
                                 const std::string& qkv_ln_2_out_pos = "Y",
                                 const std::string& matmul_type = "matmul",
                                 const std::string& mul_type = "mul",
                                 bool with_q_scale = true,
                                 bool norm_before = false)
      : act_type_(act_type),
        input_pos_(input_pos),
        qkv_ln_2_out_pos_(qkv_ln_2_out_pos),
        matmul_type_(matmul_type),
        mul_type_(mul_type),
        with_q_scale_(with_q_scale),
        norm_before_(norm_before) {}

  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("elementwise_add", input_pos_)
                      ->AsInput();
    PMNode* ln_before_scale = nullptr;
    PMNode* ln_before_bias = nullptr;
    PMNode* ln_before = nullptr;
    PMNode* ln_before_out = nullptr;
    PMNode* ln_before_mean = nullptr;
    PMNode* ln_before_var = nullptr;
    if (norm_before_) {
      input->assert_is_op_input("layer_norm", "X");
      ln_before_scale = VarNode("ln_before_scale")
                            ->assert_is_op_input("layer_norm", "Scale")
                            ->AsInput();
      ln_before_bias = VarNode("ln_before_bias")
                           ->assert_is_op_input("layer_norm", "Bias")
                           ->AsInput();
      ln_before = OpNode("ln_before", "layer_norm")->AsIntermediate();
      ln_before_out = VarNode("ln_before_out")
                          ->assert_is_op_output("layer_norm", "Y")
                          ->assert_is_op_input(mul_type_, "X")
                          ->AsIntermediate();
      ln_before_mean = VarNode("ln_before_mean")
                           ->assert_is_op_output("layer_norm", "Mean")
                           ->AsIntermediate();
      ln_before_var = VarNode("ln_before_var")
                          ->assert_is_op_output("layer_norm", "Variance")
                          ->AsIntermediate();
    } else {
      input->assert_is_op_input(mul_type_, "X");
    }

    auto* q_mul_y =
        VarNode("q_mul_y")->assert_is_op_input(mul_type_, "Y")->AsInput();
    auto* q_mul = OpNode("q_mul", mul_type_);
    auto* q_mul_out = VarNode("q_mul_out")
                          ->assert_is_op_output(mul_type_, "Out")
                          ->assert_is_op_input("elementwise_add", "X")
                          ->AsIntermediate();
    auto* q_add_y = VarNode("q_add_y")
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->AsInput();
    auto* q_add = OpNode("q_add", "elementwise_add")->AsIntermediate();
    auto* q_add_out = VarNode("q_add_out")
                          ->assert_is_op_output("elementwise_add", "Out")
                          ->assert_is_op_input("reshape2", "X")
                          ->AsIntermediate();
    auto* q_reshape2 = OpNode("q_reshape2", "reshape2")->AsIntermediate();
    auto* q_reshape2_out = VarNode("q_reshape2_out")
                               ->assert_is_op_output("reshape2", "Out")
                               ->assert_is_op_input("transpose2", "X")
                               ->AsIntermediate();
    auto* q_reshape2_xshape = VarNode("q_reshape2_xshape")
                                  ->assert_is_op_output("reshape2", "XShape")
                                  ->AsIntermediate();
    std::string target_op_type = "matmul";
    if (with_q_scale_) {
      target_op_type = "scale";
    }
    auto* q_transpose2 = OpNode("q_transpose2", "transpose2")->AsIntermediate();
    auto* q_transpose2_out = VarNode("q_transpose2_out")
                                 ->assert_is_op_output("transpose2", "Out")
                                 ->assert_is_op_input(target_op_type, "X")
                                 ->AsIntermediate();
    auto* q_transpose2_xshape =
        VarNode("q_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();

    PMNode* q_scale = nullptr;
    PMNode* q_scale_out = nullptr;
    if (with_q_scale_) {
      q_scale = OpNode("q_scale", "scale")->AsIntermediate();
      q_scale_out = VarNode("q_scale_out")
                        ->assert_is_op_output("scale", "Out")
                        ->assert_is_op_input("matmul", "X")
                        ->AsIntermediate();
    }

    auto* k_mul_y =
        VarNode("k_mul_y")->assert_is_op_input(mul_type_, "Y")->AsInput();
    auto* k_mul = OpNode("k_mul", mul_type_)->AsIntermediate();
    auto* k_mul_out = VarNode("k_mul_out")
                          ->assert_is_op_output(mul_type_, "Out")
                          ->assert_is_op_input("elementwise_add", "X")
                          ->AsIntermediate();
    auto* k_add_y = VarNode("k_add_y")
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->AsInput();
    auto* k_add = OpNode("k_add", "elementwise_add")->AsIntermediate();
    auto* k_add_out = VarNode("k_add_out")
                          ->assert_is_op_output("elementwise_add", "Out")
                          ->assert_is_op_input("reshape2", "X")
                          ->AsIntermediate();
    auto* k_reshape2 = OpNode("k_reshape2", "reshape2")->AsIntermediate();
    auto* k_reshape2_out = VarNode("k_reshape2_out")
                               ->assert_is_op_output("reshape2", "Out")
                               ->assert_is_op_input("transpose2", "X")
                               ->AsIntermediate();
    auto* k_reshape2_xshape = VarNode("k_reshape2_xshape")
                                  ->assert_is_op_output("reshape2", "XShape")
                                  ->AsIntermediate();
    auto* k_transpose2 = OpNode("k_transpose2", "transpose2")->AsIntermediate();
    auto* k_transpose2_out = VarNode("k_transpose2_out")
                                 ->assert_is_op_output("transpose2", "Out")
                                 ->assert_is_op_input("matmul", "Y")
                                 ->AsIntermediate();
    auto* k_transpose2_xshape =
        VarNode("k_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();

    auto* qk_matmul = OpNode("qk_matmul", "matmul")->AsIntermediate();
    auto* qk_matmul_out = VarNode("qk_matmul_out")
                              ->assert_is_op_output("matmul", "Out")
                              ->assert_is_op_input("elementwise_add", "X")
                              ->AsIntermediate();
    auto* qk_mask = VarNode("qk_mask")
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->AsInput();
    auto* qk_add = OpNode("qk_add", "elementwise_add")->AsIntermediate();
    auto* qk_add_out = VarNode("qk_add_out")
                           ->assert_is_op_output("elementwise_add", "Out")
                           ->assert_is_op_input("softmax", "X")
                           ->AsIntermediate();
    auto* qk_softmax = OpNode("qk_softmax", "softmax")->AsIntermediate();
    auto* qk_softmax_out = VarNode("qk_softmax_out")
                               ->assert_is_op_output("softmax", "Out")
                               ->AsIntermediate();

    auto* v_mul_y =
        VarNode("v_mul_y")->assert_is_op_input(mul_type_, "Y")->AsInput();
    auto* v_mul = OpNode("v_mul", mul_type_)->AsIntermediate();
    auto* v_mul_out = VarNode("v_mul_out")
                          ->assert_is_op_output(mul_type_, "Out")
                          ->assert_is_op_input("elementwise_add", "X")
                          ->AsIntermediate();
    auto* v_add_y = VarNode("v_add_y")
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->AsInput();
    auto* v_add = OpNode("v_add", "elementwise_add")->AsIntermediate();
    auto* v_add_out = VarNode("v_add_out")
                          ->assert_is_op_output("elementwise_add", "Out")
                          ->assert_is_op_input("reshape2", "X")
                          ->AsIntermediate();
    auto* v_reshape2 = OpNode("v_reshape2", "reshape2")->AsIntermediate();
    auto* v_reshape2_out = VarNode("v_reshape2_out")
                               ->assert_is_op_output("reshape2", "Out")
                               ->assert_is_op_input("transpose2", "X")
                               ->AsIntermediate();
    auto* v_reshape2_xshape = VarNode("v_reshape2_xshape")
                                  ->assert_is_op_output("reshape2", "XShape")
                                  ->AsIntermediate();
    auto* v_transpose2 = OpNode("v_transpose2", "transpose2")->AsIntermediate();
    auto* v_transpose2_out = VarNode("v_transpose2_out")
                                 ->assert_is_op_output("transpose2", "Out")
                                 ->assert_is_op_input(matmul_type_, "Y")
                                 ->AsIntermediate();
    auto* v_transpose2_xshape =
        VarNode("v_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();

    auto* qkv_matmul = OpNode("qkv_matmul", matmul_type_)->AsIntermediate();
    auto* qkv_matmul_out = VarNode("qkv_matmul_out")
                               ->assert_is_op_output(matmul_type_, "Out")
                               ->assert_is_op_input("transpose2", "X")
                               ->AsIntermediate();
    auto* qkv_transpose2 =
        OpNode("qkv_transpose2", "transpose2")->AsIntermediate();
    auto* qkv_transpose2_out = VarNode("qkv_transpose2_out")
                                   ->assert_is_op_output("transpose2", "Out")
                                   ->assert_is_op_input("reshape2", "X")
                                   ->AsIntermediate();
    auto* qkv_transpose2_xshape =
        VarNode("qkv_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();
    auto* qkv_reshape2 = OpNode("qkv_reshape2", "reshape2")->AsIntermediate();
    auto* qkv_reshape2_out = VarNode("qkv_reshape2_out")
                                 ->assert_is_op_output("reshape2", "Out")
                                 ->assert_is_op_input(mul_type_, "X")
                                 ->AsIntermediate();
    auto* qkv_reshape2_xshape = VarNode("qkv_reshape2_xshape")
                                    ->assert_is_op_output("reshape2", "XShape")
                                    ->AsIntermediate();
    auto* qkv_mul_y =
        VarNode("qkv_mul_y")->assert_is_op_input(mul_type_, "Y")->AsInput();
    auto* qkv_mul = OpNode("qkv_mul", mul_type_)->AsIntermediate();
    auto* qkv_mul_out = VarNode("qkv_mul_out")
                            ->assert_is_op_output(mul_type_, "Out")
                            ->assert_is_op_input("elementwise_add", "X")
                            ->AsIntermediate();
    auto* qkv_add_y = VarNode("qkv_add_y")
                          ->assert_is_op_input("elementwise_add", "Y")
                          ->AsInput();
    auto* qkv_add = OpNode("qkv_add", "elementwise_add")->AsIntermediate();
    auto* qkv_add_out = VarNode("qkv_add_out")
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->AsIntermediate();

    auto* qkv_add_2 = OpNode("qkv_add_2", "elementwise_add")->AsIntermediate();
    auto* qkv_add_2_out = VarNode("qkv_add_2_out")
                              ->assert_is_op_output("elementwise_add", "Out")
                              ->assert_is_op_input("layer_norm", "X")
                              ->AsIntermediate();
    if (norm_before_) {
      qkv_add_2_out->assert_is_op_input("elementwise_add", qkv_ln_2_out_pos_);
    }
    auto* qkv_ln_2_scale = VarNode("qkv_ln_2_scale")
                               ->assert_is_op_input("layer_norm", "Scale")
                               ->AsInput();
    auto* qkv_ln_2_bias = VarNode("qkv_ln_2_bias")
                              ->assert_is_op_input("layer_norm", "Bias")
                              ->AsInput();
    auto* qkv_ln_2 = OpNode("qkv_ln_2", "layer_norm")->AsIntermediate();
    auto* qkv_ln_2_out = VarNode("qkv_ln_2_out")
                             ->assert_is_op_output("layer_norm", "Y")
                             ->assert_is_op_input(mul_type_, "X")
                             ->AsIntermediate();
    if (!norm_before_) {
      qkv_ln_2_out->assert_is_op_input("elementwise_add", qkv_ln_2_out_pos_);
    }
    auto* qkv_ln_2_mean = VarNode("qkv_ln_2_mean")
                              ->assert_is_op_output("layer_norm", "Mean")
                              ->AsIntermediate();
    auto* qkv_ln_2_var = VarNode("qkv_ln_2_var")
                             ->assert_is_op_output("layer_norm", "Variance")
                             ->AsIntermediate();

    auto* qkv_mul_3_y =
        VarNode("qkv_mul_3_y")->assert_is_op_input(mul_type_, "Y")->AsInput();
    auto* qkv_mul_3 = OpNode("qkv_mul_3", mul_type_)->AsIntermediate();
    auto* qkv_mul_3_out = VarNode("qkv_mul_3_out")
                              ->assert_is_op_output(mul_type_, "Out")
                              ->assert_is_op_input("elementwise_add", "X")
                              ->AsIntermediate();
    auto* qkv_add_3_y = VarNode("qkv_add_3_y")
                            ->assert_is_op_input("elementwise_add", "Y")
                            ->AsInput();
    auto* qkv_add_3 = OpNode("qkv_add_3", "elementwise_add")->AsIntermediate();
    auto* qkv_add_3_out = VarNode("qkv_add_3_out")
                              ->assert_is_op_output("elementwise_add", "Out")
                              ->assert_is_op_input(act_type_, "X")
                              ->AsIntermediate();
    auto* qkv_act = OpNode("qkv_act", act_type_)->AsIntermediate();
    auto* qkv_act_out = VarNode("qkv_act_out")
                            ->assert_is_op_output(act_type_, "Out")
                            ->assert_is_op_input(mul_type_, "X")
                            ->AsIntermediate();
    auto* qkv_mul_4_y =
        VarNode("qkv_mul_4_y")->assert_is_op_input(mul_type_, "Y")->AsInput();
    auto* qkv_mul_4 = OpNode("qkv_mul_4", mul_type_)->AsIntermediate();
    auto* qkv_mul_4_out = VarNode("qkv_mul_4_out")
                              ->assert_is_op_output(mul_type_, "Out")
                              ->assert_is_op_input("elementwise_add", "X")
                              ->AsIntermediate();
    auto* qkv_add_4_y = VarNode("qkv_add_4_y")
                            ->assert_is_op_input("elementwise_add", "Y")
                            ->AsInput();
    auto* qkv_add_4 = OpNode("qkv_add_4", "elementwise_add")->AsIntermediate();
    auto* qkv_add_4_out = VarNode("qkv_add_4_out")
                              ->assert_is_op_output("elementwise_add", "Out")
                              ->AsIntermediate();

    auto* qkv_add_5 = OpNode("qkv_add_5", "elementwise_add")->AsIntermediate();
    auto* qkv_add_5_out =
        VarNode("qkv_add_5_out")->assert_is_op_output("elementwise_add", "Out");
    PMNode* qkv_ln_5_scale = nullptr;
    PMNode* qkv_ln_5_bias = nullptr;
    PMNode* qkv_ln_5 = nullptr;
    PMNode* qkv_ln_5_out = nullptr;
    PMNode* qkv_ln_5_mean = nullptr;
    PMNode* qkv_ln_5_var = nullptr;
    if (norm_before_) {
      qkv_add_5_out->AsOutput();
    } else {
      qkv_add_5_out->assert_is_op_input("layer_norm", "X")->AsIntermediate();
      qkv_ln_5_scale = VarNode("qkv_ln_5_scale")
                           ->assert_is_op_input("layer_norm", "Scale")
                           ->AsInput();
      qkv_ln_5_bias = VarNode("qkv_ln_5_bias")
                          ->assert_is_op_input("layer_norm", "Bias")
                          ->AsInput();
      qkv_ln_5 = OpNode("qkv_ln_5", "layer_norm")->AsIntermediate();
      qkv_ln_5_out = VarNode("qkv_ln_5_out")
                         ->assert_is_op_output("layer_norm", "Y")
                         ->AsOutput();
      qkv_ln_5_mean = VarNode("qkv_ln_5_mean")
                          ->assert_is_op_output("layer_norm", "Mean")
                          ->AsIntermediate();
      qkv_ln_5_var = VarNode("qkv_ln_5_var")
                         ->assert_is_op_output("layer_norm", "Variance")
                         ->AsIntermediate();
    }

    // TODO(miaotianxiang): use LinksFrom/LinksTo() instead
    if (norm_before_) {
      std::vector<PMNode*> ln_before_input{
          input, ln_before_scale, ln_before_bias};
      std::vector<PMNode*> ln_before_output{
          ln_before_out, ln_before_mean, ln_before_var};
      ln_before_input >> *ln_before >> ln_before_output;
      *ln_before_out >> *q_mul;
    } else {
      *input >> *q_mul;
    }
    if (with_q_scale_) {
      *q_mul >> *q_mul_out >> *q_add >> *q_add_out >> *q_reshape2 >>
          *q_reshape2_out >> *q_transpose2 >> *q_transpose2_out >> *q_scale >>
          *q_scale_out >> *qk_matmul;
    } else {
      *q_mul >> *q_mul_out >> *q_add >> *q_add_out >> *q_reshape2 >>
          *q_reshape2_out >> *q_transpose2 >> *q_transpose2_out >> *qk_matmul;
    }
    *q_mul_y >> *q_mul;
    *q_add_y >> *q_add;
    *q_reshape2 >> *q_reshape2_xshape;
    *q_transpose2 >> *q_transpose2_xshape;

    if (norm_before_) {
      *ln_before_out >> *k_mul;
    } else {
      *input >> *k_mul;
    }
    *k_mul >> *k_mul_out >> *k_add >> *k_add_out >> *k_reshape2 >>
        *k_reshape2_out >> *k_transpose2 >> *k_transpose2_out >> *qk_matmul;

    *k_mul_y >> *k_mul;
    *k_add_y >> *k_add;
    *k_reshape2 >> *k_reshape2_xshape;
    *k_transpose2 >> *k_transpose2_xshape;

    *qk_matmul >> *qk_matmul_out >> *qk_add >> *qk_add_out >> *qk_softmax >>
        *qk_softmax_out >> *qkv_matmul;
    *qk_mask >> *qk_add;

    if (norm_before_) {
      *ln_before_out >> *v_mul;
    } else {
      *input >> *v_mul;
    }
    *v_mul >> *v_mul_out >> *v_add >> *v_add_out >> *v_reshape2 >>
        *v_reshape2_out >> *v_transpose2 >> *v_transpose2_out >> *qkv_matmul;
    *v_mul_y >> *v_mul;
    *v_add_y >> *v_add;
    *v_reshape2 >> *v_reshape2_xshape;
    *v_transpose2 >> *v_transpose2_xshape;

    *qkv_matmul >> *qkv_matmul_out >> *qkv_transpose2 >> *qkv_transpose2_out >>
        *qkv_reshape2 >> *qkv_reshape2_out >> *qkv_mul >> *qkv_mul_out >>
        *qkv_add >> *qkv_add_out >> *qkv_add_2;
    *qkv_transpose2 >> *qkv_transpose2_xshape;
    *qkv_reshape2 >> *qkv_reshape2_xshape;
    *qkv_mul_y >> *qkv_mul;
    *qkv_add_y >> *qkv_add;

    *input >> *qkv_add_2 >> *qkv_add_2_out >> *qkv_ln_2 >> *qkv_ln_2_out;
    *qkv_ln_2_scale >> *qkv_ln_2;
    *qkv_ln_2_bias >> *qkv_ln_2;
    *qkv_ln_2 >> *qkv_ln_2_mean;
    *qkv_ln_2 >> *qkv_ln_2_var;

    *qkv_ln_2_out >> *qkv_mul_3 >> *qkv_mul_3_out >> *qkv_add_3 >>
        *qkv_add_3_out >> *qkv_act >> *qkv_act_out >> *qkv_mul_4 >>
        *qkv_mul_4_out >> *qkv_add_4 >> *qkv_add_4_out >> *qkv_add_5;
    *qkv_mul_3_y >> *qkv_mul_3;
    *qkv_add_3_y >> *qkv_add_3;
    *qkv_mul_4_y >> *qkv_mul_4;
    *qkv_add_4_y >> *qkv_add_4;

    if (norm_before_) {
      *qkv_add_2_out >> *qkv_add_5 >> *qkv_add_5_out;
    } else {
      *qkv_ln_2_out >> *qkv_add_5 >> *qkv_add_5_out >> *qkv_ln_5 >>
          *qkv_ln_5_out;
      *qkv_ln_5_scale >> *qkv_ln_5;
      *qkv_ln_5_bias >> *qkv_ln_5;
      *qkv_ln_5 >> *qkv_ln_5_mean;
      *qkv_ln_5 >> *qkv_ln_5_var;
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("single_encoder");
    op_desc.SetInput("Inputs", {matched.at("input")->arg()->name});
    op_desc.SetInput("Mask", {matched.at("qk_mask")->arg()->name});
    op_desc.SetInput("FCWeight",
                     {
                         matched.at("q_mul_y")->arg()->name,
                         matched.at("k_mul_y")->arg()->name,
                         matched.at("v_mul_y")->arg()->name,
                         matched.at("qkv_mul_y")->arg()->name,
                         matched.at("qkv_mul_3_y")->arg()->name,
                         matched.at("qkv_mul_4_y")->arg()->name,
                     });
    op_desc.SetInput("FCBias",
                     {
                         matched.at("q_add_y")->arg()->name,
                         matched.at("k_add_y")->arg()->name,
                         matched.at("v_add_y")->arg()->name,
                         matched.at("qkv_add_y")->arg()->name,
                         matched.at("qkv_add_3_y")->arg()->name,
                         matched.at("qkv_add_4_y")->arg()->name,
                     });
    if (norm_before_) {
      op_desc.SetInput("LNScale",
                       {
                           matched.at("ln_before_scale")->arg()->name,
                           matched.at("qkv_ln_2_scale")->arg()->name,
                       });
      op_desc.SetInput("LNBias",
                       {
                           matched.at("ln_before_bias")->arg()->name,
                           matched.at("qkv_ln_2_bias")->arg()->name,
                       });
      op_desc.SetOutput("Outputs", {matched.at("qkv_add_5_out")->arg()->name});
    } else {
      op_desc.SetInput("LNScale",
                       {
                           matched.at("qkv_ln_2_scale")->arg()->name,
                           matched.at("qkv_ln_5_scale")->arg()->name,
                       });
      op_desc.SetInput("LNBias",
                       {
                           matched.at("qkv_ln_2_bias")->arg()->name,
                           matched.at("qkv_ln_5_bias")->arg()->name,
                       });
      op_desc.SetOutput("Outputs", {matched.at("qkv_ln_5_out")->arg()->name});
    }
    // XXX: keep these to fool SubgraphOp::AttachImpl()
    op_desc.SetAttr<int>("sub_block", 0);
    op_desc.SetAttr<std::vector<std::string>>("input_data_names", {});
    op_desc.SetAttr<std::vector<std::string>>("output_data_names", {});

    // extra traits to distill
    auto* reshape_op_info = matched.at("q_reshape2")->stmt()->op_info();
    auto reshape_dim = reshape_op_info->GetAttr<std::vector<int>>("shape");
    // scale attr must be equal to 1 / std::sqrt(size_per_head)
    int size_per_head = reshape_dim[3];
    float scale_val = 0.f;
    if (with_q_scale_) {
      scale_val =
          matched.at("q_scale")->stmt()->op_info()->GetAttr<float>("scale");
    } else {
      scale_val =
          matched.at("qk_matmul")->stmt()->op_info()->GetAttr<float>("alpha");
    }
    float expected_value = 1.f / std::sqrt(size_per_head);
    CHECK(std::abs(expected_value - scale_val) < 1e-6f);
    op_desc.SetAttr<int>("head_num", reshape_dim[2]);
    op_desc.SetAttr<int>("size_per_head", size_per_head);
    op_desc.SetAttr<std::string>("act_type", act_type_);
    op_desc.SetAttr<bool>("norm_before", norm_before_);

    auto fake_subgraph_op = LiteOpRegistry::Global().Create("subgraph");
    auto sub_program_desc = std::make_shared<cpp::ProgramDesc>();
    sub_program_desc->AddBlock<cpp::BlockDesc>();
    static_cast<operators::SubgraphOp*>(fake_subgraph_op.get())
        ->SetProgramDesc(sub_program_desc);
    auto* single_encoder_stmt = matched.at("q_mul")->stmt();
    fake_subgraph_op->Attach(op_desc, single_encoder_stmt->op()->scope());
    fake_subgraph_op->SetValidPlaces(single_encoder_stmt->op()->valid_places());
    single_encoder_stmt->SetOp(fake_subgraph_op);

    std::vector<std::string> froms = {
        "qk_mask",
        "k_mul_y",
        "v_mul_y",
        "qkv_mul_y",
        "qkv_mul_3_y",
        "qkv_mul_4_y",
        "q_add_y",
        "k_add_y",
        "v_add_y",
        "qkv_add_y",
        "qkv_add_3_y",
        "qkv_add_4_y",
        "qkv_ln_2_scale",
        "qkv_ln_2_bias",
    };
    if (norm_before_) {
      froms.push_back("ln_before_scale");
      froms.push_back("ln_before_bias");
      froms.push_back("input");
    } else {
      froms.push_back("qkv_ln_5_scale");
      froms.push_back("qkv_ln_5_bias");
    }
    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), matched.at("q_mul"));
    }
    if (norm_before_) {
      IR_OP_VAR_LINK(matched.at("q_mul"), matched.at("qkv_add_5_out"));
    } else {
      IR_OP_VAR_LINK(matched.at("q_mul"), matched.at("qkv_ln_5_out"));
    }
  }

 private:
  std::string act_type_;
  std::string input_pos_;
  std::string qkv_ln_2_out_pos_;
  std::string matmul_type_;
  std::string mul_type_;
  bool with_q_scale_;
  bool norm_before_;
};

class XPUMultiEncoderFuser {
 public:
  explicit XPUMultiEncoderFuser(const std::string& fc_precision,
                                bool adaptive_seqlen) {
    fc_precision_ = fc_precision;
    adaptive_seqlen_ = adaptive_seqlen;
  }
  bool IsDirectPredecessorOf(Node* op1, Node* op2) {
    for (auto* out : op1->outlinks) {
      for (auto* in : op2->inlinks) {
        if (out == in) return true;
      }
    }
    return false;
  }

  void operator()(SSAGraph* graph) {
    std::vector<Node*> all_encoders;
    for (auto* node : graph->StmtTopologicalOrder()) {
      CHECK(node->IsStmt());
      if (node->stmt()->op_info()->Type() == "single_encoder") {
        all_encoders.push_back(node);
      }
    }
    VLOG(3) << "Found " << all_encoders.size() << " single_encoder";
    if (all_encoders.size() == 0) {
      return;
    }

    // TODO(miaotianxiang): more verification
    const bool norm_before_0 =
        all_encoders[0]->stmt()->op_info()->GetAttr<bool>("norm_before");
    for (size_t i = 0; i < all_encoders.size() - 1; ++i) {
      CHECK(IsDirectPredecessorOf(all_encoders[i], all_encoders[i + 1]));
      const bool norm_before_i =
          all_encoders[i + 1]->stmt()->op_info()->GetAttr<bool>("norm_before");
      CHECK_EQ(norm_before_0, norm_before_i);
    }
    std::string mask_name;
    for (auto* encoder : all_encoders) {
      auto* op_info = encoder->stmt()->op_info();
      if (mask_name.empty()) {
        mask_name = op_info->Input("Mask").front();
      } else {
        // CHECK(mask_name == op_info->Input("Mask").front());
      }
    }

    std::set<const Node*> to_remove;
    Node* first_encoder = all_encoders[0];
    std::string in_name, out_name;
    std::vector<std::string> arg_names{
        "FCWeight", "FCBias", "LNScale", "LNBias"};
    std::map<std::string, std::vector<std::string>> arg_map;
    for (size_t i = 0; i < all_encoders.size(); ++i) {
      Node* cur_encoder = all_encoders[i];
      auto* op_info = cur_encoder->stmt()->op_info();
      for (auto arg_name : arg_names) {
        auto real_names = op_info->Input(arg_name);
        for (auto name : real_names) {
          auto* arg_node = graph->RetrieveArgument(name);
          DirectedLink(arg_node, first_encoder);
          arg_map[arg_name].push_back(name);
        }
      }

      auto* cur_out =
          graph->RetrieveArgument(op_info->Output("Outputs").front());
      if (i == 0) {
        // first encoder
        to_remove.insert(cur_out);
        in_name = op_info->Input("Inputs").front();
        mask_name = op_info->Input("Mask").front();
      } else if (i == all_encoders.size() - 1) {
        // last encoder
        to_remove.insert(cur_encoder);
        DirectedLink(first_encoder, cur_out);
        out_name = op_info->Output("Outputs").front();
      } else {
        to_remove.insert(cur_encoder);
        to_remove.insert(cur_out);
      }
    }
    GraphSafeRemoveNodes(graph, to_remove);

    auto* multi_encoder_stmt = first_encoder->stmt();
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__multi_encoder");
    op_desc.SetInput("Input", {in_name});
    for (auto kv : arg_map) {
      op_desc.SetInput(kv.first, kv.second);
    }
    op_desc.SetInput("Mask", {mask_name});
    op_desc.SetOutput("Output", {out_name});
    op_desc.SetAttr<int>("xpu", 1);
    op_desc.SetAttr<bool>("norm_before", norm_before_0);
    auto* first_encoder_op_info = multi_encoder_stmt->op_info();
    op_desc.SetAttr<int>("head_num",
                         first_encoder_op_info->GetAttr<int>("head_num"));
    op_desc.SetAttr<int>("size_per_head",
                         first_encoder_op_info->GetAttr<int>("size_per_head"));
    op_desc.SetAttr<int>("n_layers", all_encoders.size());
    op_desc.SetAttr<std::string>(
        "act_type", first_encoder_op_info->GetAttr<std::string>("act_type"));
    op_desc.SetAttr<std::string>("precision", fc_precision_);
    op_desc.SetAttr<bool>("adaptive_seqlen", adaptive_seqlen_);

    // q/k/v fusion
    bool enable_qkv_fusion = true;
    if (norm_before_0) {
      enable_qkv_fusion = false;
    }
    op_desc.SetAttr<bool>("enable_qkv_fusion", enable_qkv_fusion);

    auto* scope = multi_encoder_stmt->op()->scope();
    std::vector<float> fc_weight_max(arg_map["FCWeight"].size());
    auto& fc_weight_names = arg_map["FCWeight"];
    for (size_t i = 0; i < fc_weight_names.size(); ++i) {
      if (enable_qkv_fusion && (i % 6 == 0)) {
        // q/k/v FCWeight fusion
        auto* weight_q = scope->FindMutableTensor(fc_weight_names[i]);
        auto* weight_k = scope->FindMutableTensor(fc_weight_names[i + 1]);
        auto* weight_v = scope->FindMutableTensor(fc_weight_names[i + 2]);
        auto weight_q_dims = weight_q->dims();
        auto weight_k_dims = weight_k->dims();
        auto weight_v_dims = weight_v->dims();
        int weight_q_len = weight_q->numel();
        int weight_k_len = weight_k->numel();
        int weight_v_len = weight_v->numel();
        float* weight_q_on_host = weight_q->mutable_data<float>();
        float* weight_k_on_host = weight_k->mutable_data<float>();
        float* weight_v_on_host = weight_v->mutable_data<float>();
        int qkv_len = weight_q_len + weight_k_len + weight_v_len;
        int qkv_offset = 0;
        CHECK_EQ(weight_q_dims[0], weight_k_dims[0]);
        CHECK_EQ(weight_q_dims[0], weight_v_dims[0]);

        // 1. transpose
        std::unique_ptr<float[]> weight_q_trans(new float[weight_q_len]);
        std::unique_ptr<float[]> weight_k_trans(new float[weight_k_len]);
        std::unique_ptr<float[]> weight_v_trans(new float[weight_v_len]);
        std::unique_ptr<float[]> weight_qkv_trans(new float[qkv_len]);
        paddle::lite::xpu::math::Transpose(weight_q_on_host,
                                           weight_q_trans.get(),
                                           weight_q_dims[0],
                                           weight_q_dims[1]);
        paddle::lite::xpu::math::Transpose(weight_k_on_host,
                                           weight_k_trans.get(),
                                           weight_k_dims[0],
                                           weight_k_dims[1]);
        paddle::lite::xpu::math::Transpose(weight_v_on_host,
                                           weight_v_trans.get(),
                                           weight_v_dims[0],
                                           weight_v_dims[1]);

        // 2. concat
        memcpy(weight_qkv_trans.get() + qkv_offset,
               weight_q_trans.get(),
               weight_q_len * sizeof(float));
        qkv_offset += weight_q_len;
        memcpy(weight_qkv_trans.get() + qkv_offset,
               weight_k_trans.get(),
               weight_k_len * sizeof(float));
        qkv_offset += weight_k_len;
        memcpy(weight_qkv_trans.get() + qkv_offset,
               weight_v_trans.get(),
               weight_v_len * sizeof(float));
        qkv_offset += weight_v_len;
        CHECK_EQ(qkv_offset, qkv_len);

        weight_q->Resize(
            {weight_q_dims[1] + weight_k_dims[1] + weight_v_dims[1],
             weight_q_dims[0]});

        // 3. int31 or int16
        float max_f = paddle::lite::xpu::math::FindMaxAbs(
            weight_qkv_trans.get(), qkv_len);
        fc_weight_max[i] = max_f;
        VLOG(3) << "QKV fused FC-" << i << ", weight_max:" << max_f;
        if (fc_precision_ == "int31") {
          memcpy(weight_q->mutable_data<float>(),
                 weight_qkv_trans.get(),
                 qkv_len * sizeof(float));
        } else if (fc_precision_ == "int8") {
          std::unique_ptr<int8_t[]> weight_qkv_trans_int8(new int8_t[qkv_len]);
          paddle::lite::xpu::math::ConvertFP32ToInt8(
              weight_qkv_trans.get(),
              weight_qkv_trans_int8.get(),
              max_f,
              qkv_len);
          memcpy(weight_q->mutable_data<float>(),
                 weight_qkv_trans_int8.get(),
                 qkv_len * sizeof(int8_t));
        } else {
          std::unique_ptr<int16_t[]> weight_qkv_trans_int16(
              new int16_t[qkv_len]);
          paddle::lite::xpu::math::ConvertFP32ToInt16(
              weight_qkv_trans.get(),
              weight_qkv_trans_int16.get(),
              max_f,
              qkv_len);
          memcpy(weight_q->mutable_data<float>(),
                 weight_qkv_trans_int16.get(),
                 qkv_len * sizeof(int16_t));
        }

        continue;
      }

      // no q/k/v fusion
      auto* weight_t = scope->FindMutableTensor(fc_weight_names[i]);
      auto weight_dims = weight_t->dims();
      int weight_len = weight_t->numel();
      float* weight_on_host = weight_t->mutable_data<float>();

      float max_f =
          paddle::lite::xpu::math::FindMaxAbs(weight_on_host, weight_len);
      VLOG(3) << "FC-" << i << ", weight_max:" << max_f;
      // i ranges from 0 to 6*encoder_num, so we need to do i%6 to get relative
      // position in the encoder
      if (fc_precision_ == "int31") {
        // FCs in encoder use int31
        std::unique_ptr<float[]> weight_trans_fp32(new float[weight_len]);
        paddle::lite::xpu::math::Transpose(weight_on_host,
                                           weight_trans_fp32.get(),
                                           weight_dims[0],
                                           weight_dims[1]);

        memcpy(weight_on_host,
               weight_trans_fp32.get(),
               weight_len * sizeof(float));
      } else if (fc_precision_ == "int8") {
        std::unique_ptr<int8_t[]> weight_int8(new int8_t[weight_len]);
        std::unique_ptr<int8_t[]> weight_trans_int8(new int8_t[weight_len]);
        paddle::lite::xpu::math::ConvertFP32ToInt8(
            weight_on_host, weight_int8.get(), max_f, weight_len);
        paddle::lite::xpu::math::Transpose(weight_int8.get(),
                                           weight_trans_int8.get(),
                                           weight_dims[0],
                                           weight_dims[1]);
        memcpy(weight_on_host,
               weight_trans_int8.get(),
               weight_len * sizeof(int8_t));
      } else {
        std::unique_ptr<int16_t[]> weight_int16(new int16_t[weight_len]);
        std::unique_ptr<int16_t[]> weight_trans_int16(new int16_t[weight_len]);
        paddle::lite::xpu::math::ConvertFP32ToInt16(
            weight_on_host, weight_int16.get(), max_f, weight_len);
        paddle::lite::xpu::math::Transpose(weight_int16.get(),
                                           weight_trans_int16.get(),
                                           weight_dims[0],
                                           weight_dims[1]);
        memcpy(weight_on_host,
               weight_trans_int16.get(),
               weight_len * sizeof(int16_t));
      }
      fc_weight_max[i] = max_f;
    }

    auto& fc_bias_names = arg_map["FCBias"];
    for (size_t i = 0; enable_qkv_fusion && i < fc_bias_names.size(); i += 6) {
      // q/k/v FCBias fusion
      VLOG(3) << "Copy bias in QKV fused FC-" << i << ", " << i / 6 << "-"
              << i % 6;
      auto* bias_q = scope->FindMutableTensor(fc_bias_names[i]);
      auto* bias_k = scope->FindMutableTensor(fc_bias_names[i + 1]);
      auto* bias_v = scope->FindMutableTensor(fc_bias_names[i + 2]);
      auto bias_q_dims = bias_q->dims();
      auto bias_k_dims = bias_k->dims();
      auto bias_v_dims = bias_v->dims();
      int bias_q_len = bias_q->numel();
      int bias_k_len = bias_k->numel();
      int bias_v_len = bias_v->numel();
      float* bias_q_on_host = bias_q->mutable_data<float>();
      float* bias_k_on_host = bias_k->mutable_data<float>();
      float* bias_v_on_host = bias_v->mutable_data<float>();
      int qkv_len = bias_q_len + bias_k_len + bias_v_len;
      int qkv_offset = 0;
      CHECK_EQ(bias_q_dims.size(), 1);
      CHECK_EQ(bias_k_dims.size(), 1);
      CHECK_EQ(bias_v_dims.size(), 1);

      std::unique_ptr<float[]> bias_qkv(new float[qkv_len]);
      memcpy(bias_qkv.get() + qkv_offset,
             bias_q_on_host,
             bias_q_len * sizeof(float));
      qkv_offset += bias_q_len;
      memcpy(bias_qkv.get() + qkv_offset,
             bias_k_on_host,
             bias_k_len * sizeof(float));
      qkv_offset += bias_k_len;
      memcpy(bias_qkv.get() + qkv_offset,
             bias_v_on_host,
             bias_v_len * sizeof(float));
      qkv_offset += bias_v_len;
      CHECK_EQ(qkv_offset, qkv_len);

      bias_q->Resize({qkv_len});
      memcpy(bias_q->mutable_data<float>(),
             bias_qkv.get(),
             qkv_len * sizeof(float));
    }

    // TODO(mayang02): we could use attr to store FCWeightMax
    std::string max_name = "encoder_max";
    auto* max_filter_node = graph->NewArgumentNode(max_name);
    max_filter_node->arg()->is_weight = true;
    max_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    DirectedLink(max_filter_node, first_encoder);
    auto* max_filter_tensor = scope->NewTensor(max_name);
    max_filter_tensor->Resize({static_cast<int>(fc_weight_max.size())});
    memcpy(max_filter_tensor->mutable_data<float>(),
           &fc_weight_max[0],
           sizeof(float) * fc_weight_max.size());
    max_filter_tensor->set_precision(paddle::lite_api::PrecisionType::kFloat);
    max_filter_tensor->set_persistable(true);
    op_desc.SetInput("FCWeightMax", {max_name});

    auto multi_encoder_op = LiteOpRegistry::Global().Create(op_desc.Type());
    multi_encoder_op->Attach(op_desc, scope);
    multi_encoder_op->SetValidPlaces(multi_encoder_stmt->op()->valid_places());
    auto kernels =
        multi_encoder_op->CreateKernels(multi_encoder_op->valid_places());
    multi_encoder_stmt->SetOp(multi_encoder_op);
    multi_encoder_stmt->SetKernels(std::move(kernels));

    // remove dangling/useless cast
    Node* stack = nullptr;
    for (auto* node : graph->StmtTopologicalOrder()) {
      CHECK(node->IsStmt());
      if (node->stmt()->op_info()->Type() == "stack") {
        stack = node;
      }
    }
    if (stack) {
      std::set<const Node*> to_remove2;
      Node* stack_out = stack->outlinks.front();
      // avoid modification while traversing
      auto stack_out_outlinks = stack_out->outlinks;
      for (Node* cast : stack_out_outlinks) {
        if (cast->stmt()->op_info()->Type() != "cast") {
          continue;
        }

        Node* cast_out = cast->outlinks.front();
        if (cast_out->outlinks.size() == 0) {
          // dangling cast
          to_remove2.insert(cast);
          to_remove2.insert(cast_out);
          VLOG(3) << "Remove dangling cast [" << cast_out->arg()->name << "]";
        } else if (cast_out->outlinks.size() == 1) {
          // useless cast
          to_remove2.insert(cast);
          to_remove2.insert(cast_out);
          VLOG(3) << "Remove useless cast [" << cast_out->arg()->name << "]";

          auto* multi_encoder = cast_out->outlinks.front();
          DirectedLink(stack_out, multi_encoder);
          UpdateInputs(multi_encoder->stmt()->op().get(),
                       cast_out->arg()->name,
                       stack_out->arg()->name);
          auto update_op_info = *multi_encoder->stmt()->op_info();
          multi_encoder->stmt()->ResetOp(update_op_info, graph->valid_places());
        }
      }
      GraphSafeRemoveNodes(graph, to_remove2);
    }
  }

 private:
  std::string fc_precision_;
  bool adaptive_seqlen_;
};

}  // namespace fusion

class XPUMultiEncoderFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    // TODO(miaotianxiang): backup graph, recover from failed match
    std::vector<std::string> act_types{"gelu", "relu"};
    std::vector<std::string> input_poss{"X", "Y"};
    std::vector<std::string> qkv_ln_2_out_poss{"X", "Y"};
    std::vector<std::string> matmul_types{"matmul", "matmul_v2"};
    std::vector<std::string> mul_types{"mul", "matmul"};
    std::vector<bool> with_q_scales{true, false};
    std::vector<bool> norm_befores{true, false};

    std::string fc_precision;
    bool adaptive_seqlen = false;
#ifdef LITE_WITH_XPU
    // TODO(miaotianxiang): core/mir/*_pass.cc are compiled anyway and need to
    // access TargetWrapperXPU::multi_encoder_precision, but this static member
    // variable in class specialization defined in
    // lite/backends/xpu/target_wrapper.cc is only compiled iff
    // LITE_WITH_XPU==ON. To suppress linkage error, we use
    // #ifdef here. Any better idea?
    auto ctx =
        ContextScheduler::Global().NewContext(TARGET(kXPU))->As<XPUContext>();
    std::string multi_encoder_precision = ctx.MultiEncoderPrecision();
    if (GetStringFromEnv("XPU_ENCODER_PRECISION", "int16") == "int31" ||
        multi_encoder_precision == "int31") {
      fc_precision = "int31";
      VLOG(3) << "Use int31 in XPUMultiEncoderOp, "
              << "multi_encoder_precision=" << multi_encoder_precision;
    } else if (GetStringFromEnv("XPU_ENCODER_PRECISION", "int16") == "int8" ||
               multi_encoder_precision == "int8") {
      fc_precision = "int8";
      VLOG(3) << "Use int8 in XPUMultiEncoderOp, "
              << "multi_encoder_precision=" << multi_encoder_precision;
    } else {
      fc_precision = "int16";
      VLOG(3) << "Use int16 in XPUMultiEncoderOp, "
              << "multi_encoder_precision=" << multi_encoder_precision;
    }
    adaptive_seqlen = ctx.MultiEncoderAdaptiveSeqlen();
#endif

    for (auto& act_type : act_types) {
      for (auto& input_pos : input_poss) {
        for (auto& qkv_ln_2_out_pos : qkv_ln_2_out_poss) {
          for (auto& matmul_type : matmul_types) {
            for (auto& mul_type : mul_types) {
              for (auto with_q_scale : with_q_scales) {
                for (auto norm_before : norm_befores) {
                  fusion::XPUSingleEncoderFuser single_encoder_fuser(
                      act_type,
                      input_pos,
                      qkv_ln_2_out_pos,
                      matmul_type,
                      mul_type,
                      with_q_scale,
                      norm_before);
                  single_encoder_fuser(graph.get());
                  fusion::XPUMultiEncoderFuser multi_encoder_fuser(
                      fc_precision, adaptive_seqlen);
                  multi_encoder_fuser(graph.get());
                }
              }
            }
          }
        }
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__multi_encoder_fuse_pass,
                  paddle::lite::mir::XPUMultiEncoderFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__multi_encoder");
