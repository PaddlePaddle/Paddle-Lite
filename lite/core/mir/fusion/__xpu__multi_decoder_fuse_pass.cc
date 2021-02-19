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

class XPUSingleDecoderFuser : public FuseBase {
 public:
  explicit XPUSingleDecoderFuser(const std::string& act_type = "gelu",
                                 const std::string& input_pos = "Y",
                                 const std::string& matmul_type = "matmul",
                                 const std::string& mul_type = "mul",
                                 bool with_q_scale = true)
      : act_type_(act_type),
        input_pos_(input_pos),
        matmul_type_(matmul_type),
        mul_type_(mul_type),
        with_q_scale_(with_q_scale) {}

  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("layer_norm", "X")
                      ->assert_is_op_input("elementwise_add", input_pos_)
                      ->AsInput();

    auto* pre_ln_scale = VarNode("pre_ln_scale")
                             ->assert_is_op_input("layer_norm", "Scale")
                             ->AsInput();
    auto* pre_ln_bias = VarNode("pre_ln_bias")
                            ->assert_is_op_input("layer_norm", "Bias")
                            ->AsInput();
    auto* pre_ln = OpNode("pre_ln", "layer_norm");
    auto* pre_ln_out = VarNode("pre_ln_out")
                           ->assert_is_op_output("layer_norm", "Y")
                           ->assert_is_op_input(mul_type_, "X")
                           ->AsIntermediate();
    auto* pre_ln_mean = VarNode("pre_ln_mean")
                            ->assert_is_op_output("layer_norm", "Mean")
                            ->AsIntermediate();
    auto* pre_ln_var = VarNode("pre_ln_var")
                           ->assert_is_op_output("layer_norm", "Variance")
                           ->AsIntermediate();

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
      q_scale = OpNode("q_scale", "scale")
                    ->assert_op_attr<float>("scale", 0.125)
                    ->AsIntermediate();
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
                                 ->assert_is_op_nth_input("concat", "X", 1)
                                 ->AsIntermediate();
    auto* k_transpose2_xshape =
        VarNode("k_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();
    auto* k_concat_in0 = VarNode("k_concat_in0")
                             ->assert_is_op_nth_input("concat", "X", 0)
                             ->AsInput();
    auto* k_concat = OpNode("k_concat", "concat")->AsIntermediate();
    auto* k_concat_out = VarNode("k_concat_out")
                             ->assert_is_op_output("concat", "Out")
                             ->assert_is_op_input("matmul", "Y")
                             ->AsOutput();

    PMNode* qk_matmul = nullptr;
    if (with_q_scale_) {
      qk_matmul = OpNode("qk_matmul", "matmul")->AsIntermediate();
    } else {
      qk_matmul = OpNode("qk_matmul", "matmul")
                      ->assert_op_attr<float>("alpha", 0.125)
                      ->AsIntermediate();
    }
    auto* qk_matmul_out = VarNode("qk_matmul_out")
                              ->assert_is_op_output("matmul", "Out")
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
                                 ->assert_is_op_nth_input("concat", "X", 1)
                                 ->AsIntermediate();
    auto* v_transpose2_xshape =
        VarNode("v_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();
    auto* v_concat_in0 = VarNode("v_concat_in0")
                             ->assert_is_op_nth_input("concat", "X", 0)
                             ->AsInput();
    auto* v_concat = OpNode("v_concat", "concat")->AsIntermediate();
    auto* v_concat_out = VarNode("v_concat_out")
                             ->assert_is_op_output("concat", "Out")
                             ->assert_is_op_input(matmul_type_, "Y")
                             ->AsOutput();

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
                              ->assert_is_op_input("elementwise_add", "X")
                              ->AsIntermediate();
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
                              ->assert_is_op_input("reshape2", "X")
                              ->AsIntermediate();

    auto* qkv_reshape2_3 =
        OpNode("qkv_reshape2_3", "reshape2")->AsIntermediate();
    auto* qkv_reshape2_3_out = VarNode("qkv_reshape2_3_out")
                                   ->assert_is_op_output("reshape2", "Out")
                                   ->assert_is_op_input("transpose2", "X")
                                   ->AsIntermediate();
    auto* qkv_reshape2_3_xshape =
        VarNode("qkv_reshape2_3_xshape")
            ->assert_is_op_output("reshape2", "XShape")
            ->AsIntermediate();
    auto* qkv_transpose2_3 =
        OpNode("qkv_transpose2_3", "transpose2")->AsIntermediate();
    auto* qkv_transpose2_3_out = VarNode("qkv_transpose2_3_out")
                                     ->assert_is_op_output("transpose2", "Out")
                                     ->assert_is_op_input("matmul", "X")
                                     ->AsIntermediate();
    auto* qkv_transpose2_3_xshape =
        VarNode("qkv_transpose2_3_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();

    auto* sec_k =
        VarNode("sec_k")->assert_is_op_input("matmul", "Y")->AsInput();
    PMNode* sec_qk_matmul = nullptr;
    if (with_q_scale_) {
      sec_qk_matmul = OpNode("sec_qk_matmul", "matmul")->AsIntermediate();
    } else {
      sec_qk_matmul = OpNode("sec_qk_matmul", "matmul")
                          ->assert_op_attr<float>("alpha", 0.125)
                          ->AsIntermediate();
    }
    auto* sec_qk_matmul_out = VarNode("sec_qk_matmul_out")
                                  ->assert_is_op_output("matmul", "Out")
                                  ->assert_is_op_input("elementwise_add", "X")
                                  ->AsIntermediate();
    auto* sec_qk_mask = VarNode("sec_qk_mask")
                            ->assert_is_op_input("elementwise_add", "Y")
                            ->AsInput();
    auto* sec_qk_add =
        OpNode("sec_qk_add", "elementwise_add")->AsIntermediate();
    auto* sec_qk_add_out = VarNode("sec_qk_add_out")
                               ->assert_is_op_output("elementwise_add", "Out")
                               ->assert_is_op_input("softmax", "X")
                               ->AsIntermediate();
    auto* sec_qk_softmax =
        OpNode("sec_qk_softmax", "softmax")->AsIntermediate();
    auto* sec_qk_softmax_out = VarNode("sec_qk_softmax_out")
                                   ->assert_is_op_output("softmax", "Out")
                                   ->AsIntermediate();

    auto* sec_v =
        VarNode("sec_v")->assert_is_op_input(matmul_type_, "Y")->AsInput();
    auto* sec_qkv_matmul =
        OpNode("sec_qkv_matmul", matmul_type_)->AsIntermediate();
    auto* sec_qkv_matmul_out = VarNode("sec_qkv_matmul_out")
                                   ->assert_is_op_output(matmul_type_, "Out")
                                   ->assert_is_op_input("transpose2", "X")
                                   ->AsIntermediate();

    auto* sec_qkv_transpose2 =
        OpNode("sec_qkv_transpose2", "transpose2")->AsIntermediate();
    auto* sec_qkv_transpose2_out =
        VarNode("sec_qkv_transpose2_out")
            ->assert_is_op_output("transpose2", "Out")
            ->assert_is_op_input("reshape2", "X")
            ->AsIntermediate();
    auto* sec_qkv_transpose2_xshape =
        VarNode("sec_qkv_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();
    auto* sec_qkv_reshape2 =
        OpNode("sec_qkv_reshape2", "reshape2")->AsIntermediate();
    auto* sec_qkv_reshape2_out = VarNode("sec_qkv_reshape2_out")
                                     ->assert_is_op_output("reshape2", "Out")
                                     ->assert_is_op_input(mul_type_, "X")
                                     ->AsIntermediate();
    auto* sec_qkv_reshape2_xshape =
        VarNode("sec_qkv_reshape2_xshape")
            ->assert_is_op_output("reshape2", "XShape")
            ->AsIntermediate();
    auto* sec_qkv_mul_y =
        VarNode("sec_qkv_mul_y")->assert_is_op_input(mul_type_, "Y")->AsInput();
    auto* sec_qkv_mul = OpNode("sec_qkv_mul", mul_type_)->AsIntermediate();
    auto* sec_qkv_mul_out = VarNode("sec_qkv_mul_out")
                                ->assert_is_op_output(mul_type_, "Out")
                                ->assert_is_op_input("elementwise_add", "X")
                                ->AsIntermediate();
    auto* sec_qkv_add_y = VarNode("sec_qkv_add_y")
                              ->assert_is_op_input("elementwise_add", "Y")
                              ->AsInput();
    auto* sec_qkv_add =
        OpNode("sec_qkv_add", "elementwise_add")->AsIntermediate();
    auto* sec_qkv_add_out = VarNode("sec_qkv_add_out")
                                ->assert_is_op_output("elementwise_add", "Out")
                                ->AsIntermediate();

    auto* sec_qkv_add_2 =
        OpNode("sec_qkv_add_2", "elementwise_add")->AsIntermediate();
    auto* sec_qkv_add_2_out =
        VarNode("sec_qkv_add_2_out")
            ->assert_is_op_output("elementwise_add", "Out")
            ->assert_is_op_input("layer_norm", "X")
            ->assert_is_op_input("elementwise_add", "X")
            ->AsIntermediate();

    auto* sec_qkv_ln_2_scale = VarNode("sec_qkv_ln_2_scale")
                                   ->assert_is_op_input("layer_norm", "Scale")
                                   ->AsInput();
    auto* sec_qkv_ln_2_bias = VarNode("sec_qkv_ln_2_bias")
                                  ->assert_is_op_input("layer_norm", "Bias")
                                  ->AsInput();
    auto* sec_qkv_ln_2 = OpNode("sec_qkv_ln_2", "layer_norm")->AsIntermediate();
    auto* sec_qkv_ln_2_out = VarNode("sec_qkv_ln_2_out")
                                 ->assert_is_op_output("layer_norm", "Y")
                                 ->assert_is_op_input(mul_type_, "X")
                                 ->AsIntermediate();
    auto* sec_qkv_ln_2_mean = VarNode("sec_qkv_ln_2_mean")
                                  ->assert_is_op_output("layer_norm", "Mean")
                                  ->AsIntermediate();
    auto* sec_qkv_ln_2_var = VarNode("sec_qkv_ln_2_var")
                                 ->assert_is_op_output("layer_norm", "Variance")
                                 ->AsIntermediate();

    auto* sec_qkv_mul_3_y = VarNode("sec_qkv_mul_3_y")
                                ->assert_is_op_input(mul_type_, "Y")
                                ->AsInput();
    auto* sec_qkv_mul_3 = OpNode("sec_qkv_mul_3", mul_type_)->AsIntermediate();
    auto* sec_qkv_mul_3_out = VarNode("sec_qkv_mul_3_out")
                                  ->assert_is_op_output(mul_type_, "Out")
                                  ->assert_is_op_input("elementwise_add", "X")
                                  ->AsIntermediate();
    auto* sec_qkv_add_3_y = VarNode("sec_qkv_add_3_y")
                                ->assert_is_op_input("elementwise_add", "Y")
                                ->AsInput();
    auto* sec_qkv_add_3 =
        OpNode("sec_qkv_add_3", "elementwise_add")->AsIntermediate();
    auto* sec_qkv_add_3_out =
        VarNode("sec_qkv_add_3_out")
            ->assert_is_op_output("elementwise_add", "Out")
            ->assert_is_op_input(act_type_, "X")
            ->AsIntermediate();
    auto* sec_qkv_act = OpNode("sec_qkv_act", act_type_)->AsIntermediate();
    auto* sec_qkv_act_out = VarNode("sec_qkv_act_out")
                                ->assert_is_op_output(act_type_, "Out")
                                ->assert_is_op_input(mul_type_, "X")
                                ->AsIntermediate();
    auto* sec_qkv_mul_4_y = VarNode("sec_qkv_mul_4_y")
                                ->assert_is_op_input(mul_type_, "Y")
                                ->AsInput();
    auto* sec_qkv_mul_4 = OpNode("sec_qkv_mul_4", mul_type_)->AsIntermediate();
    auto* sec_qkv_mul_4_out = VarNode("sec_qkv_mul_4_out")
                                  ->assert_is_op_output(mul_type_, "Out")
                                  ->assert_is_op_input("elementwise_add", "X")
                                  ->AsIntermediate();
    auto* sec_qkv_add_4_y = VarNode("sec_qkv_add_4_y")
                                ->assert_is_op_input("elementwise_add", "Y")
                                ->AsInput();
    auto* sec_qkv_add_4 =
        OpNode("sec_qkv_add_4", "elementwise_add")->AsIntermediate();
    auto* sec_qkv_add_4_out =
        VarNode("sec_qkv_add_4_out")
            ->assert_is_op_output("elementwise_add", "Out")
            ->AsIntermediate();

    auto* sec_qkv_add_5 =
        OpNode("sec_qkv_add_5", "elementwise_add")->AsIntermediate();
    auto* sec_qkv_add_5_out =
        VarNode("sec_qkv_add_5_out")
            ->assert_is_op_output("elementwise_add", "Out")
            ->AsOutput();

    *input >> *pre_ln >> *pre_ln_out;
    *pre_ln_scale >> *pre_ln;
    *pre_ln_bias >> *pre_ln;
    *pre_ln >> *pre_ln_mean;
    *pre_ln >> *pre_ln_var;

    if (with_q_scale_) {
      *pre_ln_out >> *q_mul >> *q_mul_out >> *q_add >> *q_add_out >>
          *q_reshape2 >> *q_reshape2_out >> *q_transpose2 >>
          *q_transpose2_out >> *q_scale >> *q_scale_out >> *qk_matmul;
    } else {
      *pre_ln_out >> *q_mul >> *q_mul_out >> *q_add >> *q_add_out >>
          *q_reshape2 >> *q_reshape2_out >> *q_transpose2 >>
          *q_transpose2_out >> *qk_matmul;
    }
    *q_mul_y >> *q_mul;
    *q_add_y >> *q_add;
    *q_reshape2 >> *q_reshape2_xshape;
    *q_transpose2 >> *q_transpose2_xshape;

    *pre_ln_out >> *k_mul >> *k_mul_out >> *k_add >> *k_add_out >>
        *k_reshape2 >> *k_reshape2_out >> *k_transpose2 >> *k_transpose2_out >>
        *k_concat >> *k_concat_out >> *qk_matmul;
    *k_mul_y >> *k_mul;
    *k_add_y >> *k_add;
    *k_reshape2 >> *k_reshape2_xshape;
    *k_transpose2 >> *k_transpose2_xshape;
    *k_concat_in0 >> *k_concat;

    *qk_matmul >> *qk_matmul_out >> *qk_softmax >> *qk_softmax_out >>
        *qkv_matmul;

    *pre_ln_out >> *v_mul >> *v_mul_out >> *v_add >> *v_add_out >>
        *v_reshape2 >> *v_reshape2_out >> *v_transpose2 >> *v_transpose2_out >>
        *v_concat >> *v_concat_out >> *qkv_matmul;
    *v_mul_y >> *v_mul;
    *v_add_y >> *v_add;
    *v_reshape2 >> *v_reshape2_xshape;
    *v_transpose2 >> *v_transpose2_xshape;
    *v_concat_in0 >> *v_concat;

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
        *qkv_add_3_out >> *qkv_reshape2_3 >> *qkv_reshape2_3_out >>
        *qkv_transpose2_3 >> *qkv_transpose2_3_out;
    *qkv_mul_3_y >> *qkv_mul_3;
    *qkv_add_3_y >> *qkv_add_3;
    *qkv_reshape2_3 >> *qkv_reshape2_3_xshape;
    *qkv_transpose2_3 >> *qkv_transpose2_3_xshape;

    *qkv_transpose2_3_out >> *sec_qk_matmul >> *sec_qk_matmul_out >>
        *sec_qk_add >> *sec_qk_add_out >> *sec_qk_softmax >>
        *sec_qk_softmax_out;
    *sec_k >> *sec_qk_matmul;
    *sec_qk_mask >> *sec_qk_add;
    *sec_qk_softmax_out >> *sec_qkv_matmul >> *sec_qkv_matmul_out >>
        *sec_qkv_transpose2 >> *sec_qkv_transpose2_out >> *sec_qkv_reshape2 >>
        *sec_qkv_reshape2_out;
    *sec_v >> *sec_qkv_matmul;
    *sec_qkv_transpose2 >> *sec_qkv_transpose2_xshape;
    *sec_qkv_reshape2 >> *sec_qkv_reshape2_xshape;
    *sec_qkv_reshape2_out >> *sec_qkv_mul >> *sec_qkv_mul_out >> *sec_qkv_add >>
        *sec_qkv_add_out >> *sec_qkv_add_2 >> *sec_qkv_add_2_out;
    *sec_qkv_mul_y >> *sec_qkv_mul;
    *sec_qkv_add_y >> *sec_qkv_add;
    *qkv_add_2_out >> *sec_qkv_add_2;

    *sec_qkv_add_2_out >> *sec_qkv_ln_2 >> *sec_qkv_ln_2_out;
    *sec_qkv_ln_2_scale >> *sec_qkv_ln_2;
    *sec_qkv_ln_2_bias >> *sec_qkv_ln_2;
    *sec_qkv_ln_2 >> *sec_qkv_ln_2_mean;
    *sec_qkv_ln_2 >> *sec_qkv_ln_2_var;

    *sec_qkv_ln_2_out >> *sec_qkv_mul_3 >> *sec_qkv_mul_3_out >>
        *sec_qkv_add_3 >> *sec_qkv_add_3_out >> *sec_qkv_act >>
        *sec_qkv_act_out;
    *sec_qkv_mul_3_y >> *sec_qkv_mul_3;
    *sec_qkv_add_3_y >> *sec_qkv_add_3;
    *sec_qkv_act_out >> *sec_qkv_mul_4 >> *sec_qkv_mul_4_out >>
        *sec_qkv_add_4 >> *sec_qkv_add_4_out >> *sec_qkv_add_5 >>
        *sec_qkv_add_5_out;
    *sec_qkv_mul_4_y >> *sec_qkv_mul_4;
    *sec_qkv_add_4_y >> *sec_qkv_add_4;
    *sec_qkv_add_2_out >> *sec_qkv_add_5;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("single_decoder");
    op_desc.SetInput("Inputs", {matched.at("input")->arg()->name});
    op_desc.SetInput("Mask", {matched.at("sec_qk_mask")->arg()->name});
    op_desc.SetInput("KCache",
                     {
                         matched.at("k_concat_in0")->arg()->name,
                         matched.at("sec_k")->arg()->name,
                     });
    op_desc.SetInput("VCache",
                     {
                         matched.at("v_concat_in0")->arg()->name,
                         matched.at("sec_v")->arg()->name,
                     });
    op_desc.SetInput("FCWeight",
                     {
                         matched.at("q_mul_y")->arg()->name,
                         matched.at("k_mul_y")->arg()->name,
                         matched.at("v_mul_y")->arg()->name,
                         matched.at("qkv_mul_y")->arg()->name,
                         matched.at("qkv_mul_3_y")->arg()->name,
                         matched.at("sec_qkv_mul_y")->arg()->name,
                         matched.at("sec_qkv_mul_3_y")->arg()->name,
                         matched.at("sec_qkv_mul_4_y")->arg()->name,
                     });
    op_desc.SetInput("FCBias",
                     {
                         matched.at("q_add_y")->arg()->name,
                         matched.at("k_add_y")->arg()->name,
                         matched.at("v_add_y")->arg()->name,
                         matched.at("qkv_add_y")->arg()->name,
                         matched.at("qkv_add_3_y")->arg()->name,
                         matched.at("sec_qkv_add_y")->arg()->name,
                         matched.at("sec_qkv_add_3_y")->arg()->name,
                         matched.at("sec_qkv_add_4_y")->arg()->name,
                     });
    op_desc.SetInput("LNScale",
                     {
                         matched.at("pre_ln_scale")->arg()->name,
                         matched.at("qkv_ln_2_scale")->arg()->name,
                         matched.at("sec_qkv_ln_2_scale")->arg()->name,
                     });
    op_desc.SetInput("LNBias",
                     {
                         matched.at("pre_ln_bias")->arg()->name,
                         matched.at("qkv_ln_2_bias")->arg()->name,
                         matched.at("sec_qkv_ln_2_bias")->arg()->name,
                     });
    op_desc.SetOutput("Outputs",
                      {
                          matched.at("sec_qkv_add_5_out")->arg()->name,
                      });
    op_desc.SetOutput("KCacheOutputs",
                      {
                          matched.at("k_concat_out")->arg()->name,
                      });
    op_desc.SetOutput("VCacheOutputs",
                      {
                          matched.at("v_concat_out")->arg()->name,
                      });
    // XXX: keep these to fool SubgraphOp::AttachImpl()
    op_desc.SetAttr<int>("sub_block", 0);
    op_desc.SetAttr<std::vector<std::string>>("input_data_names", {});
    op_desc.SetAttr<std::vector<std::string>>("output_data_names", {});

    // extra traits to distill
    auto* reshape_op_info = matched.at("q_reshape2")->stmt()->op_info();
    auto reshape_dim = reshape_op_info->GetAttr<std::vector<int>>("shape");
    op_desc.SetAttr<int>("head_num", reshape_dim[2]);
    op_desc.SetAttr<int>("size_per_head", reshape_dim[3]);
    op_desc.SetAttr<std::string>("act_type", act_type_);

    auto fake_subgraph_op = LiteOpRegistry::Global().Create("subgraph");
    auto sub_program_desc = std::make_shared<cpp::ProgramDesc>();
    sub_program_desc->AddBlock<cpp::BlockDesc>();
    static_cast<operators::SubgraphOp*>(fake_subgraph_op.get())
        ->SetProgramDesc(sub_program_desc);
    auto* single_decoder_stmt = matched.at("pre_ln")->stmt();
    /// TODO: from here
    fake_subgraph_op->Attach(op_desc, single_decoder_stmt->op()->scope());
    fake_subgraph_op->SetValidPlaces(single_decoder_stmt->op()->valid_places());
    single_decoder_stmt->SetOp(fake_subgraph_op);

    std::vector<std::string> froms = {
        "sec_qk_mask",       "k_concat_in0",    "sec_k",
        "v_concat_in0",      "sec_v",           "q_mul_y",
        "k_mul_y",           "v_mul_y",         "qkv_mul_y",
        "qkv_mul_3_y",       "sec_qkv_mul_y",   "sec_qkv_mul_3_y",
        "sec_qkv_mul_4_y",   "q_add_y",         "k_add_y",
        "v_add_y",           "qkv_add_y",       "qkv_add_3_y",
        "sec_qkv_add_y",     "sec_qkv_add_3_y", "sec_qkv_add_4_y",
        "qkv_ln_2_scale",    "qkv_ln_2_bias",   "sec_qkv_ln_2_scale",
        "sec_qkv_ln_2_bias",
    };
    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), matched.at("pre_ln"));
    }
    IR_OP_VAR_LINK(matched.at("pre_ln"), matched.at("sec_qkv_add_5_out"));
    IR_OP_VAR_LINK(matched.at("pre_ln"), matched.at("k_concat_out"));
    IR_OP_VAR_LINK(matched.at("pre_ln"), matched.at("v_concat_out"));
  }

 private:
  std::string act_type_;
  std::string input_pos_;
  std::string matmul_type_;
  std::string mul_type_;
  bool with_q_scale_;
};

class XPUMultiDecoderFuser {
 public:
  explicit XPUMultiDecoderFuser(const std::string& fc_precision)
      : fc_precision_(fc_precision) {}

  bool IsDirectPredecessorOf(Node* op1, Node* op2) {
    for (auto* out : op1->outlinks) {
      for (auto* in : op2->inlinks) {
        if (out == in) return true;
      }
    }
    return false;
  }

  void operator()(SSAGraph* graph) {
    std::vector<Node*> all_decoders;
    for (auto* node : graph->StmtTopologicalOrder()) {
      CHECK(node->IsStmt());
      if (node->stmt()->op_info()->Type() == "single_decoder") {
        all_decoders.push_back(node);
      }
    }
    VLOG(3) << "Found " << all_decoders.size() << " single_decoder";
    if (all_decoders.size() == 0) {
      return;
    }

    // TODO(miaotianxiang): more verification
    for (size_t i = 0; i < all_decoders.size() - 1; ++i) {
      CHECK(IsDirectPredecessorOf(all_decoders[i], all_decoders[i + 1]));
    }
    std::string mask_name;
    for (auto* decoder : all_decoders) {
      auto* op_info = decoder->stmt()->op_info();
      if (mask_name.empty()) {
        mask_name = op_info->Input("Mask").front();
      } else {
        // CHECK(mask_name == op_info->Input("Mask").front());
      }
    }

    std::set<const Node*> to_remove;
    Node* first_decoder = all_decoders[0];
    std::string in_name, out_name;
    std::vector<std::string> arg_names{
        "KCache", "VCache", "FCWeight", "FCBias", "LNScale", "LNBias"};
    std::map<std::string, std::vector<std::string>> arg_map;
    std::vector<std::string> cache_out_names{"KCacheOutputs", "VCacheOutputs"};
    std::map<std::string, std::vector<std::string>> cache_out_map;
    for (size_t i = 0; i < all_decoders.size(); ++i) {
      Node* cur_decoder = all_decoders[i];
      auto* op_info = cur_decoder->stmt()->op_info();
      for (auto arg_name : arg_names) {
        auto real_names = op_info->Input(arg_name);
        for (auto name : real_names) {
          auto* arg_node = graph->RetrieveArgument(name);
          DirectedLink(arg_node, first_decoder);
          arg_map[arg_name].push_back(name);
        }
      }

      for (auto cache_out_name : cache_out_names) {
        auto real_names = op_info->Output(cache_out_name);
        for (auto name : real_names) {
          auto* cur_cache_out_node = graph->RetrieveArgument(name);
          DirectedLink(first_decoder, cur_cache_out_node);
          cache_out_map[cache_out_name].push_back(name);
        }
      }

      auto* cur_out =
          graph->RetrieveArgument(op_info->Output("Outputs").front());
      if (i == 0) {
        // first decoder
        to_remove.insert(cur_out);
        in_name = op_info->Input("Inputs").front();
        mask_name = op_info->Input("Mask").front();
      } else if (i == all_decoders.size() - 1) {
        // last decoder
        to_remove.insert(cur_decoder);
        DirectedLink(first_decoder, cur_out);
        out_name = op_info->Output("Outputs").front();
      } else {
        to_remove.insert(cur_decoder);
        to_remove.insert(cur_out);
      }
    }
    GraphSafeRemoveNodes(graph, to_remove);

    auto* multi_decoder_stmt = first_decoder->stmt();
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__multi_decoder");
    op_desc.SetInput("Input", {in_name});
    for (auto kv : arg_map) {
      op_desc.SetInput(kv.first, kv.second);
    }
    op_desc.SetInput("Mask", {mask_name});
    op_desc.SetOutput("Output", {out_name});
    for (auto kv : cache_out_map) {
      op_desc.SetOutput(kv.first, kv.second);
    }
    op_desc.SetAttr<int>("xpu", 1);
    auto* first_decoder_op_info = multi_decoder_stmt->op_info();
    op_desc.SetAttr<int>("head_num",
                         first_decoder_op_info->GetAttr<int>("head_num"));
    op_desc.SetAttr<int>("size_per_head",
                         first_decoder_op_info->GetAttr<int>("size_per_head"));
    op_desc.SetAttr<int>("n_layers", all_decoders.size());
    op_desc.SetAttr<std::string>(
        "act_type", first_decoder_op_info->GetAttr<std::string>("act_type"));
    op_desc.SetAttr<std::string>("precision", fc_precision_);

    // q/k/v fusion
    bool enable_qkv_fusion = false;
    op_desc.SetAttr<bool>("enable_qkv_fusion", enable_qkv_fusion);

    auto* scope = multi_decoder_stmt->op()->scope();
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
      // i ranges from 0 to 6*decoder_num, so we need to do i%6 to get relative
      // position in the decoder
      if (fc_precision_ == "int31") {
        // FCs in decoder use int31
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

    std::string max_name = "decoder_max";
    auto* max_filter_node = graph->NewArgumentNode(max_name);
    max_filter_node->arg()->is_weight = true;
    max_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    DirectedLink(max_filter_node, first_decoder);
    auto* max_filter_tensor = scope->NewTensor(max_name);
    max_filter_tensor->Resize({static_cast<int>(fc_weight_max.size())});
    memcpy(max_filter_tensor->mutable_data<float>(),
           &fc_weight_max[0],
           sizeof(float) * fc_weight_max.size());
    op_desc.SetInput("FCWeightMax", {max_name});

    auto multi_decoder_op = LiteOpRegistry::Global().Create(op_desc.Type());
    multi_decoder_op->Attach(op_desc, scope);
    multi_decoder_op->SetValidPlaces(multi_decoder_stmt->op()->valid_places());
    auto kernels =
        multi_decoder_op->CreateKernels(multi_decoder_op->valid_places());
    multi_decoder_stmt->SetOp(multi_decoder_op);
    multi_decoder_stmt->SetKernels(std::move(kernels));

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

          auto* multi_decoder = cast_out->outlinks.front();
          DirectedLink(stack_out, multi_decoder);
          UpdateInputs(multi_decoder->stmt()->op().get(),
                       cast_out->arg()->name,
                       stack_out->arg()->name);
          auto update_op_info = *multi_decoder->stmt()->op_info();
          multi_decoder->stmt()->ResetOp(update_op_info, graph->valid_places());
        }
      }
      GraphSafeRemoveNodes(graph, to_remove2);
    }
  }

 private:
  std::string fc_precision_;
};

}  // namespace fusion

class XPUMultiDecoderFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    // TODO(miaotianxiang): backup graph, recover from failed match
    std::vector<std::string> act_types{"relu"};
    std::vector<std::string> input_poss{"X"};
    std::vector<std::string> matmul_types{"matmul_v2"};
    std::vector<std::string> mul_types{"matmul"};
    std::vector<bool> with_q_scales{false};

    std::string fc_precision;
#ifdef LITE_WITH_XPU
    // TODO(miaotianxiang): core/mir/*_pass.cc are compiled anyway and need to
    // access TargetWrapperXPU::multi_decoder_precision, but this static member
    // variable in class specialization defined in
    // lite/backends/xpu/target_wrapper.cc is only compiled iff
    // LITE_WITH_XPU==ON. To suppress linkage error, we use
    // #ifdef here. Any better idea?
    if (GetStringFromEnv("XPU_ENCODER_PRECISION", "int16") == "int31" ||
        lite::TargetWrapperXPU::multi_decoder_precision == "int31") {
      fc_precision = "int31";
      VLOG(3) << "Use int31 in XPUMultiDecoderOp, "
              << "lite::TargetWrapperXPU::multi_decoder_precision="
              << lite::TargetWrapperXPU::multi_decoder_precision;
    } else if (GetStringFromEnv("XPU_ENCODER_PRECISION", "int16") == "int8" ||
               lite::TargetWrapperXPU::multi_decoder_precision == "int8") {
      fc_precision = "int8";
      VLOG(3) << "Use int8 in XPUMultiDecoderOp, "
              << "lite::TargetWrapperXPU::multi_decoder_precision="
              << lite::TargetWrapperXPU::multi_decoder_precision;
    } else {
      fc_precision = "int16";
      VLOG(3) << "Use int16 in XPUMultiDecoderOp, "
              << "lite::TargetWrapperXPU::multi_decoder_precision="
              << lite::TargetWrapperXPU::multi_decoder_precision;
    }
#endif

    for (auto& act_type : act_types) {
      for (auto& input_pos : input_poss) {
        for (auto& matmul_type : matmul_types) {
          for (auto& mul_type : mul_types) {
            for (auto with_q_scale : with_q_scales) {
              fusion::XPUSingleDecoderFuser single_decoder_fuser(
                  act_type, input_pos, matmul_type, mul_type, with_q_scale);
              single_decoder_fuser(graph.get());
              fusion::XPUMultiDecoderFuser multi_decoder_fuser(fc_precision);
              multi_decoder_fuser(graph.get());
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

REGISTER_MIR_PASS(__xpu__multi_decoder_fuse_pass,
                  paddle::lite::mir::XPUMultiDecoderFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__multi_decoder");
