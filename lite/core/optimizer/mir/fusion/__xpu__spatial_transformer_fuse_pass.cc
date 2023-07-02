// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

static std::vector<int> IntVec2DTo1D(const std::vector<std::vector<int>>& vec) {
  std::vector<int> res;
  for (const auto& v : vec) {
    for (const auto& ele : v) {
      res.emplace_back(ele);
    }
  }
  return res;
}

class SpatialTransformerfuser : public FuseBase {
 public:
  explicit SpatialTransformerfuser(bool dyn_reshape = false,
                                   bool output_xshape = true,
                                   bool post_reshape_new_pattern = false)
      : dyn_reshape_(dyn_reshape),
        output_xshape_(output_xshape),
        post_reshape_new_pattern_(post_reshape_new_pattern) {}

  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("group_norm", "X")
                      ->assert_is_op_input("__xpu__conv2d", "Branch")
                      ->AsInput();
    PMNode* shape = nullptr;
    PMNode* slice0 = nullptr;
    PMNode* slice1 = nullptr;
    PMNode* slice2 = nullptr;
    PMNode* shape_out = nullptr;
    PMNode* slice0_out = nullptr;
    PMNode* slice1_out = nullptr;
    PMNode* slice2_out = nullptr;
    PMNode* ew_mul = nullptr;
    PMNode* ew_mul_out = nullptr;
    PMNode* fill_constant = nullptr;
    PMNode* fill_constant_out = nullptr;
    if (dyn_reshape_) {
      input->assert_is_op_input("shape", "Input");
      shape = OpNode("shape", "shape")->AsIntermediate();
      shape_out = VarNode("shape_out")
                      ->assert_is_op_output("shape", "Out")
                      ->assert_is_op_input("slice", "Input")
                      ->AsIntermediate();
      slice0 = OpNode("slice0", "slice")->AsIntermediate();
      slice1 = OpNode("slice1", "slice")->AsIntermediate();
      slice2 = OpNode("slice2", "slice")->AsIntermediate();
      slice0_out = VarNode("slice0_out")
                       ->assert_is_op_output("slice", "Out")
                       ->assert_is_op_nth_input("reshape2", "ShapeTensor", 0)
                       ->AsIntermediate();
      slice1_out = VarNode("slice1_out")
                       ->assert_is_op_output("slice", "Out")
                       ->assert_is_op_input("elementwise_mul", "X")
                       ->AsIntermediate();
      slice2_out = VarNode("slice2_out")
                       ->assert_is_op_output("slice", "Out")
                       ->assert_is_op_input("elementwise_mul", "Y")
                       ->AsIntermediate();
      ew_mul = OpNode("ew_mul", "elementwise_mul")->AsIntermediate();
      ew_mul_out = VarNode("ew_mul_out")
                       ->assert_is_op_output("elementwise_mul", "Out")
                       ->assert_is_op_nth_input("reshape2", "ShapeTensor", 1)
                       ->AsIntermediate();
      fill_constant =
          OpNode("fill_constant", "fill_constant")->AsIntermediate();
      fill_constant_out =
          VarNode("fill_constant_out")
              ->assert_is_op_output("fill_constant", "Out")
              ->assert_is_op_nth_input("reshape2", "ShapeTensor", 2)
              ->AsIntermediate();
    }
    // image to sequence
    auto* gn_scale = VarNode("gn_scale")
                         ->assert_is_op_input("group_norm", "Scale")
                         ->AsInput();
    auto* gn_bias =
        VarNode("gn_bias")->assert_is_op_input("group_norm", "Bias")->AsInput();
    auto* gn = OpNode("gn", "group_norm")->AsIntermediate();
    auto* gn_out = VarNode("gn_out")
                       ->assert_is_op_output("group_norm", "Y")
                       ->assert_is_op_input("__xpu__conv2d", "Input")
                       ->AsIntermediate();
    auto* gn_mean = VarNode("gn_mean")
                        ->assert_is_op_output("group_norm", "Mean")
                        ->AsIntermediate();
    auto* gn_var = VarNode("gn_var")
                       ->assert_is_op_output("group_norm", "Variance")
                       ->AsIntermediate();

    auto* pre_xpu_conv2d =
        OpNode("pre__xpu__conv2d", "__xpu__conv2d")->AsIntermediate();
    auto* pre_xpu_conv2d_bias =
        VarNode("pre__xpu__conv2d_bias")
            ->assert_is_op_input("__xpu__conv2d", "Bias")
            ->AsInput();
    auto* pre_xpu_conv2d_filter =
        VarNode("pre__xpu__conv2d_filter")
            ->assert_is_op_input("__xpu__conv2d", "Filter")
            ->AsInput();
    auto* pre_xpu_conv2d_output =
        VarNode("pre__xpu__conv2d_output")
            ->AsIntermediate()
            ->assert_is_op_input("transpose2", "X")
            ->assert_is_op_output("__xpu__conv2d", "Output");
    auto* pre_xpu_conv2d_output_max =
        VarNode("pre__xpu__conv2d_output_max")
            ->AsIntermediate()
            ->assert_is_op_output("__xpu__conv2d", "OutputMax");

    auto* transpose2 = OpNode("transpose2", "transpose2")->AsIntermediate();
    auto* transpose2_output = VarNode("transpose2_output")
                                  ->AsIntermediate()
                                  ->assert_is_op_output("transpose2", "Out");
    PMNode* flatten = nullptr;
    PMNode* flatten_output = nullptr;
    PMNode* flatten_output_xshape = nullptr;

    PMNode* reshape2 = nullptr;
    PMNode* reshape2_output = nullptr;
    PMNode* reshape2_output_xshape = nullptr;
    if (dyn_reshape_) {
      reshape2 = OpNode("reshape2", "reshape2")->AsIntermediate();
      reshape2_output =
          VarNode("reshape2_output")
              ->assert_is_op_output("reshape2", "Out")
              ->AsIntermediate()
              ->assert_is_op_input("__xpu__multihead_self_attn", "Input")
              ->assert_is_op_input("elementwise_add", "Y");
      reshape2_output_xshape = VarNode("reshape2_output_xshape")
                                   ->AsIntermediate()
                                   ->assert_is_op_output("reshape2", "XShape");
      transpose2_output->assert_is_op_input("reshape2", "X");
    } else {
      flatten = OpNode("flatten_contiguous_range", "flatten_contiguous_range")
                    ->AsIntermediate();
      flatten_output =
          VarNode("flatten_output")
              ->AsIntermediate()
              ->assert_is_op_output("flatten_contiguous_range", "Out")
              ->assert_is_op_input("__xpu__multihead_self_attn", "Input")
              ->assert_is_op_input("elementwise_add", "Y");
      transpose2_output->assert_is_op_input("flatten_contiguous_range", "X");
    }

    // __xpu__multihead_self_attn
    auto* __xpu__multihead_self_attn =
        OpNode("__xpu__multihead_self_attn", "__xpu__multihead_self_attn")
            ->AsIntermediate();
    auto* __xpu__multihead_self_attn_fcbias =
        VarNode("__xpu__multihead_self_attn_fcbias")
            ->assert_is_op_input("__xpu__multihead_self_attn", "FCBias")
            ->AsInput();
    auto* __xpu__multihead_self_attn_lnbias =
        VarNode("__xpu__multihead_self_attn_lnbias")
            ->assert_is_op_input("__xpu__multihead_self_attn", "LNBias")
            ->AsInput();
    auto* __xpu__multihead_self_attn_lnscale =
        VarNode("__xpu__multihead_self_attn_lnscale")
            ->assert_is_op_input("__xpu__multihead_self_attn", "LNScale")
            ->AsInput();
    auto* __xpu__multihead_self_attn_fcweight0 =
        VarNode("__xpu__multihead_self_attn_fcweight0")
            ->assert_is_op_nth_input(
                "__xpu__multihead_self_attn", "FCWeight", 0)
            ->AsInput();
    auto* __xpu__multihead_self_attn_fcweight1 =
        VarNode("__xpu__multihead_self_attn_fcweight1")
            ->assert_is_op_nth_input(
                "__xpu__multihead_self_attn", "FCWeight", 1)
            ->AsInput();
    auto* __xpu__multihead_self_attn_fcweight2 =
        VarNode("__xpu__multihead_self_attn_fcweight2")
            ->assert_is_op_nth_input(
                "__xpu__multihead_self_attn", "FCWeight", 2)
            ->AsInput();
    auto* __xpu__multihead_self_attn_fcweight3 =
        VarNode("__xpu__multihead_self_attn_fcweight3")
            ->assert_is_op_nth_input(
                "__xpu__multihead_self_attn", "FCWeight", 3)
            ->AsInput();
    auto* __xpu__multihead_self_attn_output =
        VarNode("__xpu__multihead_self_attn_output")
            ->AsIntermediate()
            ->assert_is_op_output("__xpu__multihead_self_attn", "Output")
            ->assert_is_op_input("elementwise_add", "X");
    auto* residual_add =
        OpNode("elementwise_add", "elementwise_add")->AsIntermediate();
    auto* residual_add_output =
        VarNode("residual_add_output")
            ->AsIntermediate()
            ->assert_is_op_output("elementwise_add", "Out")
            ->assert_is_op_input("elementwise_add", "Y")
            ->assert_is_op_input("__xpu__multihead_cross_attn", "Input");

    // __xpu__multihead_cross_attn
    auto* __xpu__multihead_cross_attn =
        OpNode("__xpu__multihead_cross_attn", "__xpu__multihead_cross_attn")
            ->AsIntermediate();
    auto* __xpu__multihead_cross_attn_embedding =
        VarNode("__xpu__multihead_cross_attn_embedding")
            ->assert_is_op_input("__xpu__multihead_cross_attn", "Embedding")
            ->AsInput();
    auto* __xpu__multihead_cross_attn_fcbias =
        VarNode("__xpu__multihead_cross_attn_fcbias")
            ->assert_is_op_input("__xpu__multihead_cross_attn", "FCBias")
            ->AsInput();
    auto* __xpu__multihead_cross_attn_lnbias =
        VarNode("__xpu__multihead_cross_attn_lnbias")
            ->assert_is_op_input("__xpu__multihead_cross_attn", "LNBias")
            ->AsInput();
    auto* __xpu__multihead_cross_attn_lnscale =
        VarNode("__xpu__multihead_cross_attn_lnscale")
            ->assert_is_op_input("__xpu__multihead_cross_attn", "LNScale")
            ->AsInput();
    auto* __xpu__multihead_cross_attn_fcweight0 =
        VarNode("__xpu__multihead_cross_attn_fcweight0")
            ->assert_is_op_nth_input(
                "__xpu__multihead_cross_attn", "FCWeight", 0)
            ->AsInput();
    auto* __xpu__multihead_cross_attn_fcweight1 =
        VarNode("__xpu__multihead_cross_attn_fcweight1")
            ->assert_is_op_nth_input(
                "__xpu__multihead_cross_attn", "FCWeight", 1)
            ->AsInput();
    auto* __xpu__multihead_cross_attn_fcweight2 =
        VarNode("__xpu__multihead_cross_attn_fcweight2")
            ->assert_is_op_nth_input(
                "__xpu__multihead_cross_attn", "FCWeight", 2)
            ->AsInput();
    auto* __xpu__multihead_cross_attn_fcweight3 =
        VarNode("__xpu__multihead_cross_attn_fcweight3")
            ->assert_is_op_nth_input(
                "__xpu__multihead_cross_attn", "FCWeight", 3)
            ->AsInput();
    auto* __xpu__multihead_cross_attn_output =
        VarNode("__xpu__multihead_cross_attn_output")
            ->AsIntermediate()
            ->assert_is_op_output("__xpu__multihead_cross_attn", "Output")
            ->assert_is_op_input("elementwise_add", "X");
    auto* residual_add2 =
        OpNode("elementwise_add2", "elementwise_add")->AsIntermediate();
    auto* residual_add2_output =
        VarNode("residual2_add_output")
            ->AsIntermediate()
            ->assert_is_op_output("elementwise_add", "Out")
            ->assert_is_op_input("elementwise_add", "Y")
            ->assert_is_op_input("__xpu__geglu", "Input");

    // geglu
    auto* __xpu__geglu =
        OpNode("__xpu__geglu", "__xpu__geglu")->AsIntermediate();
    auto* __xpu__geglu_fcbias0 =
        VarNode("__xpu__geglu_fcbias0")
            ->assert_is_op_nth_input("__xpu__geglu", "FCBias", 0)
            ->AsInput();
    auto* __xpu__geglu_fcbias1 =
        VarNode("__xpu__geglu_fcbias1")
            ->assert_is_op_nth_input("__xpu__geglu", "FCBias", 1)
            ->AsInput();
    auto* __xpu__geglu_lnbias =
        VarNode("__xpu__geglu_lnbias")
            ->assert_is_op_input("__xpu__geglu", "LNBias")
            ->AsInput();
    auto* __xpu__geglu_lnscale =
        VarNode("__xpu__geglu_lnscale")
            ->assert_is_op_input("__xpu__geglu", "LNScale")
            ->AsInput();
    auto* __xpu__geglu_fcweight0 =
        VarNode("__xpu__geglu_fcweight0")
            ->assert_is_op_nth_input("__xpu__geglu", "FCWeight", 0)
            ->AsInput();
    auto* __xpu__geglu_fcweight1 =
        VarNode("__xpu__geglu_fcweight1")
            ->assert_is_op_nth_input("__xpu__geglu", "FCWeight", 1)
            ->AsInput();
    auto* __xpu__geglu_output =
        VarNode("__xpu__geglu_output")
            ->AsIntermediate()
            ->assert_is_op_output("__xpu__geglu", "Output")
            ->assert_is_op_input("elementwise_add", "X");
    auto* residual_add3 =
        OpNode("elementwise_add3", "elementwise_add")->AsIntermediate();
    auto* residual_add3_output =
        VarNode("residual3_add_output")
            ->AsIntermediate()
            ->assert_is_op_output("elementwise_add", "Out")
            ->assert_is_op_input("reshape2", "X");

    // sequence to image
    PMNode* fill_constant2 = nullptr;
    PMNode* fill_constant2_out = nullptr;
    if (dyn_reshape_) {
      fill_constant2 =
          OpNode("fill_constant2", "fill_constant")->AsIntermediate();
      fill_constant2_out =
          VarNode("fill_constant2_out")
              ->AsIntermediate()
              ->assert_is_op_nth_input("reshape2", "ShapeTensor", 3);
      slice0_out->assert_is_op_nth_input("reshape2", "ShapeTensor", 0);
      slice1_out->assert_is_op_nth_input("reshape2", "ShapeTensor", 1);
      slice2_out->assert_is_op_nth_input("reshape2", "ShapeTensor", 2);
    }
    auto* post_reshape = OpNode("post_reshape", "reshape2")->AsIntermediate();
    auto* reshape_output = VarNode("reshape_output")
                               ->AsIntermediate()
                               ->assert_is_op_input("transpose2", "X")
                               ->assert_is_op_output("reshape2", "Out");
    auto* reshape_output_xshape =
        VarNode("reshape_output_xshape")
            ->AsIntermediate()
            ->assert_is_op_output("reshape2", "XShape");
    auto* transpose2_2 = OpNode("transpose2_2", "transpose2")->AsIntermediate();
    auto* transpose2_2_output =
        VarNode("transpose2_2_output")
            ->AsIntermediate()
            ->assert_is_op_input("__xpu__conv2d", "Input")
            ->assert_is_op_output("transpose2", "Out");
    auto* post_xpu_conv2d =
        OpNode("post__xpu__conv2d", "__xpu__conv2d")->AsIntermediate();
    auto* post_xpu_conv2d_bias =
        VarNode("post__xpu__conv2d_bias")
            ->assert_is_op_input("__xpu__conv2d", "Bias")
            ->AsInput();
    auto* post_xpu_conv2d_filter =
        VarNode("post__xpu__conv2d_filter")
            ->assert_is_op_input("__xpu__conv2d", "Filter")
            ->AsInput();
    auto* post_xpu_conv2d_output =
        VarNode("post__xpu__conv2d_output")
            ->AsOutput()
            ->assert_is_op_output("__xpu__conv2d", "Output");
    auto* post_xpu_conv2d_outputmax =
        VarNode("post__xpu__conv2d_output_max")
            ->AsIntermediate()
            ->assert_is_op_output("__xpu__conv2d", "OutputMax");

    std::vector<PMNode*> gn_input{input, gn_bias, gn_scale};
    std::vector<PMNode*> gn_output{gn_out, gn_mean, gn_var};
    gn_input >> *gn >> gn_output;
    std::vector<PMNode*> pre_conv2d_input{
        gn_out, pre_xpu_conv2d_bias, pre_xpu_conv2d_filter};
    std::vector<PMNode*> pre_conv2d_output{pre_xpu_conv2d_output,
                                           pre_xpu_conv2d_output_max};
    pre_conv2d_input >> *pre_xpu_conv2d >> pre_conv2d_output;
    *pre_xpu_conv2d_output >> *transpose2 >> *transpose2_output;

    if (output_xshape_) {
      auto* transpose2_output_xshape =
          VarNode("transpose2_output_xshape")
              ->AsIntermediate()
              ->assert_is_op_output("transpose2", "XShape");
      *transpose2 >> *transpose2_output_xshape;
    }
    PMNode* sequence_input = nullptr;
    if (dyn_reshape_) {
      *input >> *shape >> *shape_out;
      *shape_out >> *slice0 >> *slice0_out >> *reshape2;
      *shape_out >> *slice1 >> *slice1_out >> *ew_mul >> *ew_mul_out >>
          *reshape2;
      *shape_out >> *slice2 >> *slice2_out >> *ew_mul;
      *fill_constant >> *fill_constant_out >> *reshape2;
      *transpose2_output >> *reshape2 >> *reshape2_output;
      *reshape2 >> *reshape2_output_xshape;
      sequence_input = reshape2_output;
    } else {
      *transpose2_output >> *flatten >> *flatten_output;
      if (output_xshape_) {
        flatten_output_xshape =
            VarNode("flatten_output_xshape")
                ->AsIntermediate()
                ->assert_is_op_output("flatten_contiguous_range", "XShape");
        *flatten >> *flatten_output_xshape;
      }
      sequence_input = flatten_output;
    }

    std::vector<PMNode*> mhsa_input{sequence_input,
                                    __xpu__multihead_self_attn_fcbias,
                                    __xpu__multihead_self_attn_fcweight0,
                                    __xpu__multihead_self_attn_fcweight1,
                                    __xpu__multihead_self_attn_fcweight2,
                                    __xpu__multihead_self_attn_fcweight3,
                                    __xpu__multihead_self_attn_lnbias,
                                    __xpu__multihead_self_attn_lnscale};
    mhsa_input >> *__xpu__multihead_self_attn >>
        *__xpu__multihead_self_attn_output >> *residual_add >>
        *residual_add_output;
    *sequence_input >> *residual_add;

    std::vector<PMNode*> mhca_input{residual_add_output,
                                    __xpu__multihead_cross_attn_embedding,
                                    __xpu__multihead_cross_attn_fcbias,
                                    __xpu__multihead_cross_attn_lnbias,
                                    __xpu__multihead_cross_attn_lnscale,
                                    __xpu__multihead_cross_attn_fcweight0,
                                    __xpu__multihead_cross_attn_fcweight1,
                                    __xpu__multihead_cross_attn_fcweight2,
                                    __xpu__multihead_cross_attn_fcweight3};
    mhca_input >> *__xpu__multihead_cross_attn >>
        *__xpu__multihead_cross_attn_output >> *residual_add2 >>
        *residual_add2_output;
    *residual_add_output >> *residual_add2;

    std::vector<PMNode*> geglu_input{residual_add2_output,
                                     __xpu__geglu_fcbias0,
                                     __xpu__geglu_fcbias1,
                                     __xpu__geglu_lnbias,
                                     __xpu__geglu_lnscale,
                                     __xpu__geglu_fcweight0,
                                     __xpu__geglu_fcweight1};
    geglu_input >> *__xpu__geglu >> *__xpu__geglu_output >> *residual_add3 >>
        *residual_add3_output;
    *residual_add2_output >> *residual_add3;

    *residual_add3_output >> *post_reshape >> *reshape_output >>
        *transpose2_2 >> *transpose2_2_output;
    *post_reshape >> *reshape_output_xshape;
    if (output_xshape_) {
      auto* transpose2_2_output_xshape =
          VarNode("transpose2_2_output_xshape")
              ->AsIntermediate()
              ->assert_is_op_output("transpose2", "XShape");
      *transpose2_2 >> *transpose2_2_output_xshape;
    }
    if (dyn_reshape_) {
      *fill_constant2 >> *fill_constant2_out;
      std::vector<PMNode*> dyn_reshape__inputs = {residual_add3_output,
                                                  slice0_out,
                                                  slice1_out,
                                                  slice2_out,
                                                  fill_constant2_out};
      dyn_reshape__inputs >> *post_reshape;
    }

    if (post_reshape_new_pattern_) {
      // This is for new pattern in new pipeline model.
      input->assert_is_op_input("shape", "Input");
      PMNode* shape_1 = OpNode("shape_1", "shape")->AsIntermediate();
      PMNode* shape_out_1 = VarNode("shape_out_1")
                                ->assert_is_op_output("shape", "Out")
                                ->assert_is_op_input("slice", "Input")
                                ->AsIntermediate();

      PMNode* slice_1 = OpNode("slice_1", "slice")->AsIntermediate();
      PMNode* slice_2 = OpNode("slice_2", "slice")->AsIntermediate();
      PMNode* slice_1_out = VarNode("slice_1_out")
                                ->assert_is_op_output("slice", "Out")
                                ->AsIntermediate();
      PMNode* slice_2_out = VarNode("slice_2_out")
                                ->assert_is_op_output("slice", "Out")
                                ->AsIntermediate();

      PMNode* fill_constant3 =
          OpNode("fill_constant3", "fill_constant")->AsIntermediate();
      PMNode* fill_constant3_out =
          VarNode("fill_constant3_out")
              ->AsIntermediate()
              ->assert_is_op_nth_input("reshape2", "ShapeTensor", 0);
      PMNode* fill_constant4 =
          OpNode("fill_constant4", "fill_constant")->AsIntermediate();
      PMNode* fill_constant4_out =
          VarNode("fill_constant4_out")
              ->AsIntermediate()
              ->assert_is_op_nth_input("reshape2", "ShapeTensor", 3);
      slice_1_out->assert_is_op_nth_input("reshape2", "ShapeTensor", 1);
      slice_2_out->assert_is_op_nth_input("reshape2", "ShapeTensor", 2);

      *fill_constant3 >> *fill_constant3_out;
      *fill_constant4 >> *fill_constant4_out;
      *input >> *shape_1 >> *shape_out_1;
      *shape_out_1 >> *slice_1 >> *slice_1_out;
      *shape_out_1 >> *slice_2 >> *slice_2_out;

      std::vector<PMNode*> dyn_reshape__inputs = {residual_add3_output,
                                                  fill_constant3_out,
                                                  slice_1_out,
                                                  slice_2_out,
                                                  fill_constant4_out};
      dyn_reshape__inputs >> *post_reshape;
    }

    std::vector<PMNode*> post_conv2d_input{transpose2_2_output,
                                           post_xpu_conv2d_bias,
                                           input,
                                           post_xpu_conv2d_filter};
    std::vector<PMNode*> post_conv2d_output{post_xpu_conv2d_output,
                                            post_xpu_conv2d_outputmax};
    post_conv2d_input >> *post_xpu_conv2d >> post_conv2d_output;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    // OpDesc
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__spatial_transformer");
    auto* gn_op_info = matched.at("gn")->stmt()->op_info();
    auto* mhsa_op_info =
        matched.at("__xpu__multihead_self_attn")->stmt()->op_info();
    auto* mhca_op_info =
        matched.at("__xpu__multihead_cross_attn")->stmt()->op_info();
    auto* geglu_op_info = matched.at("__xpu__geglu")->stmt()->op_info();

    std::vector<std::string> fc_weight_names;
    for (const auto& name : mhsa_op_info->Input("FCWeight")) {
      fc_weight_names.push_back(name);
    }
    for (const auto& name : mhca_op_info->Input("FCWeight")) {
      fc_weight_names.push_back(name);
    }
    for (const auto& name : geglu_op_info->Input("FCWeight")) {
      fc_weight_names.push_back(name);
    }
    CHECK_EQ(fc_weight_names.size(), 10);
    std::vector<std::string> fc_weight_maxptr_names;
    for (const auto& name :
         mhsa_op_info->GetAttr<std::vector<std::string>>("FCWeightMax")) {
      fc_weight_maxptr_names.push_back(name);
    }
    for (const auto& name :
         mhca_op_info->GetAttr<std::vector<std::string>>("FCWeightMax")) {
      fc_weight_maxptr_names.push_back(name);
    }
    for (const auto& name :
         geglu_op_info->GetAttr<std::vector<std::string>>("FCWeightMax")) {
      fc_weight_maxptr_names.push_back(name);
    }
    CHECK_EQ(fc_weight_maxptr_names.size(), 10);

    std::vector<std::string> ln_scale_names;
    for (const auto& name : mhsa_op_info->Input("LNScale")) {
      ln_scale_names.push_back(name);
    }
    for (const auto& name : mhca_op_info->Input("LNScale")) {
      ln_scale_names.push_back(name);
    }
    for (const auto& name : geglu_op_info->Input("LNScale")) {
      ln_scale_names.push_back(name);
    }
    std::vector<std::string> ln_bias_names;
    for (const auto& name : mhsa_op_info->Input("LNBias")) {
      ln_bias_names.push_back(name);
    }
    for (const auto& name : mhca_op_info->Input("LNBias")) {
      ln_bias_names.push_back(name);
    }
    for (const auto& name : geglu_op_info->Input("LNBias")) {
      ln_bias_names.push_back(name);
    }
    std::vector<std::string> fc_bias_names;
    for (const auto& name : mhsa_op_info->Input("FCBias")) {
      fc_bias_names.push_back(name);
    }
    for (const auto& name : mhca_op_info->Input("FCBias")) {
      fc_bias_names.push_back(name);
    }
    for (const auto& name : geglu_op_info->Input("FCBias")) {
      fc_bias_names.push_back(name);
    }

    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetInput("Embedding", mhca_op_info->Input("Embedding"));
    op_desc.SetInput("FCWeight", fc_weight_names);
    op_desc.SetInput("FCBias", fc_bias_names);
    op_desc.SetInput("LNScale", ln_scale_names);
    op_desc.SetInput("LNBias", ln_bias_names);
    op_desc.SetAttr<int>("groups", gn_op_info->GetAttr<int>("groups"));
    op_desc.SetAttr<float>("epsilon", gn_op_info->GetAttr<float>("epsilon"));
    op_desc.SetInput("ConvBias",
                     {matched.at("pre__xpu__conv2d_bias")->arg()->name,
                      matched.at("post__xpu__conv2d_bias")->arg()->name});
    op_desc.SetInput("GNScale", {matched.at("gn_scale")->arg()->name});
    op_desc.SetInput("GNBias", {matched.at("gn_bias")->arg()->name});
    std::vector<std::string> conv_filter_names = {
        matched.at("pre__xpu__conv2d_filter")->arg()->name,
        matched.at("post__xpu__conv2d_filter")->arg()->name};
    op_desc.SetInput("ConvWeight", conv_filter_names);
    std::vector<std::string> conv_filter_maxptr_names = {
        matched.at("pre__xpu__conv2d_filter")->arg()->name + "_max",
        matched.at("post__xpu__conv2d_filter")->arg()->name + "_max"};
    op_desc.SetAttr<std::vector<std::string>>("ConvFilterMax",
                                              conv_filter_maxptr_names);
    op_desc.SetOutput("Output",
                      {matched.at("post__xpu__conv2d_output")->arg()->name});
    op_desc.SetAttr<std::vector<std::string>>("FCWeightMax",
                                              fc_weight_maxptr_names);
    op_desc.SetAttr<int>("head_num", mhsa_op_info->GetAttr<int>("head_num"));
    op_desc.SetAttr<int>("size_per_head",
                         mhsa_op_info->GetAttr<int>("size_per_head"));
    op_desc.SetAttr<int>("hidden_dim",
                         mhsa_op_info->GetAttr<int>("hidden_dim"));
    op_desc.SetAttr<int>("embedding_dim",
                         mhca_op_info->GetAttr<int>("embedding_dim"));
    op_desc.SetAttr<int>("gelu_dim", geglu_op_info->GetAttr<int>("gelu_dim"));

    std::vector<std::vector<int>> strides;
    std::vector<std::vector<int>> paddings;
    std::vector<std::vector<int>> dilations;
    std::vector<std::vector<int>> filter_dims;
    std::vector<int> groups;
    std::vector<std::string> conv_vec = {"pre__xpu__conv2d",
                                         "post__xpu__conv2d"};
    for (auto pm_name : conv_vec) {
      auto* conv_op_info = matched.at(pm_name)->stmt()->op_info();
      auto strides_tmp = conv_op_info->GetAttr<std::vector<int>>("strides");
      strides.emplace_back(std::move(strides_tmp));
      auto paddings_tmp = conv_op_info->GetAttr<std::vector<int>>("paddings");
      paddings.emplace_back(std::move(paddings_tmp));
      auto dilations_tmp = conv_op_info->GetAttr<std::vector<int>>("dilations");
      dilations.emplace_back(std::move(dilations_tmp));
      std::vector<int> groups_tmp =
          conv_op_info->GetAttr<std::vector<int>>("groups");
      groups.push_back(groups_tmp[0]);
      auto filter_dims_tmp =
          conv_op_info->GetAttr<std::vector<int>>("filter_dims");
      filter_dims.emplace_back(std::move(filter_dims_tmp));
    }
    op_desc.SetAttr<std::vector<int>>("Conv_Groups", groups);
    op_desc.SetAttr<std::vector<int>>("Strides", IntVec2DTo1D(strides));
    op_desc.SetAttr<std::vector<int>>("Paddings", IntVec2DTo1D(paddings));
    op_desc.SetAttr<std::vector<int>>("Dilations", IntVec2DTo1D(dilations));
    op_desc.SetAttr<std::vector<int>>("FilterDims", IntVec2DTo1D(filter_dims));

    auto spatial_transformer_op =
        LiteOpRegistry::Global().Create(op_desc.Type());
    auto* scope = matched.at("gn")->stmt()->op()->scope();
    UpdateWeight(scope, conv_filter_names, conv_filter_maxptr_names, false);
    spatial_transformer_op->Attach(op_desc, scope);
    spatial_transformer_op->SetValidPlaces(
        matched.at("gn")->stmt()->op()->valid_places());
    auto kernels = spatial_transformer_op->CreateKernels(
        spatial_transformer_op->valid_places());
    auto* new_op_node = graph->GraphCreateInstructNode(
        spatial_transformer_op, spatial_transformer_op->valid_places());

    std::vector<std::string> froms = {"input",
                                      "gn_scale",
                                      "gn_bias",
                                      "pre__xpu__conv2d_bias",
                                      "pre__xpu__conv2d_filter",
                                      "__xpu__multihead_self_attn_fcbias",
                                      "__xpu__multihead_self_attn_lnbias",
                                      "__xpu__multihead_self_attn_lnscale",
                                      "__xpu__multihead_self_attn_fcweight0",
                                      "__xpu__multihead_self_attn_fcweight1",
                                      "__xpu__multihead_self_attn_fcweight2",
                                      "__xpu__multihead_self_attn_fcweight3",
                                      "__xpu__multihead_cross_attn_embedding",
                                      "__xpu__multihead_cross_attn_fcbias",
                                      "__xpu__multihead_cross_attn_lnbias",
                                      "__xpu__multihead_cross_attn_lnscale",
                                      "__xpu__multihead_cross_attn_fcweight0",
                                      "__xpu__multihead_cross_attn_fcweight1",
                                      "__xpu__multihead_cross_attn_fcweight2",
                                      "__xpu__multihead_cross_attn_fcweight3",
                                      "__xpu__geglu_fcbias0",
                                      "__xpu__geglu_fcbias1",
                                      "__xpu__geglu_lnbias",
                                      "__xpu__geglu_lnscale",
                                      "__xpu__geglu_fcweight0",
                                      "__xpu__geglu_fcweight1",
                                      "post__xpu__conv2d_bias",
                                      "post__xpu__conv2d_filter"};

    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), new_op_node);
    }

    IR_OP_VAR_LINK(new_op_node, matched.at("post__xpu__conv2d_output"));
  }

 private:
  void UpdateWeight(Scope* scope,
                    const std::vector<std::string>& fc_weight_names,
                    const std::vector<std::string>& fc_weight_max_names,
                    bool trans) {
    std::vector<Tensor*> weight_tensor_vec(fc_weight_names.size(), nullptr);
    std::vector<DDimLite> weight_dims_vec(fc_weight_names.size());
    std::vector<int> weight_len_vec(fc_weight_names.size());

    for (size_t i = 0; i < fc_weight_names.size(); ++i) {
      weight_tensor_vec[i] = scope->FindMutableTensor(fc_weight_names[i]);
      CHECK(weight_tensor_vec[i] != nullptr);
      weight_dims_vec[i] = weight_tensor_vec[i]->dims();
      weight_len_vec[i] = weight_tensor_vec[i]->numel();
      if (trans && i > 0) {
        CHECK_EQ(weight_dims_vec[i][0], weight_dims_vec[i - 1][0]);
      }
    }
    for (size_t i = 0; i < fc_weight_names.size(); ++i) {
      float* weight_host_ptr = weight_tensor_vec[i]->mutable_data<float>();
      std::unique_ptr<float[]> weight_host_trans(new float[weight_len_vec[i]]);
      std::unique_ptr<int16_t[]> weight_host_trans_int16(
          new int16_t[weight_len_vec[i]]);
      if (trans) {
        paddle::lite::xpu::math::Transpose<float>(weight_host_ptr,
                                                  weight_host_trans.get(),
                                                  weight_dims_vec[i][0],
                                                  weight_dims_vec[i][1]);
      } else {
        memcpy(weight_host_trans.get(),
               weight_host_ptr,
               weight_len_vec[i] * sizeof(float));
      }
      float max_f = paddle::lite::xpu::math::FindMaxAbs(weight_host_trans.get(),
                                                        weight_len_vec[i]);
      paddle::lite::xpu::math::ConvertFP32ToInt16(weight_host_trans.get(),
                                                  weight_host_trans_int16.get(),
                                                  max_f,
                                                  weight_len_vec[i]);
      memcpy(weight_tensor_vec[i]->mutable_data<int16_t>(),
             weight_host_trans_int16.get(),
             weight_len_vec[i] * sizeof(int16_t));
      scope->NewTensor(fc_weight_max_names[i]);
      Tensor* weight_maxptr_tensor =
          scope->FindMutableTensor(fc_weight_max_names[i]);
      weight_maxptr_tensor->Resize({6});
      std::vector<float> weight_maxptr_host(6, max_f);
      memcpy(weight_maxptr_tensor->mutable_data<float>(),
             weight_maxptr_host.data(),
             weight_maxptr_host.size() * sizeof(float));
    }
  }
  bool dyn_reshape_;
  bool output_xshape_;
  bool post_reshape_new_pattern_;
};

}  // namespace fusion

class XPUSpatialTransformerfusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto dyn_reshape : {true, false}) {
      for (auto output_xshape : {true, false}) {
        for (auto post_reshape_new_pattern : {true, false}) {
          fusion::SpatialTransformerfuser fuser(
              dyn_reshape, output_xshape, post_reshape_new_pattern);
          fuser(graph.get());
        }
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__spatial_transformer_fuse_pass,
                  paddle::lite::mir::XPUSpatialTransformerfusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__spatial_transformer");
