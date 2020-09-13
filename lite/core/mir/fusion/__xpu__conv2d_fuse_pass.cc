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
#include <string>
#include "lite/backends/xpu/math.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {
/* fuse conv2d block in resnet50-like model to xpu_conv2d op    */
/* For example:                                                 */
/* graph[1]: sub block                                          */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                     conv2d----in_Filter                      */
/*                       |                                      */
/*                       |                                      */
/*                  batch_norm ------in_Bias                    */
/*                       |                                      */
/*                       |                                      */
/*                     relu                                     */
/*                       |                                      */
/*                       |                                      */
/*                     out_Out                                  */
/*                                                              */
/* After the pass is applied:                                   */
/*                     in_Input                                 */
/*        in_Filter      |     in_FilterMax                     */
/*                  \    |    /                                 */
/*                   \   |   /                                  */
/*     in_Bias ------- __xpu__conv2d                            */
/*                       |    \                                 */
/*                       |     \                                */
/*                       |      out_OutputMax                   */
/*                 out_Output                                   */
/*                                                              */
/* ------------------------------------------------------       */
/* graph[2]: sub block                                          */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                     conv2d----in_Filter                      */
/*                       |                                      */
/*                       |                                      */
/*                  batch_norm ------in_Bias                    */
/*                       |                                      */
/*                       |                                      */
/*                     out_Out                                  */
/*                                                              */
/* After the pass is applied:                                   */
/*                     in_Input                                 */
/*        in_Filter      |     in_FilterMax                     */
/*                  \    |    /                                 */
/*                   \   |   /                                  */
/*     in_Bias ------- __xpu__conv2d                            */
/*                       |    \                                 */
/*                       |     \                                */
/*                       |      out_OutputMax                   */
/*                     out_Output                               */
/*                                                              */
/* ------------------------------------------------------       */
/* graph[3]: sub block                                          */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                     conv2d----in_Filter                      */
/*                       |                                      */
/*                       |                                      */
/*        in_X       batch_norm ------in_Bias                   */
/*             \         |                                      */
/*               \       |                                      */
/*                elementwise_add                               */
/*                       |                                      */
/*                       |                                      */
/*                     relu                                     */
/*                       |                                      */
/*                       |                                      */
/*                     out_Out                                  */
/*                                                              */
/* After the pass is applied:                                   */
/*                     in_Input                                 */
/*        in_Filter      |     in_FilterMax                     */
/*                  \    |    /                                 */
/*                   \   |   /                                  */
/*  in_Branch ------- __xpu__conv2d ------ in_Bias              */
/*                       |    \                                 */
/*                       |     \                                */
/*                       |      out_OutputMax                   */
/*                    out_Output                                */

class XPUConv2dBlock0Fuser : public FuseBase {
 public:
  explicit XPUConv2dBlock0Fuser(bool with_relu) : _with_relu(with_relu) {}

  void BuildPattern() override {
    auto* input =
        VarNode("input")->assert_is_op_input("conv2d", "Input")->AsInput();

    auto* conv_filter = VarNode("conv_filter")
                            ->assert_is_op_input("conv2d", "Filter")
                            ->AsInput();
    auto* conv = OpNode("conv", "conv2d")->AsIntermediate();
    auto* conv_out = VarNode("conv_out")
                         ->assert_is_op_output("conv2d", "Output")
                         ->assert_is_op_input("batch_norm", "X")
                         ->AsIntermediate();
    auto* bn_bias =
        VarNode("bn_bias")->assert_is_op_input("batch_norm", "Bias")->AsInput();
    auto* bn_mean = VarNode("bn_mean")
                        ->assert_is_op_input("batch_norm", "Mean")
                        ->AsIntermediate();
    auto* bn_scale = VarNode("bn_scale")
                         ->assert_is_op_input("batch_norm", "Scale")
                         ->AsIntermediate();
    auto* bn_var = VarNode("bn_variance")
                       ->assert_is_op_input("batch_norm", "Variance")
                       ->AsIntermediate();
    auto* bn = OpNode("bn", "batch_norm")->AsIntermediate();
    auto* bn_out = VarNode("bn_out")->assert_is_op_output("batch_norm", "Y");
    auto* bn_mean_out = VarNode("bn_mean_out")
                            ->assert_is_op_output("batch_norm", "MeanOut")
                            ->AsIntermediate();
    auto* bn_saved_mean = VarNode("bn_saved_mean")
                              ->assert_is_op_output("batch_norm", "SavedMean")
                              ->AsIntermediate();
    auto* bn_var_out = VarNode("bn_var_out")
                           ->assert_is_op_output("batch_norm", "VarianceOut")
                           ->AsIntermediate();
    auto* bn_saved_var =
        VarNode("bn_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();

    *input >> *conv >> *conv_out >> *bn >> *bn_out;

    *conv_filter >> *conv;
    *bn_bias >> *bn;
    *bn_mean >> *bn;
    *bn_scale >> *bn;
    *bn_var >> *bn;
    *bn >> *bn_mean_out;
    *bn >> *bn_saved_mean;
    *bn >> *bn_saved_var;
    *bn >> *bn_var_out;

    if (_with_relu) {
      bn_out->assert_is_op_input("relu", "X")->AsIntermediate();
      auto* relu = OpNode("relu", "relu")->AsIntermediate();
      auto* relu_out =
          VarNode("relu_out")->assert_is_op_output("relu", "Out")->AsOutput();

      *bn_out >> *relu >> *relu_out;
    } else {
      bn_out->AsOutput();
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto op_desc = *matched.at("conv")->stmt()->op_info();
    auto conv_old = matched.at("conv")->stmt()->op();
    auto* scope = conv_old->scope();
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("__xpu__conv2d");
    std::string input_name = matched.at("input")->arg()->name;
    op_desc.SetInput("Input", {input_name});

    auto filter_name = matched.at("conv_filter")->arg()->name;
    auto scale_name = matched.at("bn_scale")->arg()->name;
    auto bias_name = matched.at("bn_bias")->arg()->name;
    auto mean_name = matched.at("bn_mean")->arg()->name;
    auto var_name = matched.at("bn_variance")->arg()->name;

    auto* filter_t = scope->FindMutableTensor(filter_name);
    auto* scale_t = scope->FindMutableTensor(scale_name);
    auto* bias_t = scope->FindMutableTensor(bias_name);
    auto* mean_t = scope->FindMutableTensor(mean_name);
    auto* var_t = scope->FindMutableTensor(var_name);

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
    std::string max_filter_name = filter_name + "_max";
    auto* max_filter_node = graph->NewArgumentNode(max_filter_name);
    max_filter_node->arg()->is_weight = true;
    max_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));

    auto* max_filter_t = scope->NewTensor(max_filter_name);
    max_filter_t->Resize({4});
    float* max_ptr = max_filter_t->mutable_data<float>();
    max_ptr[0] = max_f;
    max_ptr[1] = max_f;
    max_ptr[2] = max_f;
    max_ptr[3] = max_f;

    op_desc.SetInput("Filter", {filter_name});
    op_desc.SetInput("Bias", {bias_name});
    op_desc.SetInput("FilterMax", {max_filter_name});

    std::string output_name = "";
    if (_with_relu) {
      op_desc.SetAttr("act_type", std::string{"relu"});
      output_name = matched.at("relu_out")->arg()->name;
    } else {
      output_name = matched.at("bn_out")->arg()->name;
    }
    op_desc.SetOutput("Output", {output_name});

    // add new arg output_max
    std::string max_output_name = output_name + "_max";
    auto* max_output_node = graph->NewArgumentNode(max_output_name);
    max_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    scope->NewTensor(max_output_name);
    op_desc.SetOutput("OutputMax", {max_output_name});

    auto conv_op = LiteOpRegistry::Global().Create("__xpu__conv2d");
    auto& valid_places = conv_old->valid_places();
    conv_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(conv_op, valid_places);
    DirectedLink(matched.at("input"), new_op_node);
    DirectedLink(matched.at("conv_filter"), new_op_node);
    DirectedLink(matched.at("bn_bias"), new_op_node);
    DirectedLink(max_filter_node, new_op_node);
    DirectedLink(new_op_node, max_output_node);
    if (_with_relu) {
      DirectedLink(new_op_node, matched.at("relu_out"));
    } else {
      DirectedLink(new_op_node, matched.at("bn_out"));
    }
  }

 private:
  bool _with_relu;
};

// block with branch
class XPUConv2dBlock1Fuser : public FuseBase {
 public:
  XPUConv2dBlock1Fuser() {}

  void BuildPattern() override {
    auto* input =
        VarNode("input")->assert_is_op_input("conv2d", "Input")->AsInput();

    auto* conv_filter = VarNode("conv_filter")
                            ->assert_is_op_input("conv2d", "Filter")
                            ->AsInput();
    auto* conv = OpNode("conv", "conv2d")->AsIntermediate();
    auto* conv_out = VarNode("conv_out")
                         ->assert_is_op_output("conv2d", "Output")
                         ->assert_is_op_input("batch_norm", "X")
                         ->AsIntermediate();
    auto* bn_bias =
        VarNode("bn_bias")->assert_is_op_input("batch_norm", "Bias")->AsInput();
    auto* bn_mean = VarNode("bn_mean")
                        ->assert_is_op_input("batch_norm", "Mean")
                        ->AsIntermediate();
    auto* bn_scale = VarNode("bn_scale")
                         ->assert_is_op_input("batch_norm", "Scale")
                         ->AsIntermediate();
    auto* bn_var = VarNode("bn_variance")
                       ->assert_is_op_input("batch_norm", "Variance")
                       ->AsIntermediate();
    auto* bn = OpNode("bn", "batch_norm")->AsIntermediate();
    auto* bn_out = VarNode("bn_out")
                       ->assert_is_op_output("batch_norm", "Y")
                       ->assert_is_op_input("elementwise_add", "Y")
                       ->AsIntermediate();
    auto* bn_mean_out = VarNode("bn_mean_out")
                            ->assert_is_op_output("batch_norm", "MeanOut")
                            ->AsIntermediate();
    auto* bn_saved_mean = VarNode("bn_saved_mean")
                              ->assert_is_op_output("batch_norm", "SavedMean")
                              ->AsIntermediate();
    auto* bn_var_out = VarNode("bn_var_out")
                           ->assert_is_op_output("batch_norm", "VarianceOut")
                           ->AsIntermediate();
    auto* bn_saved_var =
        VarNode("bn_saved_var")
            ->assert_is_op_output("batch_norm", "SavedVariance")
            ->AsIntermediate();
    auto* ew_x =
        VarNode("ew_x")->assert_is_op_input("elementwise_add", "X")->AsInput();
    auto* ew_add = OpNode("ew_add", "elementwise_add")->AsIntermediate();
    auto* ew_out = VarNode("ew_out")
                       ->assert_is_op_output("elementwise_add", "Out")
                       ->assert_is_op_input("relu", "X")
                       ->AsIntermediate();
    auto* relu = OpNode("relu", "relu")->AsIntermediate();
    auto* relu_out =
        VarNode("relu_out")->assert_is_op_output("relu", "Out")->AsOutput();

    *input >> *conv >> *conv_out >> *bn >> *bn_out >> *ew_add >> *ew_out >>
        *relu >> *relu_out;

    *conv_filter >> *conv;
    *bn_bias >> *bn;
    *bn_mean >> *bn;
    *bn_scale >> *bn;
    *bn_var >> *bn;
    *bn >> *bn_mean_out;
    *bn >> *bn_saved_mean;
    *bn >> *bn_saved_var;
    *bn >> *bn_var_out;

    *ew_x >> *ew_add;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto op_desc = *matched.at("conv")->stmt()->op_info();
    auto conv_old = matched.at("conv")->stmt()->op();
    auto* scope = conv_old->scope();
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("__xpu__conv2d");
    std::string input_name = matched.at("input")->arg()->name;
    op_desc.SetInput("Input", {input_name});

    auto filter_name = matched.at("conv_filter")->arg()->name;
    auto scale_name = matched.at("bn_scale")->arg()->name;
    auto bias_name = matched.at("bn_bias")->arg()->name;
    auto mean_name = matched.at("bn_mean")->arg()->name;
    auto var_name = matched.at("bn_variance")->arg()->name;

    auto* filter_t = scope->FindMutableTensor(filter_name);
    auto* scale_t = scope->FindMutableTensor(scale_name);
    auto* bias_t = scope->FindMutableTensor(bias_name);
    auto* mean_t = scope->FindMutableTensor(mean_name);
    auto* var_t = scope->FindMutableTensor(var_name);

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
    std::string max_filter_name = filter_name + "_max";
    auto* max_filter_node = graph->NewArgumentNode(max_filter_name);
    max_filter_node->arg()->is_weight = true;
    max_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));

    auto* max_filter_t = scope->NewTensor(max_filter_name);
    max_filter_t->Resize({4});
    float* max_ptr = max_filter_t->mutable_data<float>();
    max_ptr[0] = max_f;
    max_ptr[1] = max_f;
    max_ptr[2] = max_f;
    max_ptr[3] = max_f;

    op_desc.SetInput("Filter", {filter_name});
    op_desc.SetInput("Bias", {bias_name});
    op_desc.SetInput("FilterMax", {max_filter_name});
    op_desc.SetInput("Branch", {matched.at("ew_x")->arg()->name});

    std::string output_name = matched.at("relu_out")->arg()->name;
    op_desc.SetOutput("Output", {output_name});

    // add new arg output_max
    std::string max_output_name = output_name + "_max";
    auto* max_output_node = graph->NewArgumentNode(max_output_name);
    max_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    scope->NewTensor(max_output_name);
    op_desc.SetOutput("OutputMax", {max_output_name});
    op_desc.SetAttr("act_type", std::string{"relu"});

    auto conv_op = LiteOpRegistry::Global().Create("__xpu__conv2d");
    auto& valid_places = conv_old->valid_places();
    conv_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(conv_op, valid_places);
    DirectedLink(matched.at("input"), new_op_node);
    DirectedLink(matched.at("conv_filter"), new_op_node);
    DirectedLink(matched.at("bn_bias"), new_op_node);
    DirectedLink(matched.at("ew_x"), new_op_node);
    DirectedLink(max_filter_node, new_op_node);
    DirectedLink(new_op_node, matched.at("relu_out"));
    DirectedLink(new_op_node, max_output_node);
  }
};

}  // namespace fusion

class XPUConv2dFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;

    fusion::XPUConv2dBlock1Fuser fuser; /* branch fuse */
    fuser(graph.get());

    fusion::XPUConv2dBlock0Fuser fuser1(true /* with_relu */);
    fuser1(graph.get());

    fusion::XPUConv2dBlock0Fuser fuser2(false /* with_relu */);
    fuser2(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__conv2d_fuse_pass, paddle::lite::mir::XPUConv2dFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("conv2d");
