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

class Geglufuser : public FuseBase {
 public:
  void BuildPattern() override {
    // layer norm
    auto* input = VarNode("input")
                      ->assert_is_op_input("layer_norm", "X")
                      ->assert_is_op_input("elementwise_add", "Y")
                      ->AsInput();
    auto* ln_before_scale = VarNode("ln_before_scale")
                                ->assert_is_op_input("layer_norm", "Scale")
                                ->AsInput();
    auto* ln_before_bias = VarNode("ln_before_bias")
                               ->assert_is_op_input("layer_norm", "Bias")
                               ->AsInput();
    auto* ln_before = OpNode("ln_before", "layer_norm")->AsIntermediate();
    auto* ln_before_out = VarNode("ln_before_out")
                              ->assert_is_op_output("layer_norm", "Y")
                              ->assert_is_op_input("matmul_v2", "X")
                              ->AsIntermediate();
    auto* ln_before_mean = VarNode("ln_before_mean")
                               ->assert_is_op_output("layer_norm", "Mean")
                               ->AsIntermediate();
    auto* ln_before_var = VarNode("ln_before_var")
                              ->assert_is_op_output("layer_norm", "Variance")
                              ->AsIntermediate();
    auto* mul_1_y =
        VarNode("mul_1_y")->assert_is_op_input("matmul_v2", "Y")->AsInput();
    auto* mul_1 = OpNode("mul_1", "matmul_v2")->AsIntermediate();
    auto* mul_1_out = VarNode("mul_1_out")
                          ->assert_is_op_output("matmul_v2", "Out")
                          ->assert_is_op_input("elementwise_add", "X")
                          ->AsIntermediate();
    auto* mul_1_add_y = VarNode("mul_1_add_y")
                            ->assert_is_op_input("elementwise_add", "Y")
                            ->AsInput();
    auto* mul_1_add = OpNode("mul_1_add", "elementwise_add")->AsIntermediate();
    auto* mul_1_add_out = VarNode("mul_1_add_out")
                              ->assert_is_op_output("elementwise_add", "Out")
                              ->assert_is_op_input("split", "X")
                              ->AsIntermediate();
    auto* split = OpNode("split", "split");
    auto* split_output_0 = VarNode("split_output_0")
                               ->assert_is_op_nth_output("split", "Out", 0)
                               ->assert_is_op_input("elementwise_mul", "X")
                               ->AsIntermediate();

    auto* split_output_1 = VarNode("split_output_1")
                               ->assert_is_op_nth_output("split", "Out", 1)
                               ->assert_is_op_input("gelu", "X")
                               ->AsIntermediate();
    auto* gelu = OpNode("gelu", "gelu")->AsIntermediate();
    auto* gelu_output = VarNode("gelu_output")
                            ->assert_is_op_output("gelu", "Out")
                            ->assert_is_op_input("elementwise_mul", "Y")
                            ->AsIntermediate();
    auto* elementwise_mul =
        OpNode("elementwise_mul", "elementwise_mul")->AsIntermediate();
    auto* ew_mul_out = VarNode("ew_mul_out")
                           ->assert_is_op_output("elementwise_mul", "Out")
                           ->assert_is_op_input("matmul_v2", "X")
                           ->AsIntermediate();
    auto* mul_2_y =
        VarNode("mul_2_y")->assert_is_op_input("matmul_v2", "Y")->AsInput();
    auto* mul_2 = OpNode("mul_2", "matmul_v2")->AsIntermediate();
    auto* mul_2_out = VarNode("mul_2_out")
                          ->assert_is_op_output("matmul_v2", "Out")
                          ->assert_is_op_input("elementwise_add", "X")
                          ->AsIntermediate();
    auto* mul_2_add_y = VarNode("mul_2_add_y")
                            ->assert_is_op_input("elementwise_add", "Y")
                            ->AsInput();
    auto* mul_2_add = OpNode("mul_2_add", "elementwise_add")->AsIntermediate();
    auto* mul_2_add_out = VarNode("mul_2_add_out")
                              ->assert_is_op_output("elementwise_add", "Out")
                              ->AsOutput();
    // layer norm
    std::vector<PMNode*> ln_before_input{
        input, ln_before_scale, ln_before_bias};
    std::vector<PMNode*> ln_before_output{
        ln_before_out, ln_before_mean, ln_before_var};
    ln_before_input >> *ln_before >> ln_before_output;
    *ln_before_out >> *mul_1 >> *mul_1_out >> *mul_1_add >> *mul_1_add_out;
    *mul_1_y >> *mul_1;
    *mul_1_add_y >> *mul_1_add;
    *mul_1_add_out >> *split >> *split_output_1 >> *gelu >> *gelu_output >>
        *elementwise_mul >> *ew_mul_out;
    *split >> *split_output_0 >> *elementwise_mul;
    *ew_mul_out >> *mul_2 >> *mul_2_out >> *mul_2_add >> *mul_2_add_out;
    *mul_2_y >> *mul_2;
    *mul_2_add_y >> *mul_2_add;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__geglu");
    std::vector<std::string> fc_weight_names = {
        matched.at("mul_1_y")->arg()->name, matched.at("mul_2_y")->arg()->name,
    };
    std::vector<std::string> fc_weight_maxptr_names;
    for (int i = 0; i < fc_weight_names.size(); i++) {
      fc_weight_maxptr_names.push_back(fc_weight_names[i] + "_max");
    }
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetInput("FCWeight", fc_weight_names);
    op_desc.SetInput("FCBias",
                     {
                         matched.at("mul_1_add_y")->arg()->name,
                         matched.at("mul_2_add_y")->arg()->name,
                     });
    op_desc.SetInput("LNScale",
                     {
                         matched.at("ln_before_scale")->arg()->name,
                     });
    op_desc.SetInput("LNBias",
                     {
                         matched.at("ln_before_bias")->arg()->name,
                     });
    op_desc.SetOutput("Output", {matched.at("mul_2_add_out")->arg()->name});
    op_desc.SetAttr<std::vector<std::string>>("FCWeightMax",
                                              fc_weight_maxptr_names);

    int hidden_dim = 0;
    int gelu_dim = 0;
    auto* mul_1_op_info = matched.at("mul_1")->stmt()->op_info();
    auto mul_1_input_y_name = mul_1_op_info->Input("Y").front();
    auto* scope = matched.at("split")->stmt()->op()->scope();
    auto mul_1_y_shape = scope->FindMutableTensor(mul_1_input_y_name)->dims();
    hidden_dim = mul_1_y_shape[0];
    gelu_dim = mul_1_y_shape[1] / 2;
    op_desc.SetAttr<int>("hidden_dim", hidden_dim);
    op_desc.SetAttr<int>("gelu_dim", gelu_dim);
    update_weight(scope, fc_weight_names, fc_weight_maxptr_names);
    auto geglu_op = LiteOpRegistry::Global().Create(op_desc.Type());
    geglu_op->Attach(op_desc, scope);
    geglu_op->SetValidPlaces(matched.at("split")->stmt()->op()->valid_places());
    auto kernels = geglu_op->CreateKernels(geglu_op->valid_places());
    matched.at("split")->stmt()->SetOp(geglu_op);
    matched.at("split")->stmt()->SetKernels(std::move(kernels));

    std::vector<std::string> froms = {
        "mul_2_add_y",
        "mul_2_y",
        "mul_1_add_y",
        "mul_1_y",
        "ln_before_scale",
        "ln_before_bias",
        "input",
    };
    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), matched.at("split"));
    }
    IR_OP_VAR_LINK(matched.at("split"), matched.at("mul_2_add_out"));
  }

 private:
  void update_weight(Scope* scope,
                     const std::vector<std::string>& fc_weight_names,
                     const std::vector<std::string>& fc_weight_max_names) {
    std::vector<Tensor*> weight_tensor_vec(fc_weight_names.size(), nullptr);
    std::vector<DDimLite> weight_dims_vec(fc_weight_names.size());
    std::vector<int> weight_len_vec(fc_weight_names.size());

    for (int i = 0; i < fc_weight_names.size(); ++i) {
      weight_tensor_vec[i] = scope->FindMutableTensor(fc_weight_names[i]);
      CHECK(weight_tensor_vec[i] != nullptr);
      weight_dims_vec[i] = weight_tensor_vec[i]->dims();
      weight_len_vec[i] = weight_tensor_vec[i]->numel();
    }
    for (int i = 0; i < fc_weight_names.size(); ++i) {
      float* weight_host_ptr = weight_tensor_vec[i]->mutable_data<float>();
      std::unique_ptr<float[]> weight_host_trans(new float[weight_len_vec[i]]);
      std::unique_ptr<int16_t[]> weight_host_trans_int16(
          new int16_t[weight_len_vec[i]]);
      paddle::lite::xpu::math::Transpose<float>(weight_host_ptr,
                                                weight_host_trans.get(),
                                                weight_dims_vec[i][0],
                                                weight_dims_vec[i][1]);
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
};

}  // namespace fusion

/*
fuse original subgraph into __xpu_geglu op.

Original subgraph:

                          Input
                            |
                            |
                        layer norm
                            |
                            |
                          matmul
                            |
                            |
                      elementwise_add
                            |
                            |
                          split
                           / \
                          /   \
                          |  gelu
                          |   |
                          \   /
                      elementwise_mul
                            |
                            |
                          matmul
                            |
                            |
                      elementwise_add
                            |
                            |
                          Output

Fused to:
                          Input
                            |
                            |
                      __xpu_geglu op
                            |
                            |
                          Output


*/

class XPUGeglufusePassop : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    fusion::Geglufuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__geglu_fuse_pass, paddle::lite::mir::XPUGeglufusePassop)
    .BindTargets({TARGET(kXPU)});
