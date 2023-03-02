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

class MHSAfuser : public FuseBase {
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

    // x -> qx,qk,qv
    auto* q_mul_y =
        VarNode("q_mul_y")->assert_is_op_input("matmul_v2", "Y")->AsInput();
    auto* q_mul = OpNode("q_mul", "matmul_v2");
    auto* q_mul_out = VarNode("q_mul_out")
                          ->assert_is_op_output("matmul_v2", "Out")
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
    auto* q_transpose2 = OpNode("q_transpose2", "transpose2")->AsIntermediate();
    auto* q_transpose2_out = VarNode("q_transpose2_out")
                                 ->assert_is_op_output("transpose2", "Out")
                                 ->assert_is_op_input("matmul_v2", "X")
                                 ->AsIntermediate();
    auto* q_transpose2_xshape =
        VarNode("q_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();

    auto* k_mul_y =
        VarNode("k_mul_y")->assert_is_op_input("matmul_v2", "Y")->AsInput();
    auto* k_mul = OpNode("k_mul", "matmul_v2")->AsIntermediate();
    auto* k_mul_out = VarNode("k_mul_out")
                          ->assert_is_op_output("matmul_v2", "Out")
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
                                 ->AsIntermediate();
    auto* k_transpose2_xshape =
        VarNode("k_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();

    auto* qk_matmul = OpNode("qk_matmul", "matmul_v2")->AsIntermediate();
    auto* qk_matmul_out = VarNode("qk_matmul_out")
                              ->assert_is_op_output("matmul_v2", "Out")
                              ->assert_is_op_input("scale", "X")
                              ->AsIntermediate();
    auto* qk_scale =
        OpNode("qk_scale", "scale")
            ->AsIntermediate();  // TODO(wangshuaiwei): check scale value
    auto* qk_scale_out = VarNode("qk_scale_out")
                             ->assert_is_op_output("scale", "Out")
                             ->assert_is_op_input("softmax", "X")
                             ->AsIntermediate();
    auto* qk_softmax = OpNode("qk_softmax", "softmax")->AsIntermediate();
    auto* qk_softmax_out = VarNode("qk_softmax_out")
                               ->assert_is_op_output("softmax", "Out")
                               ->AsIntermediate();

    auto* v_mul_y =
        VarNode("v_mul_y")->assert_is_op_input("matmul_v2", "Y")->AsInput();
    auto* v_mul = OpNode("v_mul", "matmul_v2")->AsIntermediate();
    auto* v_mul_out = VarNode("v_mul_out")
                          ->assert_is_op_output("matmul_v2", "Out")
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
                                 ->assert_is_op_input("matmul_v2", "Y")
                                 ->AsIntermediate();
    auto* v_transpose2_xshape =
        VarNode("v_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();

    auto* qkv_matmul = OpNode("qkv_matmul", "matmul_v2")->AsIntermediate();
    auto* qkv_matmul_out = VarNode("qkv_matmul_out")
                               ->assert_is_op_output("matmul_v2", "Out")
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
                                 ->assert_is_op_input("matmul_v2", "X")
                                 ->AsIntermediate();
    auto* qkv_reshape2_xshape = VarNode("qkv_reshape2_xshape")
                                    ->assert_is_op_output("reshape2", "XShape")
                                    ->AsIntermediate();

    auto* qkv_mul_y =
        VarNode("qkv_mul_y")->assert_is_op_input("matmul_v2", "Y")->AsInput();
    auto* qkv_mul = OpNode("qkv_mul", "matmul_v2")->AsIntermediate();
    auto* qkv_mul_out = VarNode("qkv_mul_out")
                            ->assert_is_op_output("matmul_v2", "Out")
                            ->assert_is_op_input("elementwise_add", "X")
                            ->AsIntermediate();
    auto* qkv_add_y = VarNode("qkv_add_y")
                          ->assert_is_op_input("elementwise_add", "Y")
                          ->AsInput();
    auto* qkv_add = OpNode("qkv_add", "elementwise_add")->AsIntermediate();
    auto* qkv_add_out = VarNode("qkv_add_out")
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->AsOutput();
    // layer norm
    std::vector<PMNode*> ln_before_input{
        input, ln_before_scale, ln_before_bias};
    std::vector<PMNode*> ln_before_output{
        ln_before_out, ln_before_mean, ln_before_var};
    ln_before_input >> *ln_before >> ln_before_output;
    // q_info
    *ln_before_out >> *q_mul >> *q_mul_out >> *q_reshape2 >> *q_reshape2_out >>
        *q_transpose2 >> *q_transpose2_out >> *qk_matmul;
    *q_mul_y >> *q_mul;
    *q_reshape2 >> *q_reshape2_xshape;
    *q_transpose2 >> *q_transpose2_xshape;
    // k_info
    *ln_before_out >> *k_mul >> *k_mul_out >> *k_reshape2 >> *k_reshape2_out >>
        *k_transpose2 >> *k_transpose2_out >> *qk_matmul;
    *k_mul_y >> *k_mul;
    *k_reshape2 >> *k_reshape2_xshape;
    *k_transpose2 >> *k_transpose2_xshape;
    // qk_matmul
    *qk_matmul >> *qk_matmul_out >> *qk_scale >> *qk_scale_out >> *qk_softmax >>
        *qk_softmax_out >> *qkv_matmul;
    // v_info
    *v_mul >> *v_mul_out >> *v_reshape2 >> *v_reshape2_out >> *v_transpose2 >>
        *v_transpose2_out >> *qkv_matmul;
    *v_mul_y >> *v_mul;
    *v_reshape2 >> *v_reshape2_xshape;
    *v_transpose2 >> *v_transpose2_xshape;
    // after qkv_matmul
    *qkv_matmul >> *qkv_matmul_out >> *qkv_transpose2 >> *qkv_transpose2_out >>
        *qkv_reshape2 >> *qkv_reshape2_out >> *qkv_mul >> *qkv_mul_out >>
        *qkv_add >> *qkv_add_out;
    *qkv_transpose2 >> *qkv_transpose2_xshape;
    *qkv_reshape2 >> *qkv_reshape2_xshape;
    *qkv_mul_y >> *qkv_mul;
    *qkv_add_y >> *qkv_add;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__multihead_self_attn");
    std::vector<std::string> fc_weight_names = {
        matched.at("q_mul_y")->arg()->name,
        matched.at("k_mul_y")->arg()->name,
        matched.at("v_mul_y")->arg()->name,
        matched.at("qkv_mul_y")->arg()->name,
    };
    std::vector<std::string> fc_weight_maxptr_names;
    for (int i = 0; i < fc_weight_names.size(); i++) {
      fc_weight_maxptr_names.push_back(fc_weight_names[i] + "_max");
    }
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetInput("FCWeight", fc_weight_names);
    op_desc.SetInput("FCBias",
                     {
                         matched.at("qkv_add_y")->arg()->name,
                     });

    op_desc.SetInput("LNScale",
                     {
                         matched.at("ln_before_scale")->arg()->name,
                     });
    op_desc.SetInput("LNBias",
                     {
                         matched.at("ln_before_bias")->arg()->name,
                     });
    op_desc.SetOutput("Output", {matched.at("qkv_add_out")->arg()->name});
    op_desc.SetAttr<std::vector<std::string>>("FCWeightMax",
                                              fc_weight_maxptr_names);
    int hidden_dim = 0;
    auto* q_mul_op_info = matched.at("q_mul")->stmt()->op_info();
    auto q_mul_input_y_name = q_mul_op_info->Input("Y").front();
    auto* scope = matched.at("q_mul")->stmt()->op()->scope();
    auto q_mul_y_shape = scope->FindMutableTensor(q_mul_input_y_name)->dims();
    hidden_dim = q_mul_y_shape[0];

    auto* qkv_mul_op_info = matched.at("qkv_mul")->stmt()->op_info();
    auto qkv_mul_input_y_name = qkv_mul_op_info->Input("Y").front();
    auto qkv_mul_y_shape =
        scope->FindMutableTensor(qkv_mul_input_y_name)->dims();
    CHECK_EQ(q_mul_y_shape.size(), qkv_mul_y_shape.size());
    CHECK_EQ(q_mul_y_shape.size(), 2);
    CHECK_EQ(q_mul_y_shape[0], qkv_mul_y_shape[1]);
    CHECK_EQ(q_mul_y_shape[1], qkv_mul_y_shape[0]);
    CHECK_GT(hidden_dim, 0) << "invalid hidden_dim: " << hidden_dim;

    // set_quant_info(scope, matched, &op_desc);
    auto* reshape_op_info = matched.at("q_reshape2")->stmt()->op_info();
    auto reshape_dim = reshape_op_info->GetAttr<std::vector<int>>("shape");
    // scale attr must be equal to 1 / std::sqrt(size_per_head)
    int size_per_head = reshape_dim[3];
    auto* scale_op_info = matched.at("qk_scale")->stmt()->op_info();
    float scale_val = scale_op_info->GetAttr<float>("scale");
    float expected_value = 1.f / std::sqrt(size_per_head);
    CHECK(std::abs(expected_value - scale_val) < 1e-6f);
    op_desc.SetAttr<int>("head_num", reshape_dim[2]);
    op_desc.SetAttr<int>("size_per_head", size_per_head);
    CHECK_EQ(size_per_head * reshape_dim[2], q_mul_y_shape[1]);
    op_desc.SetAttr<int>("hidden_dim", hidden_dim);

    update_weight(scope, fc_weight_names, fc_weight_maxptr_names);

    auto mhsa_op = LiteOpRegistry::Global().Create(op_desc.Type());
    mhsa_op->Attach(op_desc, scope);
    mhsa_op->SetValidPlaces(matched.at("q_mul")->stmt()->op()->valid_places());
    auto kernels = mhsa_op->CreateKernels(mhsa_op->valid_places());
    matched.at("q_mul")->stmt()->SetOp(mhsa_op);
    matched.at("q_mul")->stmt()->SetKernels(std::move(kernels));

    std::vector<std::string> froms = {"k_mul_y",
                                      "v_mul_y",
                                      "qkv_mul_y",
                                      "qkv_add_y",
                                      "ln_before_scale",
                                      "ln_before_bias",
                                      "input"};

    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), matched.at("q_mul"));
    }
    // DirectedLink(matched.at("input"), new_op_node);
    IR_OP_VAR_LINK(matched.at("q_mul"), matched.at("qkv_add_out"));
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
      if (i > 0) {
        CHECK_EQ(weight_dims_vec[i][0], weight_dims_vec[i - 1][0]);
      }
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
fuse original subgraph into __xpu__multihead_self_attn op.

Original subgraph:

                        Input
                          |
                -----Layer Norm------
                |         |          |
                |         |          |
                |       matmul     matmul
                |         |          |
                |         |          |
                |       reshape    reshape
                |         |          |
                |         |          |
                |     transpose  transpose
                |         |          |
                |         \          /
                |          \        /
              matmul         matmul
                |              |
                |              |
             reshape         scale
                |              |
                |              |
            transpose        softmax
                |              |
                 \            /
                  \          /
                      matmul
                        |
                        |
                    transpose
                        |
                        |
                     reshape
                        |
                        |
                      matmul
                        |
                        |
                  elementwise_add
                        |
                      Output

Fuse to:
                      Input
                        |
                        |
            __xpu__multihead_self_attn
                        |
                        |
                      Output

*/

class XPUMHSAfusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    fusion::MHSAfuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__multihead_self_attn_fuse_pass,
                  paddle::lite::mir::XPUMHSAfusePass)
    .BindTargets({TARGET(kXPU)});
