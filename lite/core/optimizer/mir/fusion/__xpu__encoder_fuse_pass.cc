// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

#include <memory>
#include <set>
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/xpu_pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

namespace fusion {

class XPUEncoderFuser : public FuseBase {
 public:
  explicit XPUEncoderFuser(bool with_mask = true,
                           bool norm_before = false,
                           bool enable_qkv_fusion = true)
      : with_mask_(with_mask),
        norm_before_(norm_before),
        enable_qkv_fusion_(enable_qkv_fusion) {}

  void BuildPattern() override {
    auto* input = VarNode("input")->AsInput();
    // layernorm
    PMNode* ln_before_scale = nullptr;
    PMNode* ln_before_bias = nullptr;
    PMNode* ln_before = nullptr;
    PMNode* ln_before_out = nullptr;
    PMNode* ln_before_mean = nullptr;
    PMNode* ln_before_var = nullptr;
    if (norm_before_) {
      input->assert_is_op_input("layer_norm", "X");
      ln_before = OpNode("ln_before", "layer_norm")->AsIntermediate();
      ln_before_scale = VarNode("ln_before_scale")
                            ->assert_is_op_input("layer_norm", "Scale")
                            ->AsInput();
      ln_before_bias = VarNode("ln_before_bias")
                           ->assert_is_op_input("layer_norm", "Bias")
                           ->AsInput();
      ln_before_out = VarNode("ln_before_out")
                          ->assert_is_op_output("layer_norm", "Y")
                          ->assert_is_op_input("__xpu__fc", "Input")
                          ->AsIntermediate();
      ln_before_mean = VarNode("ln_before_mean")
                           ->assert_is_op_output("layer_norm", "Mean")
                           ->AsIntermediate();
      ln_before_var = VarNode("ln_before_var")
                          ->assert_is_op_output("layer_norm", "Variance")
                          ->AsIntermediate();
    } else {
      input->assert_is_op_input("__xpu__fc", "Input");
    }

    // multi-head attention
    // q
    auto* q_fc = OpNode("q_fc", "__xpu__fc")
                     ->assert_op_attr<int>("act_type", 0)  // linear
                     ->AsIntermediate();
    auto* q_fc_w = VarNode("q_fc_w")->assert_is_op_input("__xpu__fc", "Filter");
    auto* q_fc_bias = VarNode("q_fc_bias")
                          ->assert_is_op_input("__xpu__fc", "Bias")
                          ->AsInput();
    auto* q_fc_maxo = VarNode("q_fc_maxo")
                          ->assert_is_op_output("__xpu__fc", "OutputMax")
                          ->AsIntermediate();
    auto* q_fc_out = VarNode("q_fc_out")
                         ->assert_is_op_input("__xpu__qk_attention", "q")
                         ->assert_is_op_output("__xpu__fc", "Output")
                         ->AsIntermediate();

    // k
    auto* k_fc = OpNode("k_fc", "__xpu__fc")
                     ->assert_op_attr<int>("act_type", 0)  // linear
                     ->AsIntermediate();
    auto* k_fc_w =
        VarNode("k_fc_w")->assert_is_op_input("__xpu__fc", "Filter")->AsInput();
    auto* k_fc_bias = VarNode("k_fc_bias")
                          ->assert_is_op_input("__xpu__fc", "Bias")
                          ->AsInput();
    auto* k_fc_maxo = VarNode("k_fc_maxo")
                          ->assert_is_op_output("__xpu__fc", "OutputMax")
                          ->AsIntermediate();
    auto* k_fc_out = VarNode("k_fc_out")
                         ->assert_is_op_input("__xpu__qk_attention", "k")
                         ->assert_is_op_output("__xpu__fc", "Output")
                         ->AsIntermediate();

    // v
    auto* v_fc = OpNode("v_fc", "__xpu__fc")
                     ->assert_op_attr<int>("act_type", 0)  // linear
                     ->AsIntermediate();
    auto* v_fc_w =
        VarNode("v_fc_w")->assert_is_op_input("__xpu__fc", "Filter")->AsInput();
    auto* v_fc_bias = VarNode("v_fc_bias")
                          ->assert_is_op_input("__xpu__fc", "Bias")
                          ->AsInput();
    auto* v_fc_maxo = VarNode("v_fc_maxo")
                          ->assert_is_op_output("__xpu__fc", "OutputMax")
                          ->AsIntermediate();
    auto* v_fc_out = VarNode("v_fc_out")
                         ->assert_is_op_input("__xpu__qk_v_attention", "v")
                         ->assert_is_op_output("__xpu__fc", "Output")
                         ->AsIntermediate();

    // qk attention
    auto* qk_att = OpNode("qk_att", "__xpu__qk_attention")->AsIntermediate();
    PMNode* qk_mask = nullptr;
    if (with_mask_) {
      qk_mask = VarNode("qk_mask")
                    ->assert_is_op_input("__xpu__qk_attention", "mask")
                    ->AsInput();
    } else {
      qk_att->assert_node_satisfied([](const Node* node) -> bool {
        auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
        return !op_desc.HasInput("mask");
      });
    }
    auto* qk_out = VarNode("qk_out")
                       ->assert_is_op_input("__xpu__qk_v_attention", "qk")
                       ->assert_is_op_output("__xpu__qk_attention", "output")
                       ->AsIntermediate();

    // qk_v attention
    auto* qk_v_att =
        OpNode("qk_v_att", "__xpu__qk_v_attention")->AsIntermediate();
    auto* qk_v_out =
        VarNode("qk_v_out")
            ->assert_is_op_input("__xpu__fc", "Input")
            ->assert_is_op_output("__xpu__qk_v_attention", "output")
            ->AsIntermediate();

    auto* qkv_fc = OpNode("qkv_fc", "__xpu__fc")
                       ->assert_op_attr<int>("act_type", 0)  // linear
                       ->AsIntermediate();
    auto* qkv_fc_w = VarNode("qkv_fc_w")
                         ->assert_is_op_input("__xpu__fc", "Filter")
                         ->AsInput();
    auto* qkv_fc_bias = VarNode("qkv_fc_bias")
                            ->assert_is_op_input("__xpu__fc", "Bias")
                            ->AsInput();
    auto* qkv_fc_maxo = VarNode("qkv_fc_maxo")
                            ->assert_is_op_output("__xpu__fc", "OutputMax")
                            ->AsIntermediate();
    auto* qkv_fc_out = VarNode("qkv_fc_out")
                           // ->assert_is_op_input("elementwise_add","Y") X or Y
                           ->assert_is_op_output("__xpu__fc", "Output")
                           ->AsIntermediate();

    // add & layernorm after multi-head attention
    auto* att_add = OpNode("att_add", "elementwise_add")->AsIntermediate();
    auto* att_add_out = VarNode("att_add_out")
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->assert_is_op_input("layer_norm", "X")
                            ->AsIntermediate();

    auto* att_ln = OpNode("att_ln", "layer_norm")->AsIntermediate();
    auto* att_ln_scale = VarNode("att_ln_scale")
                             ->assert_is_op_input("layer_norm", "Scale")
                             ->AsInput();
    auto* att_ln_bias = VarNode("att_ln_bias")
                            ->assert_is_op_input("layer_norm", "Bias")
                            ->AsInput();
    auto* att_ln_out = VarNode("att_ln_out")
                           ->assert_is_op_output("layer_norm", "Y")
                           ->assert_is_op_input("__xpu__fc", "Input")
                           ->AsIntermediate();
    auto* att_ln_mean = VarNode("att_ln_mean")
                            ->assert_is_op_output("layer_norm", "Mean")
                            ->AsIntermediate();
    auto* att_ln_var = VarNode("att_ln_var")
                           ->assert_is_op_output("layer_norm", "Variance")
                           ->AsIntermediate();

    // ffn
    auto* ffn_fc0 = OpNode("ffn_fc0", "__xpu__fc")->AsIntermediate();
    auto* ffn_fc0_w = VarNode("ffn_fc0_w")
                          ->assert_is_op_input("__xpu__fc", "Filter")
                          ->AsInput();
    auto* ffn_fc0_bias = VarNode("ffn_fc0_bias")
                             ->assert_is_op_input("__xpu__fc", "Bias")
                             ->AsInput();
    auto* ffn_fc0_maxo = VarNode("ffn_fc0_maxo")
                             ->assert_is_op_output("__xpu__fc", "OutputMax")
                             ->AsIntermediate();
    auto* ffn_fc0_out = VarNode("ffn_fc0_out")
                            ->assert_is_op_input("__xpu__fc", "Input")
                            ->assert_is_op_output("__xpu__fc", "Output")
                            ->AsIntermediate();

    auto* ffn_fc1 = OpNode("ffn_fc1", "__xpu__fc")
                        ->assert_op_attr<int>("act_type", 0)  // linear
                        ->AsIntermediate();
    auto* ffn_fc1_w = VarNode("ffn_fc1_w")
                          ->assert_is_op_input("__xpu__fc", "Filter")
                          ->AsInput();
    auto* ffn_fc1_bias = VarNode("ffn_fc1_bias")
                             ->assert_is_op_input("__xpu__fc", "Bias")
                             ->AsInput();
    auto* ffn_fc1_maxo = VarNode("ffn_fc1_maxo")
                             ->assert_is_op_output("__xpu__fc", "OutputMax")
                             ->AsIntermediate();
    auto* ffn_fc1_out = VarNode("ffn_fc1_out")
                            ->assert_is_op_output("__xpu__fc", "Output")
                            ->AsIntermediate();

    // add & layernorm after ffn
    auto* ffn_add = OpNode("ffn_add", "elementwise_add")->AsIntermediate();
    auto* ffn_add_out =
        VarNode("ffn_add_out")->assert_is_op_output("elementwise_add", "Out");
    PMNode* ffn_ln = nullptr;
    PMNode* ffn_ln_scale = nullptr;
    PMNode* ffn_ln_bias = nullptr;
    PMNode* ffn_ln_out = nullptr;
    PMNode* ffn_ln_mean = nullptr;
    PMNode* ffn_ln_var = nullptr;
    if (norm_before_) {
      ffn_add_out->AsOutput();
    } else {
      ffn_add_out->assert_is_op_input("layer_norm", "X")->AsIntermediate();
      ffn_ln = OpNode("ffn_ln", "layer_norm")->AsIntermediate();
      ffn_ln_scale = VarNode("ffn_ln_scale")
                         ->assert_is_op_input("layer_norm", "Scale")
                         ->AsInput();
      ffn_ln_bias = VarNode("ffn_ln_bias")
                        ->assert_is_op_input("layer_norm", "Bias")
                        ->AsInput();
      ffn_ln_out = VarNode("ffn_ln_out")
                       ->assert_is_op_output("layer_norm", "Y")
                       ->AsOutput();
      ffn_ln_mean = VarNode("ffn_ln_mean")
                        ->assert_is_op_output("layer_norm", "Mean")
                        ->AsIntermediate();
      ffn_ln_var = VarNode("ffn_ln_var")
                       ->assert_is_op_output("layer_norm", "Variance")
                       ->AsIntermediate();
    }

    // Links
    if (norm_before_) {
      ln_before->LinksFrom({input, ln_before_scale, ln_before_bias})
          .LinksTo({ln_before_out, ln_before_mean, ln_before_var});
      ln_before_out->LinksTo({q_fc, k_fc, v_fc});
    } else {
      input->LinksTo({q_fc, k_fc, v_fc});
    }
    q_fc->LinksFrom({q_fc_w, q_fc_bias}).LinksTo({q_fc_maxo, q_fc_out});
    k_fc->LinksFrom({k_fc_w, k_fc_bias}).LinksTo({k_fc_maxo, k_fc_out});
    v_fc->LinksFrom({v_fc_w, v_fc_bias}).LinksTo({v_fc_maxo, v_fc_out});
    if (with_mask_) {
      qk_att->LinksFrom({q_fc_out, k_fc_out, qk_mask}).LinksTo({qk_out});
    } else {
      qk_att->LinksFrom({q_fc_out, k_fc_out}).LinksTo({qk_out});
    }
    qk_v_att->LinksFrom({qk_out, v_fc_out}).LinksTo({qk_v_out});
    qkv_fc->LinksFrom({qk_v_out, qkv_fc_w, qkv_fc_bias})
        .LinksTo({qkv_fc_maxo, qkv_fc_out});
    att_add->LinksFrom({input, qkv_fc_out}).LinksTo({att_add_out});
    att_ln->LinksFrom({att_add_out, att_ln_scale, att_ln_bias})
        .LinksTo({att_ln_out, att_ln_mean, att_ln_var});
    ffn_fc0->LinksFrom({att_ln_out, ffn_fc0_w, ffn_fc0_bias})
        .LinksTo({ffn_fc0_maxo, ffn_fc0_out});
    ffn_fc1->LinksFrom({ffn_fc0_out, ffn_fc1_w, ffn_fc1_bias})
        .LinksTo({ffn_fc1_maxo, ffn_fc1_out});
    if (norm_before_) {
      ffn_add->LinksFrom({att_add_out, ffn_fc1_out}).LinksTo({ffn_add_out});
    } else {
      ffn_add->LinksFrom({ffn_fc1_out, att_ln_out}).LinksTo({ffn_add_out});
      ffn_ln->LinksFrom({ffn_add_out, ffn_ln_scale, ffn_ln_bias})
          .LinksTo({ffn_ln_out, ffn_ln_mean, ffn_ln_var});
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    VLOG(3) << "______xpu_encoder_____";
    auto* scope = matched.at("q_fc")->stmt()->op()->scope();
    auto valid_places = matched.at("q_fc")->stmt()->op()->valid_places();
    cpp::OpDesc op_desc;
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("__xpu__encoder");

    // define nodes names
    std::vector<std::string> gemm_nodes = {"q_fc",
                                           "k_fc",
                                           "v_fc",
                                           "qkv_fc",
                                           "ffn_fc0",
                                           "ffn_fc1",
                                           "qk_att",
                                           "qk_v_att"};
    std::vector<std::string> ln_nodes;
    if (norm_before_) {
      ln_nodes.push_back("ln_before");
      ln_nodes.push_back("att_ln");
    } else {
      ln_nodes.push_back("att_ln");
      ln_nodes.push_back("ffn_ln");
    }

    // Set Attributes
    op_desc.SetAttr<int>("n_layers", 1);
    op_desc.SetAttr<bool>("enable_qkv_fusion", enable_qkv_fusion_);
    op_desc.SetAttr<bool>("adaptive_seqlen", false);
    op_desc.SetAttr<bool>("do_padding", false);
    op_desc.SetAttr<bool>("do_slice", false);
    op_desc.SetAttr<bool>("norm_before", norm_before_);

    int act_type =
        matched.at("ffn_fc0")->stmt()->op_info()->GetAttr<int>("act_type");
    op_desc.SetAttr<int>("act_type", act_type);
    int hidden_dim =
        scope->FindMutableTensor(matched.at("q_fc_w")->arg()->name)->dims()[0];
    op_desc.SetAttr<int>("hidden_dim", hidden_dim);
    int head_num =
        matched.at("qk_att")->stmt()->op_info()->GetAttr<int>("head_num");
    int head_dim =
        matched.at("qk_att")->stmt()->op_info()->GetAttr<int>("head_dim");
    op_desc.SetAttr<int>("head_num", head_num);
    op_desc.SetAttr<int>("head_dim", head_dim);
    float alpha =
        matched.at("qk_att")->stmt()->op_info()->GetAttr<float>("alpha");
    op_desc.SetAttr<float>("alpha", alpha);
    int intermediate_size =
        scope->FindMutableTensor(matched.at("ffn_fc0_w")->arg()->name)
            ->dims()[1];
    op_desc.SetAttr<int>("intermediate_size", intermediate_size);

    // Set quant attributes
    std::vector<std::string> precision;
    std::vector<std::string> quant_type;
    std::vector<float> weight_max;  // 6
    std::vector<float> io_max;      // 18
    for (std::string& node : gemm_nodes) {
      quant_type.push_back(
          matched.at(node)->stmt()->op_info()->GetAttr<std::string>(
              "quant_type"));
      precision.push_back(
          matched.at(node)->stmt()->op_info()->GetAttr<std::string>(
              "precision"));
    }
    // fc
    for (int i = 0; i < 6; ++i) {
      if (quant_type[i] == "per_tensor" || quant_type[i] == "per_channel") {
        float in_max = matched.at(gemm_nodes[i])
                           ->stmt()
                           ->op_info()
                           ->GetAttr<std::vector<float>>("Input0_scale")[0];
        float out_max = matched.at(gemm_nodes[i])
                            ->stmt()
                            ->op_info()
                            ->GetAttr<std::vector<float>>("Output0_scale")[0];
        if (act_type == 4) {
          // Use gelu10 according to whitepaper http://arxiv.org/abs/2004.09602
          // commit ID: 9d636b7e61f88317b65ae5959539cbedad7f8849 (LiuWei)
          float gelu_limit_value =
              GetDoubleFromEnv("QUANT_GELU_OUT_THRESHOLD", 10.f);
          CHECK_GT(gelu_limit_value, 0.f)
              << "QUANT_GELU_OUT_THRESHOLD should be an positive float value: "
              << gelu_limit_value;
          if (i == 4) {
            out_max = std::min(gelu_limit_value, out_max);
          } else if (i == 5) {
            in_max = std::min(gelu_limit_value, in_max);
          }
        }
        io_max.push_back(in_max);
        io_max.push_back(out_max);
        std::vector<float> fc_weight_max =
            matched.at(gemm_nodes[i])
                ->stmt()
                ->op_info()
                ->GetAttr<std::vector<float>>("Filter0_scale");
        weight_max.insert(
            weight_max.end(), fc_weight_max.begin(), fc_weight_max.end());
      } else {
        io_max.push_back(-1.0);
        io_max.push_back(-1.0);
        weight_max.push_back(-1.0);
      }
    }
    if (enable_qkv_fusion_ &&
        (quant_type[0] == "per_tensor" || quant_type[0] == "per_channel")) {
      std::string fc_quant_type = quant_type[0];
      int weight_max_size = matched.at(gemm_nodes[0])
                                ->stmt()
                                ->op_info()
                                ->GetAttr<std::vector<float>>("Filter0_scale")
                                .size();
      float input_max = io_max[0];
      float output_max = std::max(io_max[1], std::max(io_max[3], io_max[5]));
      for (int i = 0; i < 3; ++i) {
        CHECK_EQ(fc_quant_type, quant_type[i]);
        CHECK_EQ(weight_max_size,
                 matched.at(gemm_nodes[i])
                     ->stmt()
                     ->op_info()
                     ->GetAttr<std::vector<float>>("Filter0_scale")
                     .size());
        CHECK(std::abs(input_max - io_max[2 * i]) < 1e-4);
        io_max[2 * i + 1] = output_max;
      }
      if (fc_quant_type == "per_tensor") {
        CHECK_EQ(weight_max[0], weight_max[1]);
        CHECK_EQ(weight_max[1], weight_max[2]);
      }
    }
    // qk & qk_v attention
    for (int i = 6; i < 8; ++i) {
      if (quant_type[i] == "per_tensor" || quant_type[i] == "per_channel") {
        io_max.push_back(matched.at(gemm_nodes[i])
                             ->stmt()
                             ->op_info()
                             ->GetAttr<std::vector<float>>("input_scale")[0]);
        io_max.push_back(matched.at(gemm_nodes[i])
                             ->stmt()
                             ->op_info()
                             ->GetAttr<std::vector<float>>("input_scale")[1]);
        io_max.push_back(matched.at(gemm_nodes[i])
                             ->stmt()
                             ->op_info()
                             ->GetAttr<std::vector<float>>("output_scale")[0]);
      } else {
        for (int j = 0; j < 3; ++j) {
          io_max.push_back(-1.0);
        }
      }
    }
    op_desc.SetAttr<std::vector<float>>("weight_max", weight_max);
    op_desc.SetAttr<std::vector<float>>("io_max", io_max);
    op_desc.SetAttr<std::vector<std::string>>("precision", precision);
    op_desc.SetAttr<std::vector<std::string>>("quant_type", quant_type);

    // set input & outputs
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    if (with_mask_) {
      op_desc.SetInput("Mask", {matched.at("qk_mask")->arg()->name});
    }

    std::vector<std::string> fc_w;
    std::vector<std::string> fc_bias;
    for (int i = 0; i < 6; ++i) {
      fc_bias.push_back(matched.at(gemm_nodes[i] + "_bias")->arg()->name);
    }
    if (enable_qkv_fusion_) {
      // TODO(TingShen): Consider two shared-bias encoders, whoose
      // enable_qkv_fusion_ are not equal.
      Tensor* q_tensor = scope->FindMutableTensor(fc_bias[0]);
      Tensor* k_tensor = scope->FindMutableTensor(fc_bias[1]);
      if (q_tensor->numel() == k_tensor->numel()) {
        int qkv_len = k_tensor->dims()[0] * 3;
        std::unique_ptr<float[]> bias_qkv(new float[qkv_len]);
        int bias_len = k_tensor->numel();
        for (int i = 0; i < 3; ++i) {
          Tensor* tensor = scope->FindMutableTensor(fc_bias[i]);
          CHECK_EQ(tensor->numel(), bias_len);
          CHECK_EQ(tensor->dims().size(), 1);
          memcpy(bias_qkv.get() + bias_len * i,
                 tensor->data<float>(),
                 bias_len * sizeof(float));
        }
        q_tensor->Resize({qkv_len});
        memcpy(q_tensor->mutable_data<float>(),
               bias_qkv.get(),
               qkv_len * sizeof(float));
      }
    }
    fc_w = update_weight(
        scope, matched, gemm_nodes, quant_type, precision, enable_qkv_fusion_);
    op_desc.SetInput("FCWeight", fc_w);
    op_desc.SetInput("FCBias", fc_bias);

    std::vector<std::string> ln_scale;
    std::vector<std::string> ln_bias;
    for (auto& ln_node : ln_nodes) {
      ln_scale.push_back(matched.at(ln_node + "_scale")->arg()->name);
      ln_bias.push_back(matched.at(ln_node + "_bias")->arg()->name);
    }
    op_desc.SetInput("LNScale", ln_scale);
    op_desc.SetInput("LNBias", ln_bias);
    if (norm_before_) {
      op_desc.SetOutput("Output", {matched.at("ffn_add_out")->arg()->name});
    } else {
      op_desc.SetOutput("Output", {matched.at("ffn_ln_out")->arg()->name});
    }

    auto op = LiteOpRegistry::Global().Create("__xpu__encoder");
    op->Attach(op_desc, scope);
    op->SetValidPlaces(valid_places);
    auto* op_node = graph->GraphCreateInstructNode(op, valid_places);

    // Links
    std::vector<std::string> Input_Nodes;
    Input_Nodes.push_back("input");
    if (with_mask_) {
      Input_Nodes.push_back("qk_mask");
    }
    for (int i = 0; i < 6; ++i) {
      // Input_Nodes.push_back(gemm_nodes[i] + "_w");
      Input_Nodes.push_back(gemm_nodes[i] + "_bias");
    }

    for (auto& ln_node : ln_nodes) {
      Input_Nodes.push_back(ln_node + "_scale");
      Input_Nodes.push_back(ln_node + "_bias");
    }
    if (norm_before_) {
      DirectedLink(op_node, matched.at("ffn_add_out"));
    } else {
      DirectedLink(op_node, matched.at("ffn_ln_out"));
    }
    for (auto& input_node : Input_Nodes) {
      DirectedLink(matched.at(input_node), op_node);
    }
  }

 private:
  bool with_mask_;
  bool norm_before_;
  bool enable_qkv_fusion_;

  std::vector<std::string> update_weight(
      Scope* scope,
      const key2nodes_t& matched,
      const std::vector<std::string>& gemm_nodes,
      const std::vector<std::string>& quant_type,
      const std::vector<std::string>& precision,
      bool enable_qkv_fusion) {
    std::vector<std::string> fc_weight_names;
    for (int i = 0; i < 6; ++i) {
      std::string weight_name = matched.at(gemm_nodes[i] + "_w")->arg()->name;
      std::string weight_trans_name = weight_name + "_trans";
      fc_weight_names.push_back(weight_trans_name);
      // matched.at(gemm_nodes[i] + "_w") -> AsArg(weight_trans_name);
      Tensor* weight_trans_tensor =
          const_cast<Tensor*>(scope->FindTensor(weight_trans_name));
      if (weight_trans_tensor != nullptr) continue;
      Tensor* weight_tensor = scope->FindMutableTensor(weight_name);
      CHECK(weight_tensor != nullptr);
      weight_trans_tensor = scope->NewTensor(weight_trans_name);
      weight_trans_tensor->CopyDataFrom(*weight_tensor);
      weight_trans_tensor->Resize(
          {weight_tensor->dims()[1], weight_tensor->dims()[0]});
      if (quant_type[i] == "per_tensor" || quant_type[i] == "per_channel") {
        if (precision[i] == "int8") {
          paddle::lite::xpu::math::Transpose<int8_t>(
              weight_tensor->data<int8_t>(),
              weight_trans_tensor->mutable_data<int8_t>(),
              weight_tensor->dims()[0],
              weight_tensor->dims()[1]);
        } else if (precision[i] == "int16") {
          paddle::lite::xpu::math::Transpose<int16_t>(
              weight_tensor->data<int16_t>(),
              weight_trans_tensor->mutable_data<int16_t>(),
              weight_tensor->dims()[0],
              weight_tensor->dims()[1]);
        } else {
          LOG(FATAL) << "Unsupported fc quant type";
        }
      } else {
        paddle::lite::xpu::math::Transpose<float>(
            weight_tensor->data<float>(),
            weight_trans_tensor->mutable_data<float>(),
            weight_tensor->dims()[0],
            weight_tensor->dims()[1]);
      }
    }
    if (!enable_qkv_fusion) return fc_weight_names;
    std::string qkv_fused_name = fc_weight_names[0] + "_qkv_fused";
    Tensor* qkv_fused_tensor = scope->NewTensor(qkv_fused_name);
    Tensor* q_tensor = scope->FindMutableTensor(fc_weight_names[0]);
    qkv_fused_tensor->CopyDataFrom(*q_tensor);
    qkv_fused_tensor->Resize({q_tensor->dims()[0] * 3, q_tensor->dims()[1]});
    int weight_len = q_tensor->numel();
    for (int i = 0; i < 3; ++i) {
      Tensor* tensor = scope->FindMutableTensor(fc_weight_names[i]);
      CHECK_EQ(tensor->numel(), weight_len);
      CHECK_EQ(quant_type[i], quant_type[0]);
      CHECK_EQ(precision[i], precision[0]);
      if (quant_type[i] == "per_tensor" || quant_type[i] == "per_channel") {
        if (precision[i] == "int8") {
          memcpy(qkv_fused_tensor->mutable_data<int8_t>() + weight_len * i,
                 tensor->data<int8_t>(),
                 weight_len * sizeof(int8_t));
        } else if (precision[i] == "int16") {
          memcpy(qkv_fused_tensor->mutable_data<int16_t>() + weight_len * i,
                 tensor->data<int16_t>(),
                 weight_len * sizeof(int16_t));
        } else {
          LOG(FATAL) << "Unsupported fc quant type";
        }
      } else {
        memcpy(qkv_fused_tensor->mutable_data<float>() + weight_len * i,
               tensor->data<float>(),
               weight_len * sizeof(float));
      }
    }
    fc_weight_names[0] = qkv_fused_name;
    return fc_weight_names;
  }
};

}  // namespace fusion

class XPUEncoderFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    std::vector<bool> with_masks{true, false};
    std::vector<bool> norm_befores{true, false};

    for (auto with_mask : with_masks) {
      for (auto norm_before : norm_befores) {
        fusion::XPUEncoderFuser fuser(with_mask, norm_before, true);
        fuser(graph.get());
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__encoder_fuse_pass,
                  paddle::lite::mir::XPUEncoderFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__encoder");
