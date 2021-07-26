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
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/xpu_pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class XPUMmdnnFloat2Fix {
 public:
  void operator()(SSAGraph* graph) {
    for (auto* node : graph->StmtTopologicalOrder()) {
      CHECK(node->IsStmt());
      auto* op_info = node->stmt()->op_info();
      std::string op_type = op_info->Type();

      static const std::vector<std::string> target_ops{"var_conv_2d",
                                                       "search_fc"};
      if (std::find(target_ops.begin(), target_ops.end(), op_type) !=
          target_ops.end()) {
        std::string weight_name = op_info->Input("W").front();
        auto* scope = node->stmt()->op()->scope();
        auto* weight_t = scope->FindMutableTensor(weight_name);
        auto weight_dims = weight_t->dims();
        auto weight_len = weight_t->numel();
        float* weight_on_host = weight_t->mutable_data<float>();
        float max_f =
            paddle::lite::xpu::math::FindMaxAbs(weight_on_host, weight_len);
        std::unique_ptr<int16_t[]> weight_int16(new int16_t[weight_len]);
        paddle::lite::xpu::math::ConvertFP32ToInt16(
            weight_on_host, weight_int16.get(), max_f, weight_len);
        memcpy(
            weight_on_host, weight_int16.get(), weight_len * sizeof(int16_t));

        auto update_op_info = *op_info;
        update_op_info.SetAttr<bool>("__xpu__float_to_fix", true);
        update_op_info.SetAttr<float>("__xpu__w_max", max_f);
        node->stmt()->ResetOp(update_op_info, graph->valid_places());
        VLOG(3) << "Float2Fix, op_type=" << op_type
                << ", weight_name=" << weight_name;
      } else if (op_type == "match_matrix_tensor") {
        std::string weight_name = op_info->Input("W").front();
        auto* scope = node->stmt()->op()->scope();
        auto* weight_t = scope->FindMutableTensor(weight_name);
        auto weight_dims = weight_t->dims();
        auto weight_len = weight_t->numel();
        float* weight_on_host = weight_t->mutable_data<float>();
        float max_f =
            paddle::lite::xpu::math::FindMaxAbs(weight_on_host, weight_len);
        std::unique_ptr<int16_t[]> weight_int16(new int16_t[weight_len]);
        std::unique_ptr<int16_t[]> weight_trans_int16(new int16_t[weight_len]);
        paddle::lite::xpu::math::ConvertFP32ToInt16(
            weight_on_host, weight_int16.get(), max_f, weight_len);
        paddle::lite::xpu::math::Transpose(weight_int16.get(),
                                           weight_trans_int16.get(),
                                           weight_dims[0],
                                           weight_dims[1] * weight_dims[2]);
        memcpy(weight_on_host,
               weight_trans_int16.get(),
               weight_len * sizeof(int16_t));

        auto update_op_info = *op_info;
        update_op_info.SetAttr<bool>("__xpu__float_to_fix", true);
        update_op_info.SetAttr<float>("__xpu__w_max", max_f);
        node->stmt()->ResetOp(update_op_info, graph->valid_places());
        VLOG(3) << "Float2Fix && Transposed, op_type=" << op_type
                << ", weight_name=" << weight_name;
      } else if (op_type == "search_grnn") {
        auto* scope = node->stmt()->op()->scope();

        std::string wi_name = op_info->Input("Wi").front();
        auto* wi_t = scope->FindMutableTensor(wi_name);
        auto wi_dims = wi_t->dims();
        auto wi_len = wi_t->numel();
        auto wi_stride_len = wi_len / 3;
        float* wi_on_host = wi_t->mutable_data<float>();
        std::unique_ptr<int16_t[]> wi_int16(new int16_t[wi_len]);
        std::vector<float> wi_max(3);
        for (int i = 0; i < 3; ++i) {
          float max_f = paddle::lite::xpu::math::FindMaxAbs(
              wi_on_host + i * wi_stride_len, wi_stride_len);
          paddle::lite::xpu::math::ConvertFP32ToInt16(
              wi_on_host + i * wi_stride_len,
              wi_int16.get() + i * wi_stride_len,
              max_f,
              wi_stride_len);
          wi_max[i] = max_f;
        }
        memcpy(wi_on_host, wi_int16.get(), wi_len * sizeof(int16_t));

        std::string wh_name = op_info->Input("Wh").front();
        auto* wh_t = scope->FindMutableTensor(wh_name);
        auto wh_dims = wh_t->dims();
        auto wh_len = wh_t->numel();
        auto wh_stride_len = wh_len / 3;
        float* wh_on_host = wh_t->mutable_data<float>();
        std::unique_ptr<int16_t[]> wh_int16(new int16_t[wh_len]);
        std::vector<float> wh_max(3);
        for (int i = 0; i < 3; ++i) {
          float max_f = paddle::lite::xpu::math::FindMaxAbs(
              wh_on_host + i * wh_stride_len, wh_stride_len);
          paddle::lite::xpu::math::ConvertFP32ToInt16(
              wh_on_host + i * wh_stride_len,
              wh_int16.get() + i * wh_stride_len,
              max_f,
              wh_stride_len);
          wh_max[i] = max_f;
        }
        memcpy(wh_on_host, wh_int16.get(), wh_len * sizeof(int16_t));

        auto update_op_info = *op_info;
        update_op_info.SetAttr<bool>("__xpu__float_to_fix", true);
        update_op_info.SetAttr<std::vector<float>>("__xpu__wi_max", wi_max);
        update_op_info.SetAttr<std::vector<float>>("__xpu__wh_max", wh_max);
        node->stmt()->ResetOp(update_op_info, graph->valid_places());
        VLOG(3) << "Float2Fix, op_type=" << op_type << ", wi_name=" << wi_name
                << ", wh_name=" << wh_name;
      }
    }
  }
};

class XPUMmdnnSearchAttentionFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input = VarNode("input")->AsInput();

    auto* search_group_padding =
        OpNode("search_group_padding", "search_group_padding");
    auto* out_emb_padding =
        VarNode("out_emb_padding")
            ->assert_is_op_output("search_group_padding", "Out_emb_padding")
            ->AsIntermediate();
    auto* out_new = VarNode("out_new")
                        ->assert_is_op_output("search_group_padding", "Out_new")
                        ->AsIntermediate();
    auto* out_padding =
        VarNode("out_padding")
            ->assert_is_op_output("search_group_padding", "Out_padding")
            ->AsIntermediate();

    auto* search_seq_fc_w = VarNode("search_seq_fc_w")
                                ->assert_is_op_input("search_seq_fc", "W")
                                ->AsInput();
    auto* search_seq_fc_b = VarNode("search_seq_fc_b")
                                ->assert_is_op_input("search_seq_fc", "b")
                                ->AsInput();
    auto* search_seq_fc =
        OpNode("search_seq_fc", "search_seq_fc")->AsIntermediate();
    auto* search_seq_fc_out = VarNode("search_seq_fc_out")
                                  ->assert_is_op_output("search_seq_fc", "Out")
                                  ->AsIntermediate();

    auto* search_aligned_mat_mul =
        OpNode("search_aligned_mat_mul", "search_aligned_mat_mul")
            ->AsIntermediate();
    auto* search_aligned_mat_mul_out =
        VarNode("search_aligned_mat_mul_out")
            ->assert_is_op_output("search_aligned_mat_mul", "Out")
            ->AsIntermediate();
    auto* search_aligned_mat_mul_a =
        VarNode("search_aligned_mat_mul_a")
            ->assert_is_op_output("search_aligned_mat_mul", "_a_addr")
            ->AsIntermediate();
    auto* search_aligned_mat_mul_b =
        VarNode("search_aligned_mat_mul_b")
            ->assert_is_op_output("search_aligned_mat_mul", "_b_addr")
            ->AsIntermediate();
    auto* search_aligned_mat_mul_c =
        VarNode("search_aligned_mat_mul_c")
            ->assert_is_op_output("search_aligned_mat_mul", "_c_addr")
            ->AsIntermediate();

    auto* search_attention_padding_mask =
        OpNode("search_attention_padding_mask", "search_attention_padding_mask")
            ->AsIntermediate();
    auto* search_attention_padding_mask_out =
        VarNode("search_attention_padding_mask_out")
            ->assert_is_op_output("search_attention_padding_mask", "Out")
            ->AsIntermediate();
    auto* search_attention_padding_mask_pad_begin =
        VarNode("search_attention_padding_mask_pad_begin")
            ->assert_is_op_output("search_attention_padding_mask", "pad_begin")
            ->AsIntermediate();

    auto* search_seq_softmax =
        OpNode("search_seq_softmax", "search_seq_softmax")->AsIntermediate();
    auto* search_seq_softmax_out =
        VarNode("search_seq_softmax_out")
            ->assert_is_op_output("search_seq_softmax", "Out")
            ->AsIntermediate();
    auto* search_seq_softmax_out_log =
        VarNode("search_seq_softmax_out_log")
            ->assert_is_op_output("search_seq_softmax", "Out_log")
            ->AsIntermediate();

    auto* search_aligned_mat_mul_2 =
        OpNode("search_aligned_mat_mul_2", "search_aligned_mat_mul")
            ->AsIntermediate();
    auto* search_aligned_mat_mul_2_out =
        VarNode("search_aligned_mat_mul_2_out")
            ->assert_is_op_output("search_aligned_mat_mul", "Out")
            ->AsIntermediate();
    auto* search_aligned_mat_mul_2_a =
        VarNode("search_aligned_mat_mul_2_a")
            ->assert_is_op_output("search_aligned_mat_mul", "_a_addr")
            ->AsIntermediate();
    auto* search_aligned_mat_mul_2_b =
        VarNode("search_aligned_mat_mul_2_b")
            ->assert_is_op_output("search_aligned_mat_mul", "_b_addr")
            ->AsIntermediate();
    auto* search_aligned_mat_mul_2_c =
        VarNode("search_aligned_mat_mul_2_c")
            ->assert_is_op_output("search_aligned_mat_mul", "_c_addr")
            ->AsIntermediate();

    auto* search_seq_depadding =
        OpNode("search_seq_depadding")->AsIntermediate();
    auto* search_seq_depadding_out =
        VarNode("search_seq_depadding_out")->AsOutput();

    *input >> *search_group_padding >> *out_emb_padding;
    *search_group_padding >> *out_new;
    *search_group_padding >> *out_padding;

    *search_seq_fc_w >> *search_seq_fc;
    *search_seq_fc_b >> *search_seq_fc;
    *out_emb_padding >> *search_seq_fc;
    *search_seq_fc >> *search_seq_fc_out;

    *search_seq_fc_out >> *search_aligned_mat_mul;
    *out_emb_padding >> *search_aligned_mat_mul;
    *search_aligned_mat_mul >> *search_aligned_mat_mul_out;
    *search_aligned_mat_mul >> *search_aligned_mat_mul_a;
    *search_aligned_mat_mul >> *search_aligned_mat_mul_b;
    *search_aligned_mat_mul >> *search_aligned_mat_mul_c;

    *search_aligned_mat_mul_out >> *search_attention_padding_mask;
    *out_padding >> *search_attention_padding_mask;
    *search_attention_padding_mask >> *search_attention_padding_mask_out;
    *search_attention_padding_mask >> *search_attention_padding_mask_pad_begin;

    *search_attention_padding_mask_out >> *search_seq_softmax;
    *search_seq_softmax >> *search_seq_softmax_out;
    *search_seq_softmax >> *search_seq_softmax_out_log;

    *search_seq_softmax_out >> *search_aligned_mat_mul_2;
    *out_emb_padding >> *search_aligned_mat_mul_2;
    *search_aligned_mat_mul_2 >> *search_aligned_mat_mul_2_out;
    *search_aligned_mat_mul_2 >> *search_aligned_mat_mul_2_a;
    *search_aligned_mat_mul_2 >> *search_aligned_mat_mul_2_b;
    *search_aligned_mat_mul_2 >> *search_aligned_mat_mul_2_c;

    *search_aligned_mat_mul_2_out >> *search_seq_depadding;
    *out_new >> *search_seq_depadding;
    *search_seq_depadding >> *search_seq_depadding_out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__mmdnn_search_attention");
    op_desc.SetInput("X", {matched.at("input")->arg()->name});
    op_desc.SetInput("W", {matched.at("search_seq_fc_w")->arg()->name});
    op_desc.SetInput("b", {matched.at("search_seq_fc_b")->arg()->name});
    op_desc.SetOutput("Out",
                      {matched.at("search_seq_depadding_out")->arg()->name});

    auto* padding_op_info =
        matched.at("search_group_padding")->stmt()->op_info();
    op_desc.SetAttr<int>("pad_id", padding_op_info->GetAttr<int>("pad_id"));
    auto* matmul_0_op_info =
        matched.at("search_aligned_mat_mul")->stmt()->op_info();
    op_desc.SetAttr<float>("alpha0", matmul_0_op_info->GetAttr<float>("alpha"));
    auto* matmul_1_op_info =
        matched.at("search_aligned_mat_mul_2")->stmt()->op_info();
    op_desc.SetAttr<float>("alpha1", matmul_1_op_info->GetAttr<float>("alpha"));
    auto* mask_op_info =
        matched.at("search_attention_padding_mask")->stmt()->op_info();
    op_desc.SetAttr<float>("mask", mask_op_info->GetAttr<float>("mask"));

    auto* new_stmt = matched.at("search_group_padding")->stmt();
    auto* scope = new_stmt->op()->scope();
    auto w_name = matched.at("search_seq_fc_w")->arg()->name;
    auto* w_t = scope->FindMutableTensor(w_name);
    auto w_dims = w_t->dims();
    int w_len = w_t->numel();
    float* w_on_host = w_t->mutable_data<float>();

    float max_f = paddle::lite::xpu::math::FindMaxAbs(w_on_host, w_len);
    std::unique_ptr<int16_t[]> w_int16(new int16_t[w_len]);
    paddle::lite::xpu::math::ConvertFP32ToInt16(
        w_on_host, w_int16.get(), max_f, w_len);
    memcpy(w_on_host, w_int16.get(), w_len * sizeof(int16_t));
    op_desc.SetAttr<float>("W_max", max_f);

    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, scope);
    new_op->SetValidPlaces(new_stmt->op()->valid_places());
    auto kernels = new_op->CreateKernels(new_op->valid_places());
    new_stmt->SetOp(new_op);
    new_stmt->SetKernels(std::move(kernels));

    DirectedLink(matched.at("search_seq_fc_w"),
                 matched.at("search_group_padding"));
    DirectedLink(matched.at("search_seq_fc_b"),
                 matched.at("search_group_padding"));
    IR_OP_VAR_LINK(matched.at("search_group_padding"),
                   matched.at("search_seq_depadding_out"));
  }
};

class XPUMmdnnSearchAttentionFuser2 : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input =
        VarNode("input")->assert_is_op_input("sequence_pad", "X")->AsInput();
    auto* assign_value = VarNode("assign_value")
                             ->assert_is_op_input("sequence_pad", "PadValue")
                             ->AsInput();

    auto* sequence_pad = OpNode("sequence_pad", "sequence_pad");
    auto* sequence_pad_out = VarNode("sequence_pad_out")
                                 ->assert_is_op_output("sequence_pad", "Out")
                                 ->AsIntermediate();
    auto* sequence_pad_length =
        VarNode("sequence_pad_length")
            ->assert_is_op_output("sequence_pad", "Length")
            ->AsIntermediate();

    auto* mul_weight =
        VarNode("mul_weight")->assert_is_op_input("mul", "Y")->AsInput();
    auto* mul = OpNode("mul", "mul")->AsIntermediate();
    auto* mul_out =
        VarNode("mul_out")->assert_is_op_output("mul", "Out")->AsIntermediate();

    auto* elementwise_add_0_bias =
        VarNode("elementwise_add_0_bias")
            ->assert_is_op_input("elementwise_add", "Y")
            ->AsInput();
    auto* elementwise_add_0 =
        OpNode("elementwise_add_0", "elementwise_add")->AsIntermediate();
    auto* elementwise_add_0_out =
        VarNode("elementwise_add_0_out")
            ->assert_is_op_output("elementwise_add", "Out")
            ->AsIntermediate();

    auto* matmul_0 = OpNode("matmul_0", "matmul")->AsIntermediate();
    auto* matmul_0_out = VarNode("matmul_0_out")
                             ->assert_is_op_output("matmul", "Out")
                             ->AsIntermediate();

    auto* transpose2_0 = OpNode("transpose2_0", "transpose2")->AsIntermediate();
    auto* transpose2_0_out = VarNode("transpose2_0_out")
                                 ->assert_is_op_output("transpose2", "Out")
                                 ->AsIntermediate();
    auto* transpose2_0_xshape =
        VarNode("transpose2_0_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();

    auto* sequence_mask =
        OpNode("sequence_mask", "sequence_mask")->AsIntermediate();
    auto* sequence_mask_out = VarNode("sequence_mask_out")
                                  ->assert_is_op_output("sequence_mask", "Y")
                                  ->AsIntermediate();

    auto* scale = OpNode("scale", "scale")->AsIntermediate();
    auto* scale_out = VarNode("scale_out")
                          ->assert_is_op_output("scale", "Out")
                          ->AsIntermediate();

    auto* elementwise_add_1 =
        OpNode("elementwise_add_1", "elementwise_add")->AsIntermediate();
    auto* elementwise_add_1_out =
        VarNode("elementwise_add_1_out")
            ->assert_is_op_output("elementwise_add", "Out")
            ->AsIntermediate();

    auto* transpose2_1 = OpNode("transpose2_1", "transpose2")->AsIntermediate();
    auto* transpose2_1_out = VarNode("transpose2_1_out")
                                 ->assert_is_op_output("transpose2", "Out")
                                 ->AsIntermediate();
    auto* transpose2_1_xshape =
        VarNode("transpose2_1_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();

    auto* softmax = OpNode("softmax", "softmax")->AsIntermediate();
    auto* softmax_out = VarNode("softmax_out")
                            ->assert_is_op_output("softmax", "Out")
                            ->AsIntermediate();

    auto* matmul_1 = OpNode("matmul_1", "matmul")->AsIntermediate();
    auto* matmul_1_out = VarNode("matmul_1_out")
                             ->assert_is_op_output("matmul", "Out")
                             ->AsIntermediate();

    auto* sequence_unpad =
        OpNode("sequence_unpad", "sequence_unpad")->AsIntermediate();
    auto* output = VarNode("output")
                       ->assert_is_op_output("sequence_unpad", "Out")
                       ->AsOutput();

    *input >> *sequence_pad;
    *assign_value >> *sequence_pad;
    *sequence_pad >> *sequence_pad_out;
    *sequence_pad >> *sequence_pad_length;

    *sequence_pad_out >> *mul;
    *mul_weight >> *mul;
    *mul >> *mul_out;

    *mul_out >> *elementwise_add_0;
    *elementwise_add_0_bias >> *elementwise_add_0;
    *elementwise_add_0 >> *elementwise_add_0_out;

    *sequence_pad_out >> *matmul_0;
    *elementwise_add_0_out >> *matmul_0;
    *matmul_0 >> *matmul_0_out;

    *matmul_0_out >> *transpose2_0;
    *transpose2_0 >> *transpose2_0_out;
    *transpose2_0 >> *transpose2_0_xshape;

    *sequence_pad_length >> *sequence_mask >> *sequence_mask_out;
    *sequence_mask_out >> *scale >> *scale_out;

    *transpose2_0_out >> *elementwise_add_1;
    *scale_out >> *elementwise_add_1;
    *elementwise_add_1 >> *elementwise_add_1_out;

    *elementwise_add_1_out >> *transpose2_1;
    *transpose2_1 >> *transpose2_1_out;
    *transpose2_1 >> *transpose2_1_xshape;

    *transpose2_1_out >> *softmax >> *softmax_out;

    *sequence_pad_out >> *matmul_1;
    *softmax_out >> *matmul_1;
    *matmul_1 >> *matmul_1_out;

    *sequence_pad_length >> *sequence_unpad;
    *matmul_1_out >> *sequence_unpad;
    *sequence_unpad >> *output;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__mmdnn_search_attention2");
    op_desc.SetInput("X", {matched.at("input")->arg()->name});
    op_desc.SetInput("W", {matched.at("mul_weight")->arg()->name});
    op_desc.SetInput("b", {matched.at("elementwise_add_0_bias")->arg()->name});
    op_desc.SetOutput("Out", {matched.at("output")->arg()->name});

    auto* new_stmt = matched.at("sequence_pad")->stmt();
    auto* scope = new_stmt->op()->scope();

    auto assign_value_name = matched.at("assign_value")->arg()->name;
    auto* assign_value_tensor = scope->FindMutableTensor(assign_value_name);
    op_desc.SetAttr<int>(
        "pad_id",
        static_cast<int>(assign_value_tensor->mutable_data<float>()[0]));
    auto* matmul_0_op_info = matched.at("matmul_0")->stmt()->op_info();
    op_desc.SetAttr<float>("alpha0", matmul_0_op_info->GetAttr<float>("alpha"));
    auto* matmul_1_op_info = matched.at("matmul_1")->stmt()->op_info();
    op_desc.SetAttr<float>("alpha1", matmul_1_op_info->GetAttr<float>("alpha"));
    auto* mask_op_info = matched.at("scale")->stmt()->op_info();
    op_desc.SetAttr<float>("mask",
                           mask_op_info->GetAttr<float>("scale") *
                               mask_op_info->GetAttr<float>("bias"));

    auto w_name = matched.at("mul_weight")->arg()->name;
    auto* w_t = scope->FindMutableTensor(w_name);
    auto w_dims = w_t->dims();
    int w_len = w_t->numel();
    float* w_on_host = w_t->mutable_data<float>();

    float max_f = paddle::lite::xpu::math::FindMaxAbs(w_on_host, w_len);
    std::unique_ptr<int16_t[]> w_int16(new int16_t[w_len]);
    std::unique_ptr<int16_t[]> w_trans_int16(new int16_t[w_len]);
    paddle::lite::xpu::math::ConvertFP32ToInt16(
        w_on_host, w_int16.get(), max_f, w_len);
    paddle::lite::xpu::math::Transpose(
        w_int16.get(), w_trans_int16.get(), w_dims[0], w_dims[1]);
    memcpy(w_on_host, w_trans_int16.get(), w_len * sizeof(int16_t));
    op_desc.SetAttr<float>("W_max", max_f);

    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, scope);
    new_op->SetValidPlaces(new_stmt->op()->valid_places());
    auto kernels = new_op->CreateKernels(new_op->valid_places());
    new_stmt->SetOp(new_op);
    new_stmt->SetKernels(std::move(kernels));

    RemoveDirectedLink(matched.at("assign_value"), matched.at("sequence_pad"));
    DirectedLink(matched.at("mul_weight"), matched.at("sequence_pad"));
    DirectedLink(matched.at("elementwise_add_0_bias"),
                 matched.at("sequence_pad"));
    IR_OP_VAR_LINK(matched.at("sequence_pad"), matched.at("output"));
  }
};

// 4 inputs
// ========
//
// input_x
// input_y
// topk_row
// topk_col
//
// input_x ------- match_matrix_tensor ------- input_y
//                           |
//                          relu
//                 ________/    \________
//                 |                    |
//            var_conv_2d               |
//                 |                    |
//                relu                  |
//                 |_______      _______|
//                         \    /
//                   sequence_concat
//                           |
// topk_row ---- sequence_topk_avg_pooling ----- topk_col
//
class XPUMmdnnMatchConvTopkFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input_x = VarNode("input_x")
                        ->assert_is_op_input("match_matrix_tensor", "X")
                        ->AsInput();
    auto* input_y = VarNode("input_y")
                        ->assert_is_op_input("match_matrix_tensor", "Y")
                        ->AsInput();
    auto* input_w = VarNode("input_w")
                        ->assert_is_op_input("match_matrix_tensor", "W")
                        ->AsInput();

    auto* match_matrix_tensor =
        OpNode("match_matrix_tensor", "match_matrix_tensor");
    auto* match_out = VarNode("match_out")
                          ->assert_is_op_output("match_matrix_tensor", "Out")
                          ->AsIntermediate();
    auto* match_tmp = VarNode("match_tmp")
                          ->assert_is_op_output("match_matrix_tensor", "Tmp")
                          ->AsIntermediate();
    auto* relu0 = OpNode("relu0", "relu")->AsIntermediate();
    auto* relu0_out = VarNode("relu0_out")
                          ->assert_is_op_output("relu", "Out")
                          ->AsIntermediate();
    auto* conv_w =
        VarNode("conv_w")->assert_is_op_input("var_conv_2d", "W")->AsInput();
    auto* conv = OpNode("conv", "var_conv_2d")->AsIntermediate();
    auto* conv_out = VarNode("conv_out")
                         ->assert_is_op_output("var_conv_2d", "Out")
                         ->AsIntermediate();
    auto* conv_col = VarNode("conv_col")
                         ->assert_is_op_output("var_conv_2d", "Col")
                         ->AsIntermediate();
    auto* relu1 = OpNode("relu1", "relu")->AsIntermediate();
    auto* relu1_out = VarNode("relu1_out")
                          ->assert_is_op_output("relu", "Out")
                          ->AsIntermediate();
    auto* seq_concat =
        OpNode("seq_concat", "sequence_concat")->AsIntermediate();
    auto* seq_concat_out =
        VarNode("seq_concat_out")
            ->assert_is_op_output("sequence_concat", "Out")
            ->assert_is_op_input("sequence_topk_avg_pooling", "X")
            ->AsIntermediate();
    auto* topk_col =
        VarNode("topk_col")
            ->assert_is_op_input("sequence_topk_avg_pooling", "COLUMN")
            ->AsInput();
    auto* topk_row =
        VarNode("topk_row")
            ->assert_is_op_input("sequence_topk_avg_pooling", "ROW")
            ->AsInput();
    auto* topk = OpNode("topk", "sequence_topk_avg_pooling")->AsIntermediate();
    auto* topk_out =
        VarNode("topk_out")
            ->assert_is_op_output("sequence_topk_avg_pooling", "Out")
            ->AsOutput();
    auto* topk_pos =
        VarNode("topk_pos")
            ->assert_is_op_output("sequence_topk_avg_pooling", "pos")
            ->AsIntermediate();

    *input_x >> *match_matrix_tensor;
    *input_y >> *match_matrix_tensor;
    *input_w >> *match_matrix_tensor;
    *match_matrix_tensor >> *match_out >> *relu0 >> *relu0_out;
    *match_matrix_tensor >> *match_tmp;

    *relu0_out >> *conv >> *conv_out >> *relu1 >> *relu1_out;
    *conv_w >> *conv;
    *conv >> *conv_col;

    *relu0_out >> *seq_concat;
    *relu1_out >> *seq_concat;
    *seq_concat >> *seq_concat_out >> *topk >> *topk_out;
    *topk_col >> *topk;
    *topk_row >> *topk;
    *topk >> *topk_pos;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__mmdnn_match_conv_topk");
    op_desc.SetInput("input_x", {matched.at("input_x")->arg()->name});
    op_desc.SetInput("input_y", {matched.at("input_y")->arg()->name});
    op_desc.SetInput("input_w", {matched.at("input_w")->arg()->name});
    op_desc.SetInput("conv_w", {matched.at("conv_w")->arg()->name});
    op_desc.SetOutput("topk_out", {matched.at("topk_out")->arg()->name});

    auto* match_op_info = matched.at("match_matrix_tensor")->stmt()->op_info();
    op_desc.SetAttr<float>("input_w_max",
                           match_op_info->GetAttr<float>("__xpu__w_max"));
    op_desc.SetAttr<int>("dim_t", match_op_info->GetAttr<int>("dim_t"));
    auto* conv_op_info = matched.at("conv")->stmt()->op_info();
    op_desc.SetAttr<float>("conv_w_max",
                           conv_op_info->GetAttr<float>("__xpu__w_max"));
    op_desc.SetAttr<int>("output_channel",
                         conv_op_info->GetAttr<int>("OutputChannel"));
    auto* topk_op_info = matched.at("topk")->stmt()->op_info();
    op_desc.SetAttr<std::vector<int>>(
        "topks", topk_op_info->GetAttr<std::vector<int>>("topks"));
    op_desc.SetAttr<int>("channel_num",
                         topk_op_info->GetAttr<int>("channel_num"));

    auto* new_stmt = matched.at("match_matrix_tensor")->stmt();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, new_stmt->op()->scope());
    new_op->SetValidPlaces(new_stmt->op()->valid_places());
    auto kernels = new_op->CreateKernels(new_op->valid_places());
    new_stmt->SetOp(new_op);
    new_stmt->SetKernels(std::move(kernels));

    // XXX(miaotianxiang): redundant links around |topk| are automatically
    // removed as |topk| is marked intermediate.
    // RemoveDirectedLink(matched.at("topk_col"), matched.at("topk"));
    // RemoveDirectedLink(matched.at("topk_row"), matched.at("topk"));
    std::vector<std::string> arg_names{"conv_w"};
    for (auto name : arg_names) {
      DirectedLink(matched.at(name), matched.at("match_matrix_tensor"));
    }
    std::vector<std::string> out_names{"topk_out"};
    for (auto name : out_names) {
      IR_OP_VAR_LINK(matched.at("match_matrix_tensor"), matched.at(name));
    }
  }
};

// 2 inputs
// ========
//
// input_x
// input_y
//
// input_x ------- match_matrix_tensor ------- input_y
//    |                      |                    |
//    |                     relu                  |
//    |            ________/    \________         |
//    |            |                    |         |
//    |       var_conv_2d               |         |
//    |            |                    |         |
//    |           relu                  |         |
//    |            |_______      _______|         |
//    |                    \    /                 |
//    |              sequence_concat              |
//    |                      |                    |
//    |--------- sequence_topk_avg_pooling -------|
//
class XPUMmdnnMatchConvTopkFuser2 : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input_x = VarNode("input_x")
                        ->assert_is_op_input("match_matrix_tensor", "X")
                        ->assert_is_op_input("sequence_topk_avg_pooling", "ROW")
                        ->AsInput();
    auto* input_y =
        VarNode("input_y")
            ->assert_is_op_input("match_matrix_tensor", "Y")
            ->assert_is_op_input("sequence_topk_avg_pooling", "COLUMN")
            ->AsInput();
    auto* input_w = VarNode("input_w")
                        ->assert_is_op_input("match_matrix_tensor", "W")
                        ->AsInput();

    auto* match_matrix_tensor =
        OpNode("match_matrix_tensor", "match_matrix_tensor");
    auto* match_out = VarNode("match_out")
                          ->assert_is_op_output("match_matrix_tensor", "Out")
                          ->AsIntermediate();
    auto* match_tmp = VarNode("match_tmp")
                          ->assert_is_op_output("match_matrix_tensor", "Tmp")
                          ->AsIntermediate();
    auto* relu0 = OpNode("relu0", "relu")->AsIntermediate();
    auto* relu0_out = VarNode("relu0_out")
                          ->assert_is_op_output("relu", "Out")
                          ->AsIntermediate();
    auto* conv_w =
        VarNode("conv_w")->assert_is_op_input("var_conv_2d", "W")->AsInput();
    auto* conv = OpNode("conv", "var_conv_2d")->AsIntermediate();
    auto* conv_out = VarNode("conv_out")
                         ->assert_is_op_output("var_conv_2d", "Out")
                         ->AsIntermediate();
    auto* conv_col = VarNode("conv_col")
                         ->assert_is_op_output("var_conv_2d", "Col")
                         ->AsIntermediate();
    auto* relu1 = OpNode("relu1", "relu")->AsIntermediate();
    auto* relu1_out = VarNode("relu1_out")
                          ->assert_is_op_output("relu", "Out")
                          ->AsIntermediate();
    auto* seq_concat =
        OpNode("seq_concat", "sequence_concat")->AsIntermediate();
    auto* seq_concat_out =
        VarNode("seq_concat_out")
            ->assert_is_op_output("sequence_concat", "Out")
            ->assert_is_op_input("sequence_topk_avg_pooling", "X")
            ->AsIntermediate();
    auto* topk = OpNode("topk", "sequence_topk_avg_pooling")->AsIntermediate();
    auto* topk_out =
        VarNode("topk_out")
            ->assert_is_op_output("sequence_topk_avg_pooling", "Out")
            ->AsOutput();
    auto* topk_pos =
        VarNode("topk_pos")
            ->assert_is_op_output("sequence_topk_avg_pooling", "pos")
            ->AsIntermediate();

    *input_x >> *match_matrix_tensor;
    *input_y >> *match_matrix_tensor;
    *input_w >> *match_matrix_tensor;
    *match_matrix_tensor >> *match_out >> *relu0 >> *relu0_out;
    *match_matrix_tensor >> *match_tmp;

    *relu0_out >> *conv >> *conv_out >> *relu1 >> *relu1_out;
    *conv_w >> *conv;
    *conv >> *conv_col;

    *relu0_out >> *seq_concat;
    *relu1_out >> *seq_concat;
    *seq_concat >> *seq_concat_out >> *topk >> *topk_out;
    *input_x >> *topk;
    *input_y >> *topk;
    *topk >> *topk_pos;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__mmdnn_match_conv_topk");
    op_desc.SetInput("input_x", {matched.at("input_x")->arg()->name});
    op_desc.SetInput("input_y", {matched.at("input_y")->arg()->name});
    op_desc.SetInput("input_w", {matched.at("input_w")->arg()->name});
    op_desc.SetInput("conv_w", {matched.at("conv_w")->arg()->name});
    op_desc.SetOutput("topk_out", {matched.at("topk_out")->arg()->name});

    auto* match_op_info = matched.at("match_matrix_tensor")->stmt()->op_info();
    op_desc.SetAttr<float>("input_w_max",
                           match_op_info->GetAttr<float>("__xpu__w_max"));
    op_desc.SetAttr<int>("dim_t", match_op_info->GetAttr<int>("dim_t"));
    auto* conv_op_info = matched.at("conv")->stmt()->op_info();
    op_desc.SetAttr<float>("conv_w_max",
                           conv_op_info->GetAttr<float>("__xpu__w_max"));
    op_desc.SetAttr<int>("output_channel",
                         conv_op_info->GetAttr<int>("OutputChannel"));
    auto* topk_op_info = matched.at("topk")->stmt()->op_info();
    op_desc.SetAttr<std::vector<int>>(
        "topks", topk_op_info->GetAttr<std::vector<int>>("topks"));
    op_desc.SetAttr<int>("channel_num",
                         topk_op_info->GetAttr<int>("channel_num"));

    auto* new_stmt = matched.at("match_matrix_tensor")->stmt();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, new_stmt->op()->scope());
    new_op->SetValidPlaces(new_stmt->op()->valid_places());
    auto kernels = new_op->CreateKernels(new_op->valid_places());
    new_stmt->SetOp(new_op);
    new_stmt->SetKernels(std::move(kernels));

    // XXX(miaotianxiang): redundant links around |topk| are automatically
    // removed as |topk| is marked intermediate.
    // RemoveDirectedLink(matched.at("topk_col"), matched.at("topk"));
    // RemoveDirectedLink(matched.at("topk_row"), matched.at("topk"));
    std::vector<std::string> arg_names{"conv_w"};
    for (auto name : arg_names) {
      DirectedLink(matched.at(name), matched.at("match_matrix_tensor"));
    }
    std::vector<std::string> out_names{"topk_out"};
    for (auto name : out_names) {
      IR_OP_VAR_LINK(matched.at("match_matrix_tensor"), matched.at(name));
    }
  }
};

class XPUMmdnnBidSeqRevEmbEltwiseFuser : public FuseBase {
 public:
  explicit XPUMmdnnBidSeqRevEmbEltwiseFuser(bool customize_op)
      : customize_op_(customize_op) {}

  void BuildPattern() override {
    std::string eltwise_add_op_name =
        customize_op_ ? "search_seq_arithmetic" : "elementwise_add";

    auto* input0 = VarNode("input0")->AsInput();
    auto* input1 = VarNode("input1")->AsInput();
    auto* emb_tbl = VarNode("emb_tbl")->AsInput();

    // fwd emb
    auto* emb0 = OpNode("emb0", "lookup_table");
    auto* emb0_out =
        VarNode("emb0_out")->assert_is_op_output("lookup_table", "Out");
    auto* emb1 = OpNode("emb1", "lookup_table");
    auto* emb1_out =
        VarNode("emb1_out")->assert_is_op_output("lookup_table", "Out");

    auto* eltwise01 = OpNode("eltwise01", eltwise_add_op_name);
    auto* eltwise01_out = VarNode("eltwise01_out")
                              ->assert_is_op_output(eltwise_add_op_name, "Out")
                              ->AsOutput();

    // rev emb
    auto* seq_rev2 = OpNode("seq_rev2", "sequence_reverse")->AsIntermediate();
    auto* seq_rev2_out = VarNode("seq_rev2_out")
                             ->assert_is_op_output("sequence_reverse", "Y")
                             ->AsIntermediate();
    auto* seq_rev3 = OpNode("seq_rev3", "sequence_reverse")->AsIntermediate();
    auto* seq_rev3_out = VarNode("seq_rev3_out")
                             ->assert_is_op_output("sequence_reverse", "Y")
                             ->AsIntermediate();
    auto* emb2 = OpNode("emb2", "lookup_table")->AsIntermediate();
    auto* emb2_out = VarNode("emb2_out")
                         ->assert_is_op_output("lookup_table", "Out")
                         ->AsIntermediate();
    auto* emb3 = OpNode("emb3", "lookup_table")->AsIntermediate();
    auto* emb3_out = VarNode("emb3_out")
                         ->assert_is_op_output("lookup_table", "Out")
                         ->AsIntermediate();

    auto* eltwise23 =
        OpNode("eltwise23", eltwise_add_op_name)->AsIntermediate();
    auto* eltwise23_out = VarNode("eltwise23_out")
                              ->assert_is_op_output(eltwise_add_op_name, "Out")
                              ->AsOutput();

    *input0 >> *emb0 >> *emb0_out >> *eltwise01 >> *eltwise01_out;
    *emb_tbl >> *emb0;
    *input1 >> *emb1 >> *emb1_out >> *eltwise01;
    *emb_tbl >> *emb1;

    *input0 >> *seq_rev2 >> *seq_rev2_out >> *emb2 >> *emb2_out >> *eltwise23 >>
        *eltwise23_out;
    *emb_tbl >> *emb2;
    *input1 >> *seq_rev3 >> *seq_rev3_out >> *emb3 >> *emb3_out >> *eltwise23;
    *emb_tbl >> *emb3;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("sequence_reverse");
    op_desc.SetInput("X", {matched.at("eltwise01_out")->arg()->name});
    op_desc.SetOutput("Y", {matched.at("eltwise23_out")->arg()->name});

    auto emb0_op = matched.at("emb0")->stmt()->op();
    auto new_seq_rev_op = LiteOpRegistry::Global().Create("sequence_reverse");
    new_seq_rev_op->Attach(op_desc, emb0_op->scope());
    auto* new_seq_rev_node =
        graph->GraphCreateInstructNode(new_seq_rev_op, emb0_op->valid_places());

    DirectedLink(matched.at("eltwise01_out"), new_seq_rev_node);
    DirectedLink(new_seq_rev_node, matched.at("eltwise23_out"));
  }

 private:
  bool customize_op_;
};

class XPUMmdnnBidEmbAttFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input0 = VarNode("input0")->AsInput();
    auto* input1 = VarNode("input1")->AsInput();
    auto* emb_tbl = VarNode("emb_tbl")->AsInput();

    auto* emb0 = OpNode("emb0", "lookup_table");
    auto* emb0_out = VarNode("emb0_out")
                         ->assert_is_op_output("lookup_table", "Out")
                         ->AsIntermediate();
    auto* emb1 = OpNode("emb1", "lookup_table")->AsIntermediate();
    auto* emb1_out = VarNode("emb1_out")
                         ->assert_is_op_output("lookup_table", "Out")
                         ->AsIntermediate();
    auto* eltwise01 =
        OpNode("eltwise01", "search_seq_arithmetic")->AsIntermediate();
    auto* eltwise01_out =
        VarNode("eltwise01_out")
            ->assert_is_op_output("search_seq_arithmetic", "Out")
            ->AsOutput();

    auto* att_2in1_w =
        VarNode("att_2in1_w")
            ->assert_is_op_input("__xpu__mmdnn_search_attention", "W")
            ->AsInput();
    auto* att_2in1_b =
        VarNode("att_2in1_b")
            ->assert_is_op_input("__xpu__mmdnn_search_attention", "b")
            ->AsInput();
    auto* att_2in1 =
        OpNode("att_2in1", "__xpu__mmdnn_search_attention")->AsIntermediate();
    auto* att_2in1_out =
        VarNode("att_2in1_out")
            ->assert_is_op_output("__xpu__mmdnn_search_attention", "Out")
            ->AsIntermediate();
    auto* seq_pool_2in1 =
        OpNode("seq_pool_2in1", "sequence_pool")->AsIntermediate();
    auto* seq_pool_2in1_out = VarNode("seq_pool_2in1_out")
                                  ->assert_is_op_output("sequence_pool", "Out")
                                  ->AsOutput();
    auto* seq_pool_2in1_max_idx =
        VarNode("seq_pool_2in1_max_idx")
            ->assert_is_op_output("sequence_pool", "MaxIndex")
            ->AsIntermediate();

    *input0 >> *emb0 >> *emb0_out >> *eltwise01 >> *eltwise01_out;
    *emb_tbl >> *emb0;
    *input1 >> *emb1 >> *emb1_out >> *eltwise01;
    *emb_tbl >> *emb1;

    *eltwise01_out >> *att_2in1 >> *att_2in1_out >> *seq_pool_2in1 >>
        *seq_pool_2in1_out;
    *seq_pool_2in1 >> *seq_pool_2in1_max_idx;
    *att_2in1_w >> *att_2in1;
    *att_2in1_b >> *att_2in1;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__mmdnn_bid_emb_att");
    op_desc.SetInput("id0", {matched.at("input0")->arg()->name});
    op_desc.SetInput("id1", {matched.at("input1")->arg()->name});
    op_desc.SetInput("emb_tbl", {matched.at("emb_tbl")->arg()->name});
    op_desc.SetInput("att_fc_w", {matched.at("att_2in1_w")->arg()->name});
    op_desc.SetInput("att_fc_b", {matched.at("att_2in1_b")->arg()->name});
    op_desc.SetOutput("att_pool_out",
                      {matched.at("seq_pool_2in1_out")->arg()->name});
    op_desc.SetOutput("emb_fw_out", {matched.at("eltwise01_out")->arg()->name});

    auto* att_fc_op_info = matched.at("att_2in1")->stmt()->op_info();
    op_desc.SetAttr<float>("att_fc_w_max",
                           att_fc_op_info->GetAttr<float>("W_max"));

    auto* new_stmt = matched.at("emb0")->stmt();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, new_stmt->op()->scope());
    new_op->SetValidPlaces(new_stmt->op()->valid_places());
    auto kernels = new_op->CreateKernels(new_op->valid_places());
    new_stmt->SetOp(new_op);
    new_stmt->SetKernels(std::move(kernels));

    std::vector<std::string> arg_names{
        "input1", "att_2in1_w", "att_2in1_b",
    };
    for (auto name : arg_names) {
      DirectedLink(matched.at(name), matched.at("emb0"));
    }
    std::vector<std::string> out_names{
        "seq_pool_2in1_out", "eltwise01_out",
    };
    for (auto name : out_names) {
      IR_OP_VAR_LINK(matched.at("emb0"), matched.at(name));
    }
  }
};

// 5 outputs
// =========
//
// eltwise01_out
// seq_pool_right_out
// seq_pool_left_out
// seq_pool_2in1_out
// concat_3in1_out
//
class XPUMmdnnBidEmbGrnnAttFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input0 = VarNode("input0")->AsInput();
    auto* input1 = VarNode("input1")->AsInput();
    auto* emb_tbl = VarNode("emb_tbl")->AsInput();

    auto* emb0 = OpNode("emb0", "lookup_table");
    auto* emb0_out = VarNode("emb0_out")
                         ->assert_is_op_output("lookup_table", "Out")
                         ->AsIntermediate();
    auto* emb1 = OpNode("emb1", "lookup_table")->AsIntermediate();
    auto* emb1_out = VarNode("emb1_out")
                         ->assert_is_op_output("lookup_table", "Out")
                         ->AsIntermediate();
    auto* eltwise01 =
        OpNode("eltwise01", "search_seq_arithmetic")->AsIntermediate();
    auto* eltwise01_out =
        VarNode("eltwise01_out")
            ->assert_is_op_output("search_seq_arithmetic", "Out")
            ->AsOutput();

    auto* seq_rev_right0 =
        OpNode("seq_rev_right0", "sequence_reverse")->AsIntermediate();
    auto* seq_rev_right0_out =
        VarNode("seq_rev_right0_out")
            ->assert_is_op_output("sequence_reverse", "Y")
            ->AsIntermediate();
    auto* grnn_right_wh = VarNode("grnn_right_wh")
                              ->assert_is_op_input("search_grnn", "Wh")
                              ->AsInput();
    auto* grnn_right_wi = VarNode("grnn_right_wi")
                              ->assert_is_op_input("search_grnn", "Wi")
                              ->AsInput();
    auto* grnn_right = OpNode("grnn_right", "search_grnn")->AsIntermediate();
    auto* grnn_right_out = VarNode("grnn_right_out")
                               ->assert_is_op_output("search_grnn", "Out")
                               ->AsIntermediate();
    auto* grnn_right_idx_sorted_by_width =
        VarNode("grnn_right_idx_sorted_by_width")
            ->assert_is_op_output("search_grnn", "idx_sorted_by_width")
            ->AsIntermediate();
    auto* grnn_right_layout_input =
        VarNode("grnn_right_layout_input")
            ->assert_is_op_output("search_grnn", "layout_input")
            ->AsIntermediate();
    auto* grnn_right_tmp_buffer =
        VarNode("grnn_right_tmp_buffer")
            ->assert_is_op_output("search_grnn", "tmp_buffer")
            ->AsIntermediate();
    auto* seq_rev_right1 =
        OpNode("seq_rev_right1", "sequence_reverse")->AsIntermediate();
    auto* seq_rev_right1_out =
        VarNode("seq_rev_right1_out")
            ->assert_is_op_output("sequence_reverse", "Y")
            ->AsIntermediate();
    auto* seq_pool_right =
        OpNode("seq_pool_right", "sequence_pool")->AsIntermediate();
    auto* seq_pool_right_out = VarNode("seq_pool_right_out")
                                   ->assert_is_op_output("sequence_pool", "Out")
                                   ->AsOutput();
    auto* seq_pool_right_max_idx =
        VarNode("seq_pool_right_max_idx")
            ->assert_is_op_output("sequence_pool", "MaxIndex")
            ->AsIntermediate();

    auto* grnn_left_wh = VarNode("grnn_left_wh")
                             ->assert_is_op_input("search_grnn", "Wh")
                             ->AsInput();
    auto* grnn_left_wi = VarNode("grnn_left_wi")
                             ->assert_is_op_input("search_grnn", "Wi")
                             ->AsInput();
    auto* grnn_left = OpNode("grnn_left", "search_grnn")->AsIntermediate();
    auto* grnn_left_out = VarNode("grnn_left_out")
                              ->assert_is_op_output("search_grnn", "Out")
                              ->AsIntermediate();
    auto* grnn_left_idx_sorted_by_width =
        VarNode("grnn_left_idx_sorted_by_width")
            ->assert_is_op_output("search_grnn", "idx_sorted_by_width")
            ->AsIntermediate();
    auto* grnn_left_layout_input =
        VarNode("grnn_left_layout_input")
            ->assert_is_op_output("search_grnn", "layout_input")
            ->AsIntermediate();
    auto* grnn_left_tmp_buffer =
        VarNode("grnn_left_tmp_buffer")
            ->assert_is_op_output("search_grnn", "tmp_buffer")
            ->AsIntermediate();
    auto* seq_pool_left =
        OpNode("seq_pool_left", "sequence_pool")->AsIntermediate();
    auto* seq_pool_left_out = VarNode("seq_pool_left_out")
                                  ->assert_is_op_output("sequence_pool", "Out")
                                  ->AsOutput();
    auto* seq_pool_left_max_idx =
        VarNode("seq_pool_left_max_idx")
            ->assert_is_op_output("sequence_pool", "MaxIndex")
            ->AsIntermediate();

    auto* concat_2in1 = OpNode("concat_2in1", "concat")->AsIntermediate();
    auto* concat_2in1_out = VarNode("concat_2in1_out")
                                ->assert_is_op_output("concat", "Out")
                                ->AsIntermediate();
    auto* att_2in1_w =
        VarNode("att_2in1_w")
            ->assert_is_op_input("__xpu__mmdnn_search_attention", "W")
            ->AsInput();
    auto* att_2in1_b =
        VarNode("att_2in1_b")
            ->assert_is_op_input("__xpu__mmdnn_search_attention", "b")
            ->AsInput();
    auto* att_2in1 =
        OpNode("att_2in1", "__xpu__mmdnn_search_attention")->AsIntermediate();
    auto* att_2in1_out =
        VarNode("att_2in1_out")
            ->assert_is_op_output("__xpu__mmdnn_search_attention", "Out")
            ->AsIntermediate();
    auto* seq_pool_2in1 =
        OpNode("seq_pool_2in1", "sequence_pool")->AsIntermediate();
    auto* seq_pool_2in1_out = VarNode("seq_pool_2in1_out")
                                  ->assert_is_op_output("sequence_pool", "Out")
                                  ->AsOutput();
    auto* seq_pool_2in1_max_idx =
        VarNode("seq_pool_2in1_max_idx")
            ->assert_is_op_output("sequence_pool", "MaxIndex")
            ->AsIntermediate();

    auto* concat_3in1 = OpNode("concat_3in1", "concat")->AsIntermediate();
    auto* concat_3in1_out = VarNode("concat_3in1_out")
                                ->assert_is_op_output("concat", "Out")
                                ->AsOutput();

    *input0 >> *emb0 >> *emb0_out >> *eltwise01 >> *eltwise01_out;
    *emb_tbl >> *emb0;
    *input1 >> *emb1 >> *emb1_out >> *eltwise01;
    *emb_tbl >> *emb1;

    *eltwise01_out >> *seq_rev_right0 >> *seq_rev_right0_out >> *grnn_right >>
        *grnn_right_out >> *seq_rev_right1 >> *seq_rev_right1_out;
    *grnn_right_out >> *seq_pool_right >> *seq_pool_right_out;
    *seq_pool_right >> *seq_pool_right_max_idx;
    *grnn_right_wh >> *grnn_right;
    *grnn_right_wi >> *grnn_right;
    *grnn_right >> *grnn_right_idx_sorted_by_width;
    *grnn_right >> *grnn_right_layout_input;
    *grnn_right >> *grnn_right_tmp_buffer;

    *eltwise01_out >> *grnn_left >> *grnn_left_out >> *seq_pool_left >>
        *seq_pool_left_out;
    *seq_pool_left >> *seq_pool_left_max_idx;
    *grnn_left_wh >> *grnn_left;
    *grnn_left_wi >> *grnn_left;
    *grnn_left >> *grnn_left_idx_sorted_by_width;
    *grnn_left >> *grnn_left_layout_input;
    *grnn_left >> *grnn_left_tmp_buffer;

    *seq_rev_right1_out >> *concat_2in1;
    *grnn_left_out >> *concat_2in1;
    *concat_2in1 >> *concat_2in1_out >> *att_2in1 >> *att_2in1_out >>
        *seq_pool_2in1 >> *seq_pool_2in1_out;
    *seq_pool_2in1 >> *seq_pool_2in1_max_idx;
    *att_2in1_w >> *att_2in1;
    *att_2in1_b >> *att_2in1;

    *eltwise01_out >> *concat_3in1;
    *seq_rev_right1_out >> *concat_3in1;
    *grnn_left_out >> *concat_3in1;
    *concat_3in1 >> *concat_3in1_out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__mmdnn_bid_emb_grnn_att");
    op_desc.SetInput("id0", {matched.at("input0")->arg()->name});
    op_desc.SetInput("id1", {matched.at("input1")->arg()->name});
    op_desc.SetInput("emb_tbl", {matched.at("emb_tbl")->arg()->name});
    op_desc.SetInput("grnn_fw_wh", {matched.at("grnn_left_wh")->arg()->name});
    op_desc.SetInput("grnn_fw_wi", {matched.at("grnn_left_wi")->arg()->name});
    op_desc.SetInput("grnn_rv_wh", {matched.at("grnn_right_wh")->arg()->name});
    op_desc.SetInput("grnn_rv_wi", {matched.at("grnn_right_wi")->arg()->name});
    op_desc.SetInput("att_fc_w", {matched.at("att_2in1_w")->arg()->name});
    op_desc.SetInput("att_fc_b", {matched.at("att_2in1_b")->arg()->name});
    op_desc.SetOutput("grnn_fw_pool_out",
                      {matched.at("seq_pool_left_out")->arg()->name});
    op_desc.SetOutput("grnn_rv_pool_out",
                      {matched.at("seq_pool_right_out")->arg()->name});
    op_desc.SetOutput("att_pool_out",
                      {matched.at("seq_pool_2in1_out")->arg()->name});
    op_desc.SetOutput("concat_3in1_out",
                      {matched.at("concat_3in1_out")->arg()->name});
    op_desc.SetOutput("emb_fw_out", {matched.at("eltwise01_out")->arg()->name});

    auto* grnn_fw_op_info = matched.at("grnn_left")->stmt()->op_info();
    op_desc.SetAttr<std::vector<float>>(
        "grnn_fw_wh_maxs",
        grnn_fw_op_info->GetAttr<std::vector<float>>("__xpu__wh_max"));
    op_desc.SetAttr<std::vector<float>>(
        "grnn_fw_wi_maxs",
        grnn_fw_op_info->GetAttr<std::vector<float>>("__xpu__wi_max"));
    auto* grnn_rv_op_info = matched.at("grnn_right")->stmt()->op_info();
    op_desc.SetAttr<std::vector<float>>(
        "grnn_rv_wh_maxs",
        grnn_rv_op_info->GetAttr<std::vector<float>>("__xpu__wh_max"));
    op_desc.SetAttr<std::vector<float>>(
        "grnn_rv_wi_maxs",
        grnn_rv_op_info->GetAttr<std::vector<float>>("__xpu__wi_max"));
    auto* att_fc_op_info = matched.at("att_2in1")->stmt()->op_info();
    op_desc.SetAttr<float>("att_fc_w_max",
                           att_fc_op_info->GetAttr<float>("W_max"));

    auto* new_stmt = matched.at("emb0")->stmt();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, new_stmt->op()->scope());
    new_op->SetValidPlaces(new_stmt->op()->valid_places());
    auto kernels = new_op->CreateKernels(new_op->valid_places());
    new_stmt->SetOp(new_op);
    new_stmt->SetKernels(std::move(kernels));

    std::vector<std::string> arg_names{
        "input1",
        "grnn_left_wh",
        "grnn_left_wi",
        "grnn_right_wh",
        "grnn_right_wi",
        "att_2in1_w",
        "att_2in1_b",
    };
    for (auto name : arg_names) {
      DirectedLink(matched.at(name), matched.at("emb0"));
    }
    std::vector<std::string> out_names{
        "seq_pool_left_out",
        "seq_pool_right_out",
        "seq_pool_2in1_out",
        "concat_3in1_out",
        "eltwise01_out",
    };
    for (auto name : out_names) {
      IR_OP_VAR_LINK(matched.at("emb0"), matched.at(name));
    }
  }
};

// 6 outputs
// =========
//
// emb0_out
// eltwise01_out
// seq_pool_right_out
// seq_pool_left_out
// seq_pool_2in1_out
// concat_3in1_out
//
class XPUMmdnnBidEmbGrnnAttFuser2 : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input0 = VarNode("input0")->AsInput();
    auto* input1 = VarNode("input1")->AsInput();
    auto* emb_tbl = VarNode("emb_tbl")->AsInput();

    auto* emb0 = OpNode("emb0", "lookup_table");
    auto* emb0_out = VarNode("emb0_out")
                         ->assert_is_op_output("lookup_table", "Out")
                         ->assert_is_op_input("search_seq_arithmetic", "X")
                         ->AsOutput();
    auto* emb1 = OpNode("emb1", "lookup_table")->AsIntermediate();
    auto* emb1_out = VarNode("emb1_out")
                         ->assert_is_op_output("lookup_table", "Out")
                         ->assert_is_op_input("search_seq_arithmetic", "Y")
                         ->AsIntermediate();
    auto* eltwise01 =
        OpNode("eltwise01", "search_seq_arithmetic")->AsIntermediate();
    auto* eltwise01_out =
        VarNode("eltwise01_out")
            ->assert_is_op_output("search_seq_arithmetic", "Out")
            ->AsOutput();

    auto* seq_rev_right0 =
        OpNode("seq_rev_right0", "sequence_reverse")->AsIntermediate();
    auto* seq_rev_right0_out =
        VarNode("seq_rev_right0_out")
            ->assert_is_op_output("sequence_reverse", "Y")
            ->AsIntermediate();
    auto* grnn_right_wh = VarNode("grnn_right_wh")
                              ->assert_is_op_input("search_grnn", "Wh")
                              ->AsInput();
    auto* grnn_right_wi = VarNode("grnn_right_wi")
                              ->assert_is_op_input("search_grnn", "Wi")
                              ->AsInput();
    auto* grnn_right = OpNode("grnn_right", "search_grnn")->AsIntermediate();
    auto* grnn_right_out = VarNode("grnn_right_out")
                               ->assert_is_op_output("search_grnn", "Out")
                               ->AsIntermediate();
    auto* grnn_right_idx_sorted_by_width =
        VarNode("grnn_right_idx_sorted_by_width")
            ->assert_is_op_output("search_grnn", "idx_sorted_by_width")
            ->AsIntermediate();
    auto* grnn_right_layout_input =
        VarNode("grnn_right_layout_input")
            ->assert_is_op_output("search_grnn", "layout_input")
            ->AsIntermediate();
    auto* grnn_right_tmp_buffer =
        VarNode("grnn_right_tmp_buffer")
            ->assert_is_op_output("search_grnn", "tmp_buffer")
            ->AsIntermediate();
    auto* seq_rev_right1 =
        OpNode("seq_rev_right1", "sequence_reverse")->AsIntermediate();
    auto* seq_rev_right1_out =
        VarNode("seq_rev_right1_out")
            ->assert_is_op_output("sequence_reverse", "Y")
            ->AsIntermediate();
    auto* seq_pool_right =
        OpNode("seq_pool_right", "sequence_pool")->AsIntermediate();
    auto* seq_pool_right_out = VarNode("seq_pool_right_out")
                                   ->assert_is_op_output("sequence_pool", "Out")
                                   ->AsOutput();
    auto* seq_pool_right_max_idx =
        VarNode("seq_pool_right_max_idx")
            ->assert_is_op_output("sequence_pool", "MaxIndex")
            ->AsIntermediate();

    auto* grnn_left_wh = VarNode("grnn_left_wh")
                             ->assert_is_op_input("search_grnn", "Wh")
                             ->AsInput();
    auto* grnn_left_wi = VarNode("grnn_left_wi")
                             ->assert_is_op_input("search_grnn", "Wi")
                             ->AsInput();
    auto* grnn_left = OpNode("grnn_left", "search_grnn")->AsIntermediate();
    auto* grnn_left_out = VarNode("grnn_left_out")
                              ->assert_is_op_output("search_grnn", "Out")
                              ->AsIntermediate();
    auto* grnn_left_idx_sorted_by_width =
        VarNode("grnn_left_idx_sorted_by_width")
            ->assert_is_op_output("search_grnn", "idx_sorted_by_width")
            ->AsIntermediate();
    auto* grnn_left_layout_input =
        VarNode("grnn_left_layout_input")
            ->assert_is_op_output("search_grnn", "layout_input")
            ->AsIntermediate();
    auto* grnn_left_tmp_buffer =
        VarNode("grnn_left_tmp_buffer")
            ->assert_is_op_output("search_grnn", "tmp_buffer")
            ->AsIntermediate();
    auto* seq_pool_left =
        OpNode("seq_pool_left", "sequence_pool")->AsIntermediate();
    auto* seq_pool_left_out = VarNode("seq_pool_left_out")
                                  ->assert_is_op_output("sequence_pool", "Out")
                                  ->AsOutput();
    auto* seq_pool_left_max_idx =
        VarNode("seq_pool_left_max_idx")
            ->assert_is_op_output("sequence_pool", "MaxIndex")
            ->AsIntermediate();

    auto* concat_2in1 = OpNode("concat_2in1", "concat")->AsIntermediate();
    auto* concat_2in1_out = VarNode("concat_2in1_out")
                                ->assert_is_op_output("concat", "Out")
                                ->AsIntermediate();
    auto* att_2in1_w =
        VarNode("att_2in1_w")
            ->assert_is_op_input("__xpu__mmdnn_search_attention", "W")
            ->AsInput();
    auto* att_2in1_b =
        VarNode("att_2in1_b")
            ->assert_is_op_input("__xpu__mmdnn_search_attention", "b")
            ->AsInput();
    auto* att_2in1 =
        OpNode("att_2in1", "__xpu__mmdnn_search_attention")->AsIntermediate();
    auto* att_2in1_out =
        VarNode("att_2in1_out")
            ->assert_is_op_output("__xpu__mmdnn_search_attention", "Out")
            ->AsIntermediate();
    auto* seq_pool_2in1 =
        OpNode("seq_pool_2in1", "sequence_pool")->AsIntermediate();
    auto* seq_pool_2in1_out = VarNode("seq_pool_2in1_out")
                                  ->assert_is_op_output("sequence_pool", "Out")
                                  ->AsOutput();
    auto* seq_pool_2in1_max_idx =
        VarNode("seq_pool_2in1_max_idx")
            ->assert_is_op_output("sequence_pool", "MaxIndex")
            ->AsIntermediate();

    auto* concat_3in1 = OpNode("concat_3in1", "concat")->AsIntermediate();
    auto* concat_3in1_out = VarNode("concat_3in1_out")
                                ->assert_is_op_output("concat", "Out")
                                ->AsOutput();

    *input0 >> *emb0 >> *emb0_out >> *eltwise01 >> *eltwise01_out;
    *emb_tbl >> *emb0;
    *input1 >> *emb1 >> *emb1_out >> *eltwise01;
    *emb_tbl >> *emb1;

    *eltwise01_out >> *seq_rev_right0 >> *seq_rev_right0_out >> *grnn_right >>
        *grnn_right_out >> *seq_rev_right1 >> *seq_rev_right1_out;
    *grnn_right_out >> *seq_pool_right >> *seq_pool_right_out;
    *seq_pool_right >> *seq_pool_right_max_idx;
    *grnn_right_wh >> *grnn_right;
    *grnn_right_wi >> *grnn_right;
    *grnn_right >> *grnn_right_idx_sorted_by_width;
    *grnn_right >> *grnn_right_layout_input;
    *grnn_right >> *grnn_right_tmp_buffer;

    *eltwise01_out >> *grnn_left >> *grnn_left_out >> *seq_pool_left >>
        *seq_pool_left_out;
    *seq_pool_left >> *seq_pool_left_max_idx;
    *grnn_left_wh >> *grnn_left;
    *grnn_left_wi >> *grnn_left;
    *grnn_left >> *grnn_left_idx_sorted_by_width;
    *grnn_left >> *grnn_left_layout_input;
    *grnn_left >> *grnn_left_tmp_buffer;

    *seq_rev_right1_out >> *concat_2in1;
    *grnn_left_out >> *concat_2in1;
    *concat_2in1 >> *concat_2in1_out >> *att_2in1 >> *att_2in1_out >>
        *seq_pool_2in1 >> *seq_pool_2in1_out;
    *seq_pool_2in1 >> *seq_pool_2in1_max_idx;
    *att_2in1_w >> *att_2in1;
    *att_2in1_b >> *att_2in1;

    *eltwise01_out >> *concat_3in1;
    *seq_rev_right1_out >> *concat_3in1;
    *grnn_left_out >> *concat_3in1;
    *concat_3in1 >> *concat_3in1_out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__mmdnn_bid_emb_grnn_att2");
    op_desc.SetInput("id0", {matched.at("input0")->arg()->name});
    op_desc.SetInput("id1", {matched.at("input1")->arg()->name});
    op_desc.SetInput("emb_tbl", {matched.at("emb_tbl")->arg()->name});
    op_desc.SetInput("grnn_fw_wh", {matched.at("grnn_left_wh")->arg()->name});
    op_desc.SetInput("grnn_fw_wi", {matched.at("grnn_left_wi")->arg()->name});
    op_desc.SetInput("grnn_rv_wh", {matched.at("grnn_right_wh")->arg()->name});
    op_desc.SetInput("grnn_rv_wi", {matched.at("grnn_right_wi")->arg()->name});
    op_desc.SetInput("att_fc_w", {matched.at("att_2in1_w")->arg()->name});
    op_desc.SetInput("att_fc_b", {matched.at("att_2in1_b")->arg()->name});
    op_desc.SetOutput("emb0_out", {matched.at("emb0_out")->arg()->name});
    op_desc.SetOutput("grnn_fw_pool_out",
                      {matched.at("seq_pool_left_out")->arg()->name});
    op_desc.SetOutput("grnn_rv_pool_out",
                      {matched.at("seq_pool_right_out")->arg()->name});
    op_desc.SetOutput("att_pool_out",
                      {matched.at("seq_pool_2in1_out")->arg()->name});
    op_desc.SetOutput("concat_3in1_out",
                      {matched.at("concat_3in1_out")->arg()->name});
    op_desc.SetOutput("emb_fw_out", {matched.at("eltwise01_out")->arg()->name});

    auto* grnn_fw_op_info = matched.at("grnn_left")->stmt()->op_info();
    op_desc.SetAttr<std::vector<float>>(
        "grnn_fw_wh_maxs",
        grnn_fw_op_info->GetAttr<std::vector<float>>("__xpu__wh_max"));
    op_desc.SetAttr<std::vector<float>>(
        "grnn_fw_wi_maxs",
        grnn_fw_op_info->GetAttr<std::vector<float>>("__xpu__wi_max"));
    auto* grnn_rv_op_info = matched.at("grnn_right")->stmt()->op_info();
    op_desc.SetAttr<std::vector<float>>(
        "grnn_rv_wh_maxs",
        grnn_rv_op_info->GetAttr<std::vector<float>>("__xpu__wh_max"));
    op_desc.SetAttr<std::vector<float>>(
        "grnn_rv_wi_maxs",
        grnn_rv_op_info->GetAttr<std::vector<float>>("__xpu__wi_max"));
    auto* att_fc_op_info = matched.at("att_2in1")->stmt()->op_info();
    op_desc.SetAttr<float>("att_fc_w_max",
                           att_fc_op_info->GetAttr<float>("W_max"));

    auto* new_stmt = matched.at("emb0")->stmt();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, new_stmt->op()->scope());
    new_op->SetValidPlaces(new_stmt->op()->valid_places());
    auto kernels = new_op->CreateKernels(new_op->valid_places());
    new_stmt->SetOp(new_op);
    new_stmt->SetKernels(std::move(kernels));

    std::vector<std::string> arg_names{
        "input1",
        "grnn_left_wh",
        "grnn_left_wi",
        "grnn_right_wh",
        "grnn_right_wi",
        "att_2in1_w",
        "att_2in1_b",
    };
    for (auto name : arg_names) {
      DirectedLink(matched.at(name), matched.at("emb0"));
    }
    std::vector<std::string> out_names{
        "seq_pool_left_out",
        "seq_pool_right_out",
        "seq_pool_2in1_out",
        "concat_3in1_out",
        "eltwise01_out",
    };
    for (auto name : out_names) {
      IR_OP_VAR_LINK(matched.at("emb0"), matched.at(name));
    }
  }
};

class XPUMmdnnMergeAllFuser : public FuseBase {
 public:
  explicit XPUMmdnnMergeAllFuser(int n_concat_topk)
      : n_concat_topk_(n_concat_topk) {}

  void BuildPattern() override {
    auto* concat_7in1_input0 = VarNode("concat_7in1_input0")
                                   ->assert_is_op_nth_input("concat", "X", 0)
                                   ->AsInput();
    auto* concat_7in1_input1 = VarNode("concat_7in1_input1")
                                   ->assert_is_op_nth_input("concat", "X", 1)
                                   ->AsInput();
    auto* concat_7in1_input2 = VarNode("concat_7in1_input2")
                                   ->assert_is_op_nth_input("concat", "X", 2)
                                   ->AsInput();
    auto* concat_7in1_input3 = VarNode("concat_7in1_input3")
                                   ->assert_is_op_nth_input("concat", "X", 3)
                                   ->AsInput();
    auto* concat_7in1_input4 = VarNode("concat_7in1_input4")
                                   ->assert_is_op_nth_input("concat", "X", 4)
                                   ->AsInput();
    auto* concat_7in1_input5 = VarNode("concat_7in1_input5")
                                   ->assert_is_op_nth_input("concat", "X", 5)
                                   ->AsInput();
    auto* concat_7in1_input6 = VarNode("concat_7in1_input6")
                                   ->assert_is_op_nth_input("concat", "X", 6)
                                   ->AsInput();
    auto* concat_7in1 = OpNode("concat_7in1", "concat");
    auto* concat_7in1_out = VarNode("concat_7in1_out")
                                ->assert_is_op_output("concat", "Out")
                                ->AsIntermediate();
    auto* search_fc0_w = VarNode("search_fc0_w")
                             ->assert_is_op_input("search_fc", "W")
                             ->AsInput();
    auto* search_fc0_b = VarNode("search_fc0_b")
                             ->assert_is_op_input("search_fc", "b")
                             ->AsInput();
    auto* search_fc0 = OpNode("search_fc0", "search_fc")->AsIntermediate();
    auto* search_fc0_out = VarNode("search_fc0_out")
                               ->assert_is_op_output("search_fc", "Out")
                               ->AsIntermediate();
    auto* relu0 = OpNode("relu0", "relu")->AsIntermediate();
    auto* relu0_out = VarNode("relu0_out")
                          ->assert_is_op_output("relu", "Out")
                          ->AsIntermediate();

    auto* concat_topk_input0 = VarNode("concat_topk_input0")
                                   ->assert_is_op_nth_input("concat", "X", 0)
                                   ->AsInput();
    auto* concat_topk_input1 = VarNode("concat_topk_input1")
                                   ->assert_is_op_nth_input("concat", "X", 1)
                                   ->AsInput();
    auto* concat_topk = OpNode("concat_topk", "concat")->AsIntermediate();
    auto* concat_topk_out = VarNode("concat_topk_out")
                                ->assert_is_op_output("concat", "Out")
                                ->AsIntermediate();
    for (int i = 2; i < n_concat_topk_; ++i) {
      auto concat_topk_input_name =
          paddle::lite::string_format("concat_topk_input%d", i);
      auto* concat_topk_inputx = VarNode(concat_topk_input_name)
                                     ->assert_is_op_nth_input("concat", "X", i)
                                     ->AsInput();
      *concat_topk_inputx >> *concat_topk;
    }

    auto* seq_rev = OpNode("seq_rev", "sequence_reverse")->AsIntermediate();
    auto* seq_rev_out = VarNode("seq_rev_out")
                            ->assert_is_op_output("sequence_reverse", "Y")
                            ->AsIntermediate();

    auto* grnn_rv_wh = VarNode("grnn_rv_wh")
                           ->assert_is_op_input("search_grnn", "Wh")
                           ->AsInput();
    auto* grnn_rv_wi = VarNode("grnn_rv_wi")
                           ->assert_is_op_input("search_grnn", "Wi")
                           ->AsInput();
    auto* grnn_rv = OpNode("grnn_rv", "search_grnn")->AsIntermediate();
    auto* grnn_rv_out = VarNode("grnn_rv_out")
                            ->assert_is_op_output("search_grnn", "Out")
                            ->AsIntermediate();
    auto* grnn_rv_idx_sorted_by_width =
        VarNode("grnn_rv_idx_sorted_by_width")
            ->assert_is_op_output("search_grnn", "idx_sorted_by_width")
            ->AsIntermediate();
    auto* grnn_rv_layout_input =
        VarNode("grnn_rv_layout_input")
            ->assert_is_op_output("search_grnn", "layout_input")
            ->AsIntermediate();
    auto* grnn_rv_tmp_buffer =
        VarNode("grnn_rv_tmp_buffer")
            ->assert_is_op_output("search_grnn", "tmp_buffer")
            ->AsIntermediate();
    auto* seq_pool_rv =
        OpNode("seq_pool_rv", "sequence_pool")->AsIntermediate();
    auto* seq_pool_rv_out = VarNode("seq_pool_rv_out")
                                ->assert_is_op_output("sequence_pool", "Out")
                                ->AsIntermediate();
    auto* seq_pool_rv_max_idx =
        VarNode("seq_pool_rv_max_idx")
            ->assert_is_op_output("sequence_pool", "MaxIndex")
            ->AsIntermediate();

    auto* grnn_fw_wh = VarNode("grnn_fw_wh")
                           ->assert_is_op_input("search_grnn", "Wh")
                           ->AsInput();
    auto* grnn_fw_wi = VarNode("grnn_fw_wi")
                           ->assert_is_op_input("search_grnn", "Wi")
                           ->AsInput();
    auto* grnn_fw = OpNode("grnn_fw", "search_grnn")->AsIntermediate();
    auto* grnn_fw_out = VarNode("grnn_fw_out")
                            ->assert_is_op_output("search_grnn", "Out")
                            ->AsIntermediate();
    auto* grnn_fw_idx_sorted_by_width =
        VarNode("grnn_fw_idx_sorted_by_width")
            ->assert_is_op_output("search_grnn", "idx_sorted_by_width")
            ->AsIntermediate();
    auto* grnn_fw_layout_input =
        VarNode("grnn_fw_layout_input")
            ->assert_is_op_output("search_grnn", "layout_input")
            ->AsIntermediate();
    auto* grnn_fw_tmp_buffer =
        VarNode("grnn_fw_tmp_buffer")
            ->assert_is_op_output("search_grnn", "tmp_buffer")
            ->AsIntermediate();
    auto* seq_pool_fw =
        OpNode("seq_pool_fw", "sequence_pool")->AsIntermediate();
    auto* seq_pool_fw_out = VarNode("seq_pool_fw_out")
                                ->assert_is_op_output("sequence_pool", "Out")
                                ->AsIntermediate();
    auto* seq_pool_fw_max_idx =
        VarNode("seq_pool_fw_max_idx")
            ->assert_is_op_output("sequence_pool", "MaxIndex")
            ->AsIntermediate();

    auto* rv_fw_concat = OpNode("rv_fw_concat", "concat")->AsIntermediate();
    auto* rv_fw_concat_out = VarNode("rv_fw_concat_out")
                                 ->assert_is_op_output("concat", "Out")
                                 ->AsIntermediate();

    auto* last_concat = OpNode("last_concat", "concat")->AsIntermediate();
    auto* last_concat_out = VarNode("last_concat_out")
                                ->assert_is_op_output("concat", "Out")
                                ->AsIntermediate();
    auto* search_fc1_w = VarNode("search_fc1_w")
                             ->assert_is_op_input("search_fc", "W")
                             ->AsInput();
    auto* search_fc1_b = VarNode("search_fc1_b")
                             ->assert_is_op_input("search_fc", "b")
                             ->AsInput();
    auto* search_fc1 = OpNode("search_fc1", "search_fc")->AsIntermediate();
    auto* search_fc1_out = VarNode("search_fc1_out")
                               ->assert_is_op_output("search_fc", "Out")
                               ->AsIntermediate();
    auto* relu1 = OpNode("relu1", "relu")->AsIntermediate();
    auto* relu1_out = VarNode("relu1_out")
                          ->assert_is_op_output("relu", "Out")
                          ->AsIntermediate();
    auto* search_fc2_w = VarNode("search_fc2_w")
                             ->assert_is_op_input("search_fc", "W")
                             ->AsInput();
    auto* search_fc2_b = VarNode("search_fc2_b")
                             ->assert_is_op_input("search_fc", "b")
                             ->AsInput();
    auto* search_fc2 = OpNode("search_fc2", "search_fc")->AsIntermediate();
    auto* search_fc2_out = VarNode("search_fc2_out")
                               ->assert_is_op_output("search_fc", "Out")
                               ->AsOutput();

    *concat_7in1_input0 >> *concat_7in1;
    *concat_7in1_input1 >> *concat_7in1;
    *concat_7in1_input2 >> *concat_7in1;
    *concat_7in1_input3 >> *concat_7in1;
    *concat_7in1_input4 >> *concat_7in1;
    *concat_7in1_input5 >> *concat_7in1;
    *concat_7in1_input6 >> *concat_7in1;
    *concat_7in1 >> *concat_7in1_out >> *search_fc0 >> *search_fc0_out >>
        *relu0 >> *relu0_out;
    *search_fc0_w >> *search_fc0;
    *search_fc0_b >> *search_fc0;

    *concat_topk_input0 >> *concat_topk;
    *concat_topk_input1 >> *concat_topk;
    *concat_topk >> *concat_topk_out >> *seq_rev >> *seq_rev_out;

    *seq_rev_out >> *grnn_rv >> *grnn_rv_out >> *seq_pool_rv >>
        *seq_pool_rv_out;
    *seq_pool_rv >> *seq_pool_rv_max_idx;
    *grnn_rv_wh >> *grnn_rv;
    *grnn_rv_wi >> *grnn_rv;
    *grnn_rv >> *grnn_rv_idx_sorted_by_width;
    *grnn_rv >> *grnn_rv_layout_input;
    *grnn_rv >> *grnn_rv_tmp_buffer;

    *concat_topk_out >> *grnn_fw >> *grnn_fw_out >> *seq_pool_fw >>
        *seq_pool_fw_out;
    *seq_pool_fw >> *seq_pool_fw_max_idx;
    *grnn_fw_wh >> *grnn_fw;
    *grnn_fw_wi >> *grnn_fw;
    *grnn_fw >> *grnn_fw_idx_sorted_by_width;
    *grnn_fw >> *grnn_fw_layout_input;
    *grnn_fw >> *grnn_fw_tmp_buffer;

    *seq_pool_rv_out >> *rv_fw_concat;
    *seq_pool_fw_out >> *rv_fw_concat;
    *rv_fw_concat >> *rv_fw_concat_out;

    *rv_fw_concat_out >> *last_concat;
    *relu0_out >> *last_concat;
    *last_concat >> *last_concat_out >> *search_fc1 >> *search_fc1_out >>
        *relu1 >> *relu1_out >> *search_fc2 >> *search_fc2_out;
    *search_fc1_w >> *search_fc1;
    *search_fc1_b >> *search_fc1;
    *search_fc2_w >> *search_fc2;
    *search_fc2_b >> *search_fc2;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__mmdnn_merge_all");
    auto* concat_7in1_op_info = matched.at("concat_7in1")->stmt()->op_info();
    op_desc.SetInput("concat_7in1_x", concat_7in1_op_info->Input("X"));
    auto* concat_topk_op_info = matched.at("concat_topk")->stmt()->op_info();
    op_desc.SetInput("concat_topk_x", concat_topk_op_info->Input("X"));
    op_desc.SetInput("grnn_fw_wh", {matched.at("grnn_fw_wh")->arg()->name});
    op_desc.SetInput("grnn_fw_wi", {matched.at("grnn_fw_wi")->arg()->name});
    op_desc.SetInput("grnn_rv_wh", {matched.at("grnn_rv_wh")->arg()->name});
    op_desc.SetInput("grnn_rv_wi", {matched.at("grnn_rv_wi")->arg()->name});
    op_desc.SetInput("fc0_w", {matched.at("search_fc0_w")->arg()->name});
    op_desc.SetInput("fc0_b", {matched.at("search_fc0_b")->arg()->name});
    op_desc.SetInput("fc1_w", {matched.at("search_fc1_w")->arg()->name});
    op_desc.SetInput("fc1_b", {matched.at("search_fc1_b")->arg()->name});
    op_desc.SetInput("fc2_w", {matched.at("search_fc2_w")->arg()->name});
    op_desc.SetInput("fc2_b", {matched.at("search_fc2_b")->arg()->name});

    op_desc.SetOutput("out", {matched.at("search_fc2_out")->arg()->name});

    auto* grnn_fw_op_info = matched.at("grnn_fw")->stmt()->op_info();
    op_desc.SetAttr<std::vector<float>>(
        "grnn_fw_wh_maxs",
        grnn_fw_op_info->GetAttr<std::vector<float>>("__xpu__wh_max"));
    op_desc.SetAttr<std::vector<float>>(
        "grnn_fw_wi_maxs",
        grnn_fw_op_info->GetAttr<std::vector<float>>("__xpu__wi_max"));
    auto* grnn_rv_op_info = matched.at("grnn_rv")->stmt()->op_info();
    op_desc.SetAttr<std::vector<float>>(
        "grnn_rv_wh_maxs",
        grnn_rv_op_info->GetAttr<std::vector<float>>("__xpu__wh_max"));
    op_desc.SetAttr<std::vector<float>>(
        "grnn_rv_wi_maxs",
        grnn_rv_op_info->GetAttr<std::vector<float>>("__xpu__wi_max"));
    auto* fc0_op_info = matched.at("search_fc0")->stmt()->op_info();
    op_desc.SetAttr<float>("fc0_w_max",
                           fc0_op_info->GetAttr<float>("__xpu__w_max"));
    auto* fc1_op_info = matched.at("search_fc1")->stmt()->op_info();
    op_desc.SetAttr<float>("fc1_w_max",
                           fc1_op_info->GetAttr<float>("__xpu__w_max"));
    auto* fc2_op_info = matched.at("search_fc2")->stmt()->op_info();
    op_desc.SetAttr<float>("fc2_w_max",
                           fc2_op_info->GetAttr<float>("__xpu__w_max"));

    auto* new_stmt = matched.at("concat_7in1")->stmt();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, new_stmt->op()->scope());
    new_op->SetValidPlaces(new_stmt->op()->valid_places());
    auto kernels = new_op->CreateKernels(new_op->valid_places());
    new_stmt->SetOp(new_op);
    new_stmt->SetKernels(std::move(kernels));

    std::vector<std::string> arg_names{
        "concat_topk_input0",
        "concat_topk_input1",
        "grnn_fw_wh",
        "grnn_fw_wi",
        "grnn_rv_wh",
        "grnn_rv_wi",
        "search_fc0_w",
        "search_fc0_b",
        "search_fc1_w",
        "search_fc1_b",
        "search_fc2_w",
        "search_fc2_b",
    };
    for (int i = 2; i < n_concat_topk_; ++i) {
      auto concat_topk_input_name =
          paddle::lite::string_format("concat_topk_input%d", i);
      arg_names.push_back(concat_topk_input_name);
    }
    for (auto name : arg_names) {
      DirectedLink(matched.at(name), matched.at("concat_7in1"));
    }
    std::vector<std::string> out_names{
        "search_fc2_out",
    };
    for (auto name : out_names) {
      IR_OP_VAR_LINK(matched.at("concat_7in1"), matched.at(name));
    }
  }

 private:
  int n_concat_topk_;
};

}  // namespace fusion

class XPUMmdnnFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;

    fusion::XPUMmdnnFloat2Fix float_2_fix;
    float_2_fix(graph.get());
    fusion::XPUMmdnnSearchAttentionFuser search_att_fuser;
    search_att_fuser(graph.get());
    fusion::XPUMmdnnSearchAttentionFuser2 search_att_fuser2;
    search_att_fuser2(graph.get());
    fusion::XPUMmdnnMatchConvTopkFuser match_conv_topk_fuser;
    match_conv_topk_fuser(graph.get());
    fusion::XPUMmdnnMatchConvTopkFuser2 match_conv_topk_fuser2;
    match_conv_topk_fuser2(graph.get());

    fusion::XPUMmdnnBidSeqRevEmbEltwiseFuser bi_seq_rev_emb_eltwise_fuser_false(
        false);
    bi_seq_rev_emb_eltwise_fuser_false(graph.get());
    fusion::XPUMmdnnBidSeqRevEmbEltwiseFuser bi_seq_rev_emb_eltwise_fuser_true(
        true);
    bi_seq_rev_emb_eltwise_fuser_true(graph.get());
    fusion::XPUMmdnnBidEmbGrnnAttFuser bid_emb_grnn_att_fuser;
    bid_emb_grnn_att_fuser(graph.get());
    fusion::XPUMmdnnBidEmbGrnnAttFuser2 bid_emb_grnn_att_fuser2;
    bid_emb_grnn_att_fuser2(graph.get());
    fusion::XPUMmdnnBidEmbAttFuser bid_emb_att_fuser;
    bid_emb_att_fuser(graph.get());
    for (int n_concat_topk : {3, 2}) {
      fusion::XPUMmdnnMergeAllFuser merge_all_fuser(n_concat_topk);
      merge_all_fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__mmdnn_fuse_pass, paddle::lite::mir::XPUMmdnnFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__mmdnn_search_attention")
    .BindKernel("__xpu__mmdnn_bid_emb_grnn_att")
    .BindKernel("__xpu__mmdnn_bid_emb_grnn_att2")
    .BindKernel("__xpu__mmdnn_bid_emb_att")
    .BindKernel("__xpu__mmdnn_match_conv_topk")
    .BindKernel("__xpu__mmdnn_merge_all");
