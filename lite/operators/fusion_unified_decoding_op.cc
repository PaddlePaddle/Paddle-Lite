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

#include "lite/operators/fusion_unified_decoding_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool FusionUnifiedDecodingOp::CheckShape() const { return true; }

bool FusionUnifiedDecodingOp::InferShapeImpl() const { return true; }

bool FusionUnifiedDecodingOp::AttachImpl(const cpp::OpDesc& op_desc,
                                         lite::Scope* scope) {
  param_.attn_mask_ =
      GetMutableVar<lite::Tensor>(scope, op_desc.Input("AttnMask").front());
  param_.decoder_position_ids_ = GetMutableVar<lite::Tensor>(
      scope, op_desc.Input("DecPositionIds").front());
  param_.decoder_role_id_ =
      GetMutableVar<lite::Tensor>(scope, op_desc.Input("DecRoleIds").front());
  param_.decoder_type_id_ =
      GetMutableVar<lite::Tensor>(scope, op_desc.Input("DecTypeIds").front());
  param_.decoder_ln_bias_ = GetMutableVar<lite::Tensor>(
      scope, op_desc.Input("DecoderLayernormBias").front());
  param_.decoder_ln_weight_ = GetMutableVar<lite::Tensor>(
      scope, op_desc.Input("DecoderLayernormWeight").front());
  param_.embedding_bias_ =
      GetMutableVar<lite::Tensor>(scope, op_desc.Input("EmbBias").front());
  param_.embedding_weight_ =
      GetMutableVar<lite::Tensor>(scope, op_desc.Input("EmbWeight").front());

  param_.ffn_inter_bias_.clear();
  for (auto& name : op_desc.Input("FFNInterBias@VECTOR")) {
    param_.ffn_inter_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.ffn_inter_weight_.clear();
  for (auto& name : op_desc.Input("FFNInterWeight@VECTOR")) {
    param_.ffn_inter_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.ffn_ln_bias_.clear();
  for (auto& name : op_desc.Input("FFNLayernormBias@VECTOR")) {
    param_.ffn_ln_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.ffn_ln_weight_.clear();
  for (auto& name : op_desc.Input("FFNLayernormWeight@VECTOR")) {
    param_.ffn_ln_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.ffn_out_bias_.clear();
  for (auto& name : op_desc.Input("FFNOutBias@VECTOR")) {
    param_.ffn_out_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.ffn_out_weight_.clear();
  for (auto& name : op_desc.Input("FFNOutWeight@VECTOR")) {
    param_.ffn_out_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.input_ids_ =
      GetMutableVar<lite::Tensor>(scope, op_desc.Input("InputIds").front());
  param_.lm_ln_bias_ = GetMutableVar<lite::Tensor>(
      scope, op_desc.Input("LMLayernormBias").front());
  param_.lm_ln_weight_ = GetMutableVar<lite::Tensor>(
      scope, op_desc.Input("LMLayernormWeight").front());
  param_.logits_mask_ =
      GetMutableVar<lite::Tensor>(scope, op_desc.Input("LogitsMask").front());
  param_.mem_seq_len_ =
      GetMutableVar<lite::Tensor>(scope, op_desc.Input("MemSeqLen").front());
  param_.positional_embedding_weight_ = GetMutableVar<lite::Tensor>(
      scope, op_desc.Input("PositionEncEmb").front());
  param_.position_ids_ =
      GetMutableVar<lite::Tensor>(scope, op_desc.Input("PositionIds").front());
  param_.role_embedding_table_ = GetMutableVar<lite::Tensor>(
      scope, op_desc.Input("RoleEmbedding").front());
  param_.role_id_ =
      GetMutableVar<lite::Tensor>(scope, op_desc.Input("RoleIds").front());

  param_.self_k_bias_.clear();
  for (auto& name : op_desc.Input("SelfKeyBias@VECTOR")) {
    param_.self_k_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_k_weight_.clear();
  for (auto& name : op_desc.Input("SelfKeyWeight@VECTOR")) {
    param_.self_k_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_ln_bias_.clear();
  for (auto& name : op_desc.Input("SelfLayernormBias@VECTOR")) {
    param_.self_ln_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_ln_weight_.clear();
  for (auto& name : op_desc.Input("SelfLayernormWeight@VECTOR")) {
    param_.self_ln_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_out_bias_.clear();
  for (auto& name : op_desc.Input("SelfOutBias@VECTOR")) {
    param_.self_out_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_out_weight_.clear();
  for (auto& name : op_desc.Input("SelfOutWeight@VECTOR")) {
    param_.self_out_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_q_bias_.clear();
  for (auto& name : op_desc.Input("SelfQueryBias@VECTOR")) {
    param_.self_q_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_q_weight_.clear();
  for (auto& name : op_desc.Input("SelfQueryWeight@VECTOR")) {
    param_.self_q_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_v_bias_.clear();
  for (auto& name : op_desc.Input("SelfValueBias@VECTOR")) {
    param_.self_v_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_v_weight_.clear();
  for (auto& name : op_desc.Input("SelfValueWeight@VECTOR")) {
    param_.self_v_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }

  param_.trans_bias_ =
      GetVar<lite::Tensor>(scope, op_desc.Input("TransBias").front());
  param_.trans_weight_ =
      GetVar<lite::Tensor>(scope, op_desc.Input("TransWeight").front());
  param_.type_embedding_weight_ =
      GetVar<lite::Tensor>(scope, op_desc.Input("TypeEmb").front());
  param_.type_id_ =
      GetVar<lite::Tensor>(scope, op_desc.Input("TypeIds").front());
  param_.word_embedding_ =
      GetVar<lite::Tensor>(scope, op_desc.Input("WordEmbedding").front());

  param_.output_ids_ =
      GetMutableVar<lite::Tensor>(scope, op_desc.Output("OutputIds").front());
  param_.output_scores_ = GetMutableVar<lite::Tensor>(
      scope, op_desc.Output("OutputScores").front());
  param_.parent_ids_ =
      GetMutableVar<lite::Tensor>(scope, op_desc.Output("ParentIds").front());
  param_.sequence_length_ = GetMutableVar<lite::Tensor>(
      scope, op_desc.Output("SequenceLength").front());

  param_.decoding_strategy_ = op_desc.GetAttr<std::string>("decoding_strategy");
  param_.beam_size_ = op_desc.GetAttr<int32_t>("beam_size");
  param_.topk_ = op_desc.GetAttr<int32_t>("topk");
  param_.topp_ = op_desc.GetAttr<float>("topp");
  param_.n_head_ = op_desc.GetAttr<int32_t>("n_head");
  param_.size_per_head_ = op_desc.GetAttr<int32_t>("size_per_head");
  param_.num_layer_ = op_desc.GetAttr<int32_t>("num_layer");
  param_.bos_id_ = op_desc.GetAttr<int32_t>("bos_id");
  param_.eos_id_ = op_desc.GetAttr<int32_t>("eos_id");
  param_.max_len_ = op_desc.GetAttr<int64_t>("max_len");
  param_.beam_search_diversity_rate_ =
      op_desc.GetAttr<float>("beam_search_diversity_rate");
  param_.unk_id_ = op_desc.GetAttr<int32_t>("unk_id");
  param_.mask_id_ = op_desc.GetAttr<int32_t>("mask_id");
  param_.temperature_ = op_desc.GetAttr<float>("temperature");
  param_.len_penalty_ = op_desc.GetAttr<float>("len_penalty");
  param_.normalize_before_ = op_desc.GetAttr<bool>("normalize_before");
  param_.pos_bias_ = op_desc.GetAttr<bool>("pos_bias");
  param_.hidden_act_ = op_desc.GetAttr<std::string>("hidden_act");
  param_.rel_len_ = op_desc.GetAttr<bool>("rel_len");
  param_.early_stopping_ = op_desc.GetAttr<bool>("early_stopping");
  param_.min_length_ = op_desc.GetAttr<int32_t>("min_length");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fusion_unified_decoding,
                 paddle::lite::operators::FusionUnifiedDecodingOp);
