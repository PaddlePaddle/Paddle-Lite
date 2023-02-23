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

#include "lite/kernels/xpu/fusion_unified_decoding_compute.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

using std::cout;
using std::endl;
using std::vector;

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void FusionUnifiedDecodingCompute::PrepareForRun() {
  auto ctx = this->ctx_->As<XPUContext>().GetRawContext();
  auto& param = this->Param<param_t>();
  VLOG(2) << "Init xft config for fusion_unified_decoding";
  fud_param_.beam_search_diversity_rate = param.beam_search_diversity_rate_;
  fud_param_.beam_size = param.beam_size_;
  fud_param_.bos_id = param.bos_id_;
  fud_param_.decoding_strategy = param.decoding_strategy_;
  fud_param_.early_stopping = param.early_stopping_;
  fud_param_.eos_id = param.eos_id_;
  fud_param_.hidden_act = param.hidden_act_;
  fud_param_.len_penalty = param.len_penalty_;
  fud_param_.mask_id = param.mask_id_;
  fud_param_.max_len = param.max_len_;
  fud_param_.min_len = param.min_length_;
  fud_param_.normalize_before = param.normalize_before_;
  fud_param_.num_layer = param.num_layer_;
  fud_param_.n_head = param.n_head_;
  fud_param_.pos_bias = param.pos_bias_;
  fud_param_.rel_len = param.rel_len_;
  fud_param_.size_per_head = param.size_per_head_;
  fud_param_.temperature = param.temperature_;
  fud_param_.topk = param.topk_;
  fud_param_.topp = param.topp_;
  fud_param_.unk_id = param.unk_id_;
  word_embedding_ = xft::xftMat<float>(
      const_cast<float*>(param.word_embedding_->data<float>()),
      {param.word_embedding_->dims()[0], param.word_embedding_->dims()[1]});
  positional_embedding_ = xft::xftMat<float>(
      const_cast<float*>(param.positional_embedding_weight_->data<float>()),
      {param.positional_embedding_weight_->dims()[0],
       param.positional_embedding_weight_->dims()[1]});
  type_embedding_ = xft::xftMat<float>(
      const_cast<float*>(param.type_embedding_weight_->data<float>()),
      {param.type_embedding_weight_->dims()[0],
       param.type_embedding_weight_->dims()[1]});
  self_ln_weight_.reserve(fud_param_.num_layer);
  self_ln_bias_.reserve(fud_param_.num_layer);
  self_q_weight_.reserve(fud_param_.num_layer);
  self_q_bias_.reserve(fud_param_.num_layer);
  self_out_weight_.reserve(fud_param_.num_layer);
  self_out_bias_.reserve(fud_param_.num_layer);
  ffn_ln_weight_.reserve(fud_param_.num_layer);
  ffn_ln_bias_.reserve(fud_param_.num_layer);
  ffn_inter_weight_.reserve(fud_param_.num_layer);
  ffn_inter_bias_.reserve(fud_param_.num_layer);
  ffn_out_weight_.reserve(fud_param_.num_layer);
  ffn_out_bias_.reserve(fud_param_.num_layer);
  for (int32_t layer = 0; layer < fud_param_.num_layer; layer++) {
    self_ln_weight_.emplace_back(
        const_cast<float*>(param.self_ln_weight_[layer]->data<float>()),
        xft::xftVec<float>::dim_t{param.self_ln_weight_[layer]->dims()[0]});
    self_ln_bias_.emplace_back(
        const_cast<float*>(param.self_ln_bias_[layer]->data<float>()),
        xft::xftVec<float>::dim_t{param.self_ln_bias_[layer]->dims()[0]});
    self_q_weight_.push_back(xft::quant_weight_from_host<float, int16_t>(
        ctx,
        const_cast<float*>(param.self_q_weight_[layer]->data<float>()),
        {param.self_q_weight_[layer]->dims()[0],
         param.self_q_weight_[layer]->dims()[1]},
        true));
    self_q_bias_.emplace_back(
        const_cast<float*>(param.self_q_bias_[layer]->data<float>()),
        xft::xftVec<float>::dim_t{param.self_q_bias_[layer]->dims()[0]});
    self_out_weight_.push_back(xft::quant_weight_from_host<float, int16_t>(
        ctx,
        const_cast<float*>(param.self_out_weight_[layer]->data<float>()),
        {param.self_out_weight_[layer]->dims()[0],
         param.self_out_weight_[layer]->dims()[1]},
        true));
    self_out_bias_.emplace_back(
        const_cast<float*>(param.self_out_bias_[layer]->data<float>()),
        xft::xftVec<float>::dim_t{param.self_out_bias_[layer]->dims()[0]});
    ffn_ln_weight_.emplace_back(
        const_cast<float*>(param.ffn_ln_weight_[layer]->data<float>()),
        xft::xftVec<float>::dim_t{param.ffn_ln_weight_[layer]->dims()[0]});
    ffn_ln_bias_.emplace_back(
        const_cast<float*>(param.ffn_ln_bias_[layer]->data<float>()),
        xft::xftVec<float>::dim_t{param.ffn_ln_bias_[layer]->dims()[0]});
    ffn_inter_weight_.push_back(xft::quant_weight_from_host<float, int16_t>(
        ctx,
        const_cast<float*>(param.ffn_inter_weight_[layer]->data<float>()),
        {param.ffn_inter_weight_[layer]->dims()[0],
         param.ffn_inter_weight_[layer]->dims()[1]},
        true));
    ffn_inter_bias_.emplace_back(
        const_cast<float*>(param.ffn_inter_bias_[layer]->data<float>()),
        xft::xftVec<float>::dim_t{param.ffn_inter_bias_[layer]->dims()[0]});
    ffn_out_weight_.push_back(xft::quant_weight_from_host<float, int16_t>(
        ctx,
        const_cast<float*>(param.ffn_out_weight_[layer]->data<float>()),
        {param.ffn_out_weight_[layer]->dims()[0],
         param.ffn_out_weight_[layer]->dims()[1]},
        true));
    ffn_out_bias_.emplace_back(
        const_cast<float*>(param.ffn_out_bias_[layer]->data<float>()),
        xft::xftVec<float>::dim_t{param.ffn_out_bias_[layer]->dims()[0]});
  }
  decoder_ln_weight_ = xft::xftVec<float>(
      const_cast<float*>(param.decoder_ln_weight_->data<float>()),
      {param.decoder_ln_weight_->dims()[0]});
  decoder_ln_bias_ = xft::xftVec<float>(
      const_cast<float*>(param.decoder_ln_bias_->data<float>()),
      {param.decoder_ln_bias_->dims()[0]});
  trans_weight_ = xft::quant_weight_from_host<float, int16_t>(
      ctx,
      const_cast<float*>(param.trans_weight_->data<float>()),
      {param.trans_weight_->dims()[0], param.trans_weight_->dims()[1]},
      true);
  trans_bias_ =
      xft::xftVec<float>(const_cast<float*>(param.trans_bias_->data<float>()),
                         {param.trans_bias_->dims()[0]});
  lm_ln_weight_ =
      xft::xftVec<float>(const_cast<float*>(param.lm_ln_weight_->data<float>()),
                         {param.lm_ln_weight_->dims()[0]});
  lm_ln_bias_ =
      xft::xftVec<float>(const_cast<float*>(param.lm_ln_bias_->data<float>()),
                         {param.lm_ln_bias_->dims()[0]});
  emb_weight_ = xft::quant_weight_from_host<float, int16_t>(
      ctx,
      const_cast<float*>(param.embedding_weight_->data<float>()),
      {param.embedding_weight_->dims()[0], param.embedding_weight_->dims()[1]},
      true);
  emb_bias_ = xft::xftVec<float>(
      const_cast<float*>(param.embedding_bias_->data<float>()),
      {param.embedding_bias_->dims()[0]});
  VLOG(2) << "End xft config for fusion_unified_decoding";
  return;
}

void FusionUnifiedDecodingCompute::Run() {
  // this->RunDecodingForward();
  auto& param = this->Param<param_t>();
  auto ctx = this->ctx_->As<XPUContext>().GetRawContext();
  const int32_t batch_size = param.input_ids_->dims()[0];
  const int32_t max_out_len = param.rel_len_
                                  ? param.max_len_ + param.input_ids_->dims()[1]
                                  : param.max_len_;
  CHECK(param.decoding_strategy_ == "topk_sampling")
      << "Only 'topk_sampling' is supported now, your strategy is "
      << param.decoding_strategy_;
  param.output_ids_->Resize({max_out_len, batch_size});
  param.parent_ids_->Resize({1});
  param.output_scores_->Resize({batch_size});
  param.sequence_length_->Resize({batch_size});
  // resize input
  xft::xftMat<int32_t> input_ids(
      const_cast<int32_t*>(param.input_ids_->data<int32_t>()),
      {param.input_ids_->dims()[0], param.input_ids_->dims()[1]});
  xft::xftVec<int32_t> mem_seq_len(
      const_cast<int32_t*>(param.mem_seq_len_->data<int32_t>()),
      {param.mem_seq_len_->dims()[0]});
  xft::xftTensor<float, 4> attn_mask(
      const_cast<float*>(param.attn_mask_->data<float>()),
      {param.attn_mask_->dims()[0],
       param.attn_mask_->dims()[1],
       param.attn_mask_->dims()[2],
       param.attn_mask_->dims()[3]});
  xft::xftMat<int32_t> pos_ids(
      const_cast<int32_t*>(param.position_ids_->data<int32_t>()),
      {param.position_ids_->dims()[0], param.position_ids_->dims()[1]});
  xft::xftMat<int32_t> type_ids(
      const_cast<int32_t*>(param.type_id_->data<int32_t>()),
      {param.type_id_->dims()[0], param.type_id_->dims()[1]});
  xft::xftMat<int32_t> dec_pos_ids(
      const_cast<int32_t*>(param.decoder_position_ids_->data<int32_t>()),
      {param.decoder_position_ids_->dims()[0],
       param.decoder_position_ids_->dims()[1]});
  xft::xftMat<int32_t> dec_type_ids(
      const_cast<int32_t*>(param.decoder_type_id_->data<int32_t>()),
      {param.decoder_type_id_->dims()[0], param.decoder_type_id_->dims()[1]});
  // resize output
  xft::xftMat<int32_t> output_ids(
      param.output_ids_->mutable_data<int32_t>(TARGET(kXPU)),
      {max_out_len, batch_size});
  xft::xftVec<int32_t> output_seq_len(
      param.sequence_length_->mutable_data<int32_t>(TARGET(kXPU)),
      {batch_size});
  xft::xftVec<float> output_scores(
      param.output_scores_->mutable_data<float>(TARGET(kXPU)), {batch_size});

  int32_t ret = xft::fusion_unified_decoding<float, int16_t, int16_t>(
      ctx,
      input_ids,
      mem_seq_len,
      attn_mask,
      pos_ids,
      type_ids,
      dec_pos_ids,
      dec_type_ids,
      word_embedding_,
      positional_embedding_,
      type_embedding_,
      self_ln_weight_,
      self_ln_bias_,
      self_q_weight_,
      self_q_bias_,
      self_k_weight_,
      self_k_bias_,
      self_v_weight_,
      self_v_bias_,
      self_out_weight_,
      self_out_bias_,
      ffn_ln_weight_,
      ffn_ln_bias_,
      ffn_inter_weight_,
      ffn_inter_bias_,
      ffn_out_weight_,
      ffn_out_bias_,
      decoder_ln_weight_,
      decoder_ln_bias_,
      trans_weight_,
      trans_bias_,
      lm_ln_weight_,
      lm_ln_bias_,
      emb_weight_,
      emb_bias_,
      &output_ids,
      &output_seq_len,
      &output_scores,
      fud_param_);
  CHECK_EQ(ret, 0);
  return;
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fusion_unified_decoding,
                     kXPU,
                     kFloat,
                     kAny,
                     paddle::lite::kernels::xpu::FusionUnifiedDecodingCompute,
                     def)
    .BindInput("DecPositionIds",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("AttnMask",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("DecRoleIds",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("DecTypeIds",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("DecoderLayernormBias",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("DecoderLayernormWeight",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("EmbBias",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("EmbWeight",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("FFNInterBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNInterWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("FFNLayernormBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNLayernormWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNOutBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNOutWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("InputIds",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("LMLayernormBias",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("LMLayernormWeight",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("LogitsMask",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("MemSeqLen",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("PositionEncEmb",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("PositionIds",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("RoleEmbedding",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("RoleIds",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("SelfKeyBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfKeyWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("SelfLayernormBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfLayernormWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfOutBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfOutWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("SelfQueryBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfQueryWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("SelfValueBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfValueWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("TransBias",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("TransWeight",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("TypeEmb",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("TypeIds",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("WordEmbedding",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("OutputIds",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("OutputScores",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("ParentIds",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("SequenceLength",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();
