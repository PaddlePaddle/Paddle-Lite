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

#include "lite/kernels/xpu/fusion_decoding_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void FusionDecodingCompute::PrepareForRun() {
  auto& param = Param<operators::FusionDecodingParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  #define ASSIGN_PARAM_1(name, type) name = xft::xftTensor<type, 1>(const_cast<type*>(param.name->template data<type>()), std::array<int64_t, 1>{param.name->dims()[0]});
  #define ASSIGN_PARAM_2(name, type) name = xft::xftTensor<type, 2>(const_cast<type*>(param.name->template data<type>()), std::array<int64_t, 2>{param.name->dims()[0], param.name->dims()[1]});

  #define QUANT_PARAM_2(name, type1, type2, need_transpose)  \
  name = xft::quant_weight_from_xpu<type1, type2>(ctx.GetRawContext(), \
            param.name->template data<type1>(), std::array<int64_t, 2>{param.name->dims()[0], param.name->dims()[1]}, need_transpose);

  #define ASSIGN_PARAM_LIST_1(name, type) \
  for (auto i = 0; i < param.name.size(); i++) { \
      name.emplace_back(xft::xftTensor<type, 1>(const_cast<type*>(param.name[i]->template data<type>()), std::array<int64_t, 1>{param.name[i]->dims()[0]})); \
  }

  #define QUANT_PARAM_LIST_2(name, type1, type2, need_transpose) \
  for (auto i = 0; i < param.name.size(); i++) { \
      auto tmp = xft::quant_weight_from_xpu<type1, type2>(ctx.GetRawContext(), \
            param.name[i]->template data<type1>(), std::array<int64_t, 2>{param.name[i]->dims()[0], param.name[i]->dims()[1]}, need_transpose); \
      name.emplace_back(std::move(tmp)); \
  }

  #define ASSIGN_FDPARAM(name) fd_param.name = param.name;

  ASSIGN_PARAM_2(word_embedding, float);
  ASSIGN_PARAM_2(position_embedding, float);
  ASSIGN_PARAM_LIST_1(self_ln_weight, float);
  ASSIGN_PARAM_LIST_1(self_ln_bias, float);
  QUANT_PARAM_LIST_2(self_q_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_1(self_q_bias, float);
  QUANT_PARAM_LIST_2(self_k_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_1(self_k_bias, float);
  QUANT_PARAM_LIST_2(self_v_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_1(self_v_bias, float);
  QUANT_PARAM_LIST_2(self_out_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_1(self_out_bias, float);
  ASSIGN_PARAM_LIST_1(cross_ln_weight, float);
  ASSIGN_PARAM_LIST_1(cross_ln_bias, float);
  QUANT_PARAM_LIST_2(cross_q_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_1(cross_q_bias, float);
  QUANT_PARAM_LIST_2(cross_k_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_1(cross_k_bias, float);
  QUANT_PARAM_LIST_2(cross_v_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_1(cross_v_bias, float);
  QUANT_PARAM_LIST_2(cross_out_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_1(cross_out_bias, float);
  ASSIGN_PARAM_LIST_1(ffn_ln_weight, float);
  ASSIGN_PARAM_LIST_1(ffn_ln_bias, float);
  QUANT_PARAM_LIST_2(ffn_inter_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_1(ffn_inter_bias, float);
  QUANT_PARAM_LIST_2(ffn_out_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_1(ffn_out_bias, float);
  ASSIGN_PARAM_1(decoder_ln_weight, float);
  ASSIGN_PARAM_1(decoder_ln_bias, float);
  QUANT_PARAM_2(emb_weight, float, int16_t, true);
  ASSIGN_PARAM_1(emb_bias, float);

  ASSIGN_FDPARAM(alpha);
  ASSIGN_FDPARAM(beam_search_diversity_rate);
  ASSIGN_FDPARAM(beam_size);
  ASSIGN_FDPARAM(bos_id);
  ASSIGN_FDPARAM(decoding_strategy);
  ASSIGN_FDPARAM(eos_id);
  ASSIGN_FDPARAM(max_len);
  ASSIGN_FDPARAM(num_layer);
  ASSIGN_FDPARAM(n_head);
  ASSIGN_FDPARAM(rel_len);
  ASSIGN_FDPARAM(size_per_head);
  ASSIGN_FDPARAM(topk);
  ASSIGN_FDPARAM(topp);
  return;
}

void FusionDecodingCompute::Run() {
  auto& param = Param<operators::FusionDecodingParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto Input = xft::xftTensor<float, 3>(const_cast<float*>(param.Input->template data<float>()), std::array<int64_t, 3>{param.Input->dims()[0], param.Input->dims()[1], param.Input->dims()[2]});
  auto Memseqlen = xft::xftTensor<int32_t, 1>(const_cast<int32_t*>(param.Memseqlen->template data<int32_t>()), std::array<int64_t, 1>{param.Memseqlen->dims()[0]});
  auto OutIds = xft::xftTensor<int32_t, 3>(param.OutIds->template mutable_data<int32_t>(TARGET(kXPU)), std::array<int64_t, 3>{param.OutIds->dims()[0], param.OutIds->dims()[1], param.OutIds->dims()[2]});
  auto ParentIds = xft::xftTensor<int32_t, 3>(param.ParentIds->template mutable_data<int32_t>(TARGET(kXPU)), std::array<int64_t, 3>{param.ParentIds->dims()[0], param.ParentIds->dims()[1], param.ParentIds->dims()[2]});
  auto SequenceLength = xft::xftTensor<int32_t, 1>(param.SequenceLength->template mutable_data<int32_t>(TARGET(kXPU)), std::array<int64_t, 1>{param.SequenceLength->dims()[0]});
  int r = xft::fusion_decoding<float, int16_t, int16_t>(
            ctx.GetRawContext(), 
            Input, Memseqlen,
            word_embedding, position_embedding,
            self_ln_weight, self_ln_bias,
            self_q_weight, self_q_bias,
            self_k_weight, self_k_bias,
            self_v_weight, self_v_bias,
            self_out_weight, self_out_bias,
            cross_ln_weight, cross_ln_bias,
            cross_q_weight, cross_q_bias,
            cross_k_weight, cross_k_bias,
            cross_v_weight, cross_v_bias,
            cross_out_weight, cross_out_bias,
            ffn_ln_weight, ffn_ln_bias,
            ffn_inter_weight, ffn_inter_bias,
            ffn_out_weight, ffn_out_bias,
            decoder_ln_weight, decoder_ln_bias,
            emb_weight, emb_bias,
            &OutIds, &ParentIds, &SequenceLength, 
            fd_param);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fusion_decoding,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::FusionDecodingCompute,
                     def)
    .BindInput("Input",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("MemSeqLen",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("WordEmbedding",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("SelfLayernormWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("SelfLayernormBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("SelfQueryWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("SelfQueryBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("SelfKeyWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("SelfKeyBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("SelfValueWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("SelfValueBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("SelfOutWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("SelfOutBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("CrossLayernormWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("CrossLayernormBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("CrossQueryWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("CrossQueryBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("CrossKeyWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("CrossKeyBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("CrossValueWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("CrossValueBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("CrossOutWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("CrossOutBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("FFNLayernormWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("FFNLayernormBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("FFNInterWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("FFNInterBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("FFNOutWeight@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("FFNOutBias@VECTOR",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("DecoderLayernormWeight",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("DecoderLayernormBias",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("EmbWeight",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("EmbBias",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("PositionEncEmb",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindOutput("OutputIds",
                {LiteType::GetTensorTy(
                    TARGET(kXPU), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("ParentIds",
                {LiteType::GetTensorTy(
                    TARGET(kXPU), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("SequenceLength",
                {LiteType::GetTensorTy(
                    TARGET(kXPU), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .Finalize();
