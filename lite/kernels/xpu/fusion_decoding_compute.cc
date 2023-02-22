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

#include "lite/kernels/xpu/fusion_decoding_compute.h"
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void FusionDecodingCompute::PrepareForRun() {
  auto& param = Param<operators::FusionDecodingParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
#define ASSIGN_PARAM_VEC(name, type)                                          \
  name =                                                                      \
      xft::xftVec<type>(const_cast<type*>(param.name->template data<type>()), \
                        std::array<int64_t, 1>{param.name->dims()[0]});
#define ASSIGN_PARAM_MAT(name, type)                        \
  name = xft::xftMat<type>(                                 \
      const_cast<type*>(param.name->template data<type>()), \
      std::array<int64_t, 2>{param.name->dims()[0], param.name->dims()[1]});
#define QUANT_PARAM_MAT(name, type1, type2, need_transpose)                 \
  name = xft::quant_weight_from_xpu<type1, type2>(                          \
      ctx.GetRawContext(),                                                  \
      param.name->template data<type1>(),                                   \
      std::array<int64_t, 2>{param.name->dims()[0], param.name->dims()[1]}, \
      need_transpose);
#define ASSIGN_PARAM_LIST_VEC(name, type)                        \
  for (size_t i = 0; i < param.name.size(); i++) {               \
    name.emplace_back(xft::xftVec<type>(                         \
        const_cast<type*>(param.name[i]->template data<type>()), \
        std::array<int64_t, 1>{param.name[i]->dims()[0]}));      \
  }
#define QUANT_PARAM_LIST_MAT(name, type1, type2, need_transpose) \
  for (size_t i = 0; i < param.name.size(); i++) {               \
    auto tmp = xft::quant_weight_from_xpu<type1, type2>(         \
        ctx.GetRawContext(),                                     \
        param.name[i]->template data<type1>(),                   \
        std::array<int64_t, 2>{param.name[i]->dims()[0],         \
                               param.name[i]->dims()[1]},        \
        need_transpose);                                         \
    name.emplace_back(std::move(tmp));                           \
  }
#define ASSIGN_FDPARAM(name) fd_param.name = param.name;
  ASSIGN_PARAM_MAT(word_embedding, float);
  ASSIGN_PARAM_MAT(position_embedding, float);
  ASSIGN_PARAM_LIST_VEC(self_ln_weight, float);
  ASSIGN_PARAM_LIST_VEC(self_ln_bias, float);
  QUANT_PARAM_LIST_MAT(self_q_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_VEC(self_q_bias, float);
  QUANT_PARAM_LIST_MAT(self_k_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_VEC(self_k_bias, float);
  QUANT_PARAM_LIST_MAT(self_v_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_VEC(self_v_bias, float);
  QUANT_PARAM_LIST_MAT(self_out_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_VEC(self_out_bias, float);
  ASSIGN_PARAM_LIST_VEC(cross_ln_weight, float);
  ASSIGN_PARAM_LIST_VEC(cross_ln_bias, float);
  QUANT_PARAM_LIST_MAT(cross_q_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_VEC(cross_q_bias, float);
  QUANT_PARAM_LIST_MAT(cross_k_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_VEC(cross_k_bias, float);
  QUANT_PARAM_LIST_MAT(cross_v_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_VEC(cross_v_bias, float);
  QUANT_PARAM_LIST_MAT(cross_out_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_VEC(cross_out_bias, float);
  ASSIGN_PARAM_LIST_VEC(ffn_ln_weight, float);
  ASSIGN_PARAM_LIST_VEC(ffn_ln_bias, float);
  QUANT_PARAM_LIST_MAT(ffn_inter_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_VEC(ffn_inter_bias, float);
  QUANT_PARAM_LIST_MAT(ffn_out_weight, float, int16_t, true);
  ASSIGN_PARAM_LIST_VEC(ffn_out_bias, float);
  ASSIGN_PARAM_VEC(decoder_ln_weight, float);
  ASSIGN_PARAM_VEC(decoder_ln_bias, float);
  QUANT_PARAM_MAT(emb_weight, float, int16_t, true);
  ASSIGN_PARAM_VEC(emb_bias, float);
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
  // resize Input
  auto Input = xft::xftTensor<float, 3>(
      const_cast<float*>(param.Input->template data<float>()),
      {param.Input->dims()[0], param.Input->dims()[1], param.Input->dims()[2]});
  auto Memseqlen = xft::xftVec<int32_t>(
      const_cast<int32_t*>(param.Memseqlen->template data<int32_t>()),
      {param.Memseqlen->dims()[0]});
  // resize Output
  auto OutIds = xft::xftTensor<int32_t, 3>(
      param.OutIds->template mutable_data<int32_t>(TARGET(kXPU)),
      {param.OutIds->dims()[0],
       param.OutIds->dims()[1],
       param.OutIds->dims()[2]});
  auto ParentIds = xft::xftTensor<int32_t, 3>(
      param.ParentIds->template mutable_data<int32_t>(TARGET(kXPU)),
      {param.ParentIds->dims()[0],
       param.ParentIds->dims()[1],
       param.ParentIds->dims()[2]});
  auto SequenceLength = xft::xftVec<int32_t>(
      param.SequenceLength->template mutable_data<int32_t>(TARGET(kXPU)),
      {param.SequenceLength->dims()[0]});
  int r = xft::fusion_decoding<float, int16_t, int16_t>(ctx.GetRawContext(),
                                                        Input,
                                                        Memseqlen,
                                                        word_embedding,
                                                        position_embedding,
                                                        self_ln_weight,
                                                        self_ln_bias,
                                                        self_q_weight,
                                                        self_q_bias,
                                                        self_k_weight,
                                                        self_k_bias,
                                                        self_v_weight,
                                                        self_v_bias,
                                                        self_out_weight,
                                                        self_out_bias,
                                                        cross_ln_weight,
                                                        cross_ln_bias,
                                                        cross_q_weight,
                                                        cross_q_bias,
                                                        cross_k_weight,
                                                        cross_k_bias,
                                                        cross_v_weight,
                                                        cross_v_bias,
                                                        cross_out_weight,
                                                        cross_out_bias,
                                                        ffn_ln_weight,
                                                        ffn_ln_bias,
                                                        ffn_inter_weight,
                                                        ffn_inter_bias,
                                                        ffn_out_weight,
                                                        ffn_out_bias,
                                                        decoder_ln_weight,
                                                        decoder_ln_bias,
                                                        emb_weight,
                                                        emb_bias,
                                                        &OutIds,
                                                        &ParentIds,
                                                        &SequenceLength,
                                                        fd_param);
  CHECK_EQ(r, 0);
  return;
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fusion_decoding,
                     kXPU,
                     kFloat,
                     kAny,
                     paddle::lite::kernels::xpu::FusionDecodingCompute,
                     def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("MemSeqLen",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("WordEmbedding",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfLayernormWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfLayernormBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfQueryWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfQueryBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfKeyWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfKeyBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfValueWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfValueBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfOutWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfOutBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossLayernormWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossLayernormBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossQueryWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossQueryBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossKeyWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossKeyBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossValueWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossValueBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossOutWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossOutBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNLayernormWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNLayernormBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNInterWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNInterBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNOutWeight@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNOutBias@VECTOR",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("DecoderLayernormWeight",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("DecoderLayernormBias",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("EmbWeight",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("EmbBias",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("PositionEncEmb",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("OutputIds",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("ParentIds",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("SequenceLength",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();
