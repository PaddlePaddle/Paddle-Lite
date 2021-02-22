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

#include "lite/kernels/xpu/__xpu__multi_decoder_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
static std::vector<const T*> prepare_weight(
    const std::vector<lite::Tensor*>& fc_weight) {
  std::vector<const T*> result;
  for (auto* weight : fc_weight) {
    result.push_back(reinterpret_cast<const T*>(weight->data<float>()));
  }
  return result;
}

void XPUMultiDecoderCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  if (param.precision == "int16") {
    arg_fc_weight_int16_ = prepare_weight<int16_t>(param.fc_weight);
  } else if (param.precision == "int8") {
    arg_fc_weight_int8_ = prepare_weight<int8_t>(param.fc_weight);
  } else if (param.precision == "int31") {
    arg_fc_weight_fp32_ = prepare_weight<float>(param.fc_weight);
  }
  for (auto* fc_bias : param.fc_bias) {
    arg_fc_bias_.push_back(fc_bias->data<float>());
  }
  for (auto* ln_scale : param.ln_scale) {
    arg_ln_scale_.push_back(ln_scale->data<float>());
  }
  for (auto* ln_bias : param.ln_bias) {
    arg_ln_bias_.push_back(ln_bias->data<float>());
  }

  decoder_param_.head_num = param.head_num;
  decoder_param_.size_per_head = param.size_per_head;
  decoder_param_.n_layers = param.n_layers;
  decoder_param_.pretrans_b = true;
  decoder_param_.use_l3 = true;
  if (param.act_type == "relu") {
    decoder_param_.act_type = xdnn::Activation_t::RELU;
  } else if (param.act_type == "gelu") {
    decoder_param_.act_type = xdnn::Activation_t::GELU;
  }
}

void XPUMultiDecoderCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  decoder_param_.batch_size = param.input->dims()[0];
  decoder_param_.dec_seq_len = param.input->dims()[1];
  decoder_param_.cache_seq_len = param.k_cache_in[0]->dims()[2];
  decoder_param_.enc_seq_len = param.k_cache_in[1]->dims()[2];
  std::vector<int64_t> mask_shape = param.mask->dims().Vectorize();
  decoder_param_.mask_shape =
      std::vector<int>(mask_shape.begin(), mask_shape.end());

  for (auto* k_cache : param.k_cache_in) {
    arg_k_cache_in_.push_back(k_cache->data<float>());
  }
  for (auto* v_cache : param.v_cache_in) {
    arg_v_cache_in_.push_back(v_cache->data<float>());
  }
  for (auto* k_cache : param.k_cache_out) {
    arg_k_cache_out_.push_back(k_cache->mutable_data<float>(TARGET(kXPU)));
  }
  for (auto* v_cache : param.v_cache_out) {
    arg_v_cache_out_.push_back(v_cache->mutable_data<float>(TARGET(kXPU)));
  }

  int r = -1;

  ctx.GetRawContext()->qkv_fusion = param.enable_qkv_fusion;
  if (param.precision == "int16") {
    r = xdnn::transformer_decoder_int16<float, int16_t, float>(
        ctx.GetRawContext(),                             /* context */
        param.input->data<float>(),                      /* dec_input */
        param.mask->data<float>(),                       /* attn_mask */
        param.output->mutable_data<float>(TARGET(kXPU)), /* output */
        arg_k_cache_in_,                                 /* caches_k_in */
        arg_v_cache_in_,                                 /* caches_v_in */
        arg_k_cache_out_,                                /* caches_k_out */
        arg_v_cache_out_,                                /* caches_v_out */
        arg_fc_weight_int16_,                            /* fc_weights */
        arg_fc_bias_,                                    /* fc_biass */
        arg_ln_scale_,                                   /* ln_scales */
        arg_ln_bias_,                                    /* ln_biass */
        param.fc_weight_max->data<float>(),              /* fc_weights_max */
        decoder_param_ /* param */);
  }
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__multi_decoder,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUMultiDecoderCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("KCache", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("VCache", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCWeightMax", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("KCacheOutputs", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("VCacheOutputs", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
