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

#include "lite/kernels/xpu/__xpu__multi_encoder_compute.h"

namespace paddle {
namespace lite_metal {
namespace kernels {
namespace xpu {

template <typename T>
static std::vector<const T*> prepare_weight(
    const std::vector<lite_metal::Tensor*>& fc_weight) {
  std::vector<const T*> result;
  for (auto* weight : fc_weight) {
    result.push_back(reinterpret_cast<const T*>(weight->data<float>()));
  }
  return result;
}

void XPUMultiEncoderCompute::PrepareForRun() {
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

  encoder_param_.head_num = param.head_num;
  encoder_param_.size_per_head = param.size_per_head;
  encoder_param_.n_layers = param.n_layers;
  encoder_param_.pretrans_b = true;
  encoder_param_.use_l3 = true;
  encoder_param_.slice_starts = param.slice_starts;
  encoder_param_.slice_ends = param.slice_ends;
  encoder_param_.slice_axes = param.slice_axes;
  if (param.act_type == "relu") {
    encoder_param_.act_type = xdnn::Activation_t::RELU;
  } else if (param.act_type == "gelu") {
    encoder_param_.act_type = xdnn::Activation_t::GELU;
  }
}

int XPUMultiEncoderCompute::bert_encoder_run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  ctx.GetRawContext()->qkv_fusion = param.enable_qkv_fusion;

  int r = -1;
  if (param.precision == "int31") {
    r = xdnn::bert_encoder_transformer_int31(
        ctx.GetRawContext(),                             /* context */
        param.input->data<float>(),                      /* from_tensor */
        param.input->data<float>(),                      /* to_tensor */
        param.mask->data<float>(),                       /* att_mask */
        param.output->mutable_data<float>(TARGET(kXPU)), /* output */
        arg_fc_weight_fp32_,                             /* fc_weights */
        arg_fc_bias_,                                    /* fc_biass */
        arg_ln_scale_,                                   /* ln_scales */
        arg_ln_bias_,                                    /* ln_biass */
        param.fc_weight_max->data<float>(),              /* fc_weights_max */
        encoder_param_);
  } else if (param.precision == "int8") {
    r = xdnn::bert_encoder_transformer_int8<float, int8_t, float>(
        ctx.GetRawContext(),                             /* context */
        param.input->data<float>(),                      /* from_tensor */
        param.input->data<float>(),                      /* to_tensor */
        param.mask->data<float>(),                       /* att_mask */
        param.output->mutable_data<float>(TARGET(kXPU)), /* output */
        arg_fc_weight_int8_,                             /* fc_weights */
        arg_fc_bias_,                                    /* fc_biass */
        arg_ln_scale_,                                   /* ln_scales */
        arg_ln_bias_,                                    /* ln_biass */
        param.fc_weight_max->data<float>(),              /* fc_weights_max */
        encoder_param_);
  } else {
    r = xdnn::bert_encoder_transformer_int16<float, int16_t, float>(
        ctx.GetRawContext(),                             /* context */
        param.input->data<float>(),                      /* from_tensor */
        param.input->data<float>(),                      /* to_tensor */
        param.mask->data<float>(),                       /* att_mask */
        param.output->mutable_data<float>(TARGET(kXPU)), /* output */
        arg_fc_weight_int16_,                            /* fc_weights */
        arg_fc_bias_,                                    /* fc_biass */
        arg_ln_scale_,                                   /* ln_scales */
        arg_ln_bias_,                                    /* ln_biass */
        param.fc_weight_max->data<float>(),              /* fc_weights_max */
        encoder_param_);
  }
  return r;
}

int XPUMultiEncoderCompute::transformer_encoder_run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  ctx.GetRawContext()->qkv_fusion = param.enable_qkv_fusion;

  int r = -1;
  if (param.precision == "int31") {
    LOG(FATAL) << "Not support int31 at now";
  } else if (param.precision == "int8") {
    LOG(FATAL) << "Not support int8 at now";
  } else {
    r = xdnn::transformer_encoder_int16<float, int16_t, float>(
        ctx.GetRawContext(),                             /* context */
        param.input->data<float>(),                      /* from_tensor */
        param.input->data<float>(),                      /* to_tensor */
        param.mask->data<float>(),                       /* att_mask */
        param.output->mutable_data<float>(TARGET(kXPU)), /* output */
        arg_fc_weight_int16_,                            /* fc_weights */
        arg_fc_bias_,                                    /* fc_biass */
        arg_ln_scale_,                                   /* ln_scales */
        arg_ln_bias_,                                    /* ln_biass */
        param.fc_weight_max->data<float>(),              /* fc_weights_max */
        encoder_param_);
  }
  return r;
}

void XPUMultiEncoderCompute::Run() {
  auto& param = this->Param<param_t>();
  std::vector<int64_t> mask_shape = param.mask->dims().Vectorize();
  encoder_param_.mask_shape =
      std::vector<int>(mask_shape.begin(), mask_shape.end());
  encoder_param_.slice_starts = param.slice_starts;
  encoder_param_.slice_ends = param.slice_ends;
  encoder_param_.slice_axes = param.slice_axes;
  const bool norm_before_ = param.norm_before;
  if (param.SeqLod && param.SeqLod->data<int>()) {
    auto& ctx = this->ctx_->As<XPUContext>();
    ctx.GetRawContext()->batch_split_type = -1;  // disable auto split batch
    encoder_param_.seq_lod.resize(param.SeqLod->numel());
    memcpy(encoder_param_.seq_lod.data(),
           param.SeqLod->data<int>(),
           sizeof(int) * param.SeqLod->numel());
    encoder_param_.adaptive_seqlen = true;
    encoder_param_.batch_size = param.SeqLod->numel() - 1;
    encoder_param_.from_seq_len = param.PadSeqLen->data<int>()[0];
    encoder_param_.to_seq_len = param.PadSeqLen->data<int>()[0];
  } else {
    encoder_param_.adaptive_seqlen = false;
    encoder_param_.batch_size = param.input->dims()[0];
    encoder_param_.from_seq_len = param.input->dims()[1];
    encoder_param_.to_seq_len = param.input->dims()[1];
  }
  int r = -1;
  if (norm_before_) {
    r = transformer_encoder_run();
  } else {
    r = bert_encoder_run();
  }
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__multi_encoder,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite_metal::kernels::xpu::XPUMultiEncoderCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("SeqLod",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("PadSeqLen",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCWeightMax", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
