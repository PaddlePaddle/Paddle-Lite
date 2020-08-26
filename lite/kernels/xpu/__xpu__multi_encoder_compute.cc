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
namespace lite {
namespace kernels {
namespace xpu {

void XPUMultiEncoderCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  for (auto* fc_weight : param.fc_weight) {
    arg_fc_weight_.push_back(
        reinterpret_cast<const int16_t*>(fc_weight->data<float>()));
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
  if (param.act_type == "relu") {
    act_type_ = xdnn::Activation_t::RELU;
  }
}

void XPUMultiEncoderCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int batch_size = param.input->dims()[0];
  int seq_len = param.input->dims()[1];
  int r = -1;
  if (param.precision == "int31") {
    ctx.GetRawContext()->qkv_fusion = param.enable_qkv_fusion;
    r = xdnn::bert_encoder_transformer_int31(
        ctx.GetRawContext(),                             /* context */
        batch_size,                                      /* batch_size */
        seq_len,                                         /* from_seq_len */
        seq_len,                                         /* to_seq_len */
        param.head_num,                                  /* head_num */
        param.size_per_head,                             /* size_per_head */
        param.n_layers,                                  /* n_layers */
        param.input->data<float>(),                      /* from_tensor */
        param.input->data<float>(),                      /* to_tensor */
        param.mask->data<float>(),                       /* att_mask */
        (const float**)(&arg_fc_weight_[0]),             /* fc_weights */
        &arg_fc_bias_[0],                                /* fc_biass */
        &arg_ln_scale_[0],                               /* ln_scales */
        &arg_ln_bias_[0],                                /* ln_biass */
        param.output->mutable_data<float>(TARGET(kXPU)), /* output */
        param.fc_weight_max->data<float>(),              /* fc_weights_max */
        true,                                            /* pretrans_b */
        true,                                            /* use_l3 */
        act_type_ /* act_type */);
  } else {
    r = xdnn::bert_encoder_transformer_int16<int16_t>(
        ctx.GetRawContext(),                             /* context */
        batch_size,                                      /* batch_size */
        seq_len,                                         /* from_seq_len */
        seq_len,                                         /* to_seq_len */
        param.head_num,                                  /* head_num */
        param.size_per_head,                             /* size_per_head */
        param.n_layers,                                  /* n_layers */
        param.input->data<float>(),                      /* from_tensor */
        param.input->data<float>(),                      /* to_tensor */
        param.mask->data<float>(),                       /* att_mask */
        &arg_fc_weight_[0],                              /* fc_weights */
        &arg_fc_bias_[0],                                /* fc_biass */
        &arg_ln_scale_[0],                               /* ln_scales */
        &arg_ln_bias_[0],                                /* ln_biass */
        param.output->mutable_data<float>(TARGET(kXPU)), /* output */
        param.fc_weight_max->data<float>(),              /* fc_weights_max */
        true,                                            /* pretrans_b */
        true,                                            /* use_l3 */
        act_type_ /* act_type */);
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
                     paddle::lite::kernels::xpu::XPUMultiEncoderCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCWeightMax", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
