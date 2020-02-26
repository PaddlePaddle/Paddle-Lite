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

#include "lite/kernels/xpu/multi_encoder_compute.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {

namespace operators {

bool MultiEncoderOp::CheckShape() const { return true; }

bool MultiEncoderOp::InferShape() const {
  auto input_shape = param_.input->dims();
  param_.output->Resize(input_shape);
  return true;
}

bool MultiEncoderOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.input = const_cast<lite::Tensor *>(
      &scope->FindVar(op_desc.Input("Input").front())->Get<lite::Tensor>());
  param_.mask = const_cast<lite::Tensor *>(
      &scope->FindVar(op_desc.Input("Mask").front())->Get<lite::Tensor>());
  param_.fc_weight_max = const_cast<lite::Tensor *>(
      &scope->FindVar(op_desc.Input("FCWeightMax").front())->Get<lite::Tensor>());
  param_.output =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<lite::Tensor>();

  param_.fc_weight.clear();
  for (auto& name : op_desc.Input("FCWeight")) {
    auto t = const_cast<lite::Tensor *>(
        &scope->FindVar(name)->Get<lite::Tensor>());
    param_.fc_weight.push_back(t);
  }
  param_.fc_bias.clear();
  for (auto& name : op_desc.Input("FCBias")) {
    auto t = const_cast<lite::Tensor *>(
        &scope->FindVar(name)->Get<lite::Tensor>());
    param_.fc_bias.push_back(t);
  }
  param_.ln_scale.clear();
  for (auto& name : op_desc.Input("LNScale")) {
    auto t = const_cast<lite::Tensor *>(
        &scope->FindVar(name)->Get<lite::Tensor>());
    param_.ln_scale.push_back(t);
  }
  param_.ln_bias.clear();
  for (auto& name : op_desc.Input("LNBias")) {
    auto t = const_cast<lite::Tensor *>(
        &scope->FindVar(name)->Get<lite::Tensor>());
    param_.ln_bias.push_back(t);
  }

  param_.n_layers = op_desc.GetAttr<int>("n_layers");
  param_.head_num = op_desc.GetAttr<int>("head_num");
  param_.size_per_head = op_desc.GetAttr<int>("size_per_head");
  param_.act_type = op_desc.GetAttr<std::string>("act_type");
  return true;
}

}  // namespace operators

namespace kernels {
namespace xpu {

MultiEncoderCompute::MultiEncoderCompute() {
  set_op_type("MultiEncoder");
  set_alias("def");
}

void MultiEncoderCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  for (auto* fc_weight : param.fc_weight) {
    arg_fc_weight_.push_back((int16_t*)fc_weight->data<float>());
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

void MultiEncoderCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int batch_size = param.input->dims()[0];
  int seq_len = param.input->dims()[1];
  int r = xdnn::bert_encoder_transformer_int16<int16_t>(
      ctx.GetRawContext(), /* context */
      batch_size, /* batch_size */
      seq_len, /* from_seq_len */
      seq_len, /* to_seq_len */
      param.head_num, /* head_num */
      param.size_per_head, /* size_per_head */
      param.n_layers, /* n_layers */
      param.input->data<float>(), /* from_tensor */
      param.input->data<float>(), /* to_tensor */
      param.mask->data<float>(), /* att_mask */
      &arg_fc_weight_[0], /* fc_weights */
      &arg_fc_bias_[0], /* fc_biass */
      &arg_ln_scale_[0], /* ln_scales */
      &arg_ln_bias_[0], /* ln_biass */
      param.output->mutable_data<float>(TARGET(kXPU)), /* output */
      param.fc_weight_max->data<float>(), /* fc_weights_max */
      true, /* pretrans_b */
      true, /* use_l3 */
      act_type_ /* act_type */);
  CHECK(r == 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(MultiEncoder, paddle::lite::operators::MultiEncoderOp);

REGISTER_LITE_KERNEL(MultiEncoder,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::MultiEncoderCompute,
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
