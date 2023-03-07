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

#include "lite/kernels/xpu/__xpu__spatial_transformer_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {
template <typename T>

static std::vector<const T*> PrepareWeight(
    const std::vector<lite::Tensor*>& fc_weight) {
  std::vector<const T*> result;
  for (auto* weight : fc_weight) {
    result.push_back(reinterpret_cast<const T*>(weight->data<float>()));
  }
  return result;
}

template <typename InType, PrecisionType PType>
void XPUSpatialTransformerCompute<InType, PType>::PrepareWeightMax(
    const std::vector<lite::Tensor*>& weight_max,
    int max_ptr_len,
    std::vector<const float*>* max_xpu_ptrs) {
  int max_value_num = 0;
  for (auto max_tensor : weight_max) {
    max_value_num += max_tensor->numel();
  }
  VLOG(3) << "Total weight max value number: " << max_value_num;
  weight_max_guard_ =
      TargetWrapperXPU::MallocScratchPad(max_value_num * sizeof(float));
  float* weight_max_ptr = reinterpret_cast<float*>(weight_max_guard_->addr_);

  int offset = 0;
  for (auto max_tensor : weight_max) {
    float* cur_weight_max_ptr = weight_max_ptr + offset;
    auto len = max_tensor->numel();
    VLOG(6) << "weight max value: " << max_tensor->data<float>()[0] << " "
            << max_tensor->data<float>()[len - 1];
    std::vector<float> cpu_max(max_ptr_len, max_tensor->data<float>()[0]);
    lite::TargetWrapperXPU::MemcpySync(cur_weight_max_ptr,
                                       cpu_max.data(),
                                       sizeof(float) * max_ptr_len,
                                       IoDirection::HtoD);
    max_xpu_ptrs->push_back(cur_weight_max_ptr);
    offset += max_ptr_len;
  }
}

template <typename InType, PrecisionType PType>
void XPUSpatialTransformerCompute<InType, PType>::PrepareFilterMax(
    const std::vector<lite::Tensor*>& filter_max,
    int max_ptr_len,
    std::vector<const float*>* max_xpu_ptrs) {
  int max_value_num = 0;
  for (auto max_tensor : filter_max) {
    max_value_num += max_tensor->numel();
  }
  VLOG(3) << "Total weight max value number: " << max_value_num;
  filter_max_guard_ =
      TargetWrapperXPU::MallocScratchPad(max_value_num * sizeof(float));
  float* filter_max_ptr = reinterpret_cast<float*>(filter_max_guard_->addr_);

  int offset = 0;
  for (auto max_tensor : filter_max) {
    float* cur_filter_max_ptr = filter_max_ptr + offset;
    auto len = max_tensor->numel();
    VLOG(6) << "weight max value: " << max_tensor->data<float>()[0] << " "
            << max_tensor->data<float>()[len - 1];
    std::vector<float> cpu_max(max_ptr_len, max_tensor->data<float>()[0]);
    lite::TargetWrapperXPU::MemcpySync(cur_filter_max_ptr,
                                       cpu_max.data(),
                                       sizeof(float) * max_ptr_len,
                                       IoDirection::HtoD);
    max_xpu_ptrs->push_back(cur_filter_max_ptr);
    offset += max_ptr_len;
  }
}

template <typename InType, PrecisionType PType>
void XPUSpatialTransformerCompute<InType, PType>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();
  xft_attn_fc_bias_.emplace_back(
      const_cast<float*>(param.fc_bias[0]->template data<float>()),
      xft::xftVec<float>::dim_t{param.fc_bias[0]->dims()[0]});
  xft_attn_fc_bias_.emplace_back(
      const_cast<float*>(param.fc_bias[1]->template data<float>()),
      xft::xftVec<float>::dim_t{param.fc_bias[1]->dims()[0]});
  xft_geglu_fc_bias_.emplace_back(
      const_cast<float*>(param.fc_bias[2]->template data<float>()),
      xft::xftVec<float>::dim_t{param.fc_bias[2]->dims()[0]});
  xft_geglu_fc_bias_.emplace_back(
      const_cast<float*>(param.fc_bias[3]->template data<float>()),
      xft::xftVec<float>::dim_t{param.fc_bias[3]->dims()[0]});
  // prepare scale
  for (auto* ln_scale : param.ln_scale) {
    xft_ln_weights_.emplace_back(
        const_cast<float*>(ln_scale->template data<float>()),
        xft::xftVec<float>::dim_t{ln_scale->dims()[0]});
  }
  // prepare ln_bias
  for (auto* ln_bias : param.ln_bias) {
    xft_ln_bias_.emplace_back(
        const_cast<float*>(ln_bias->template data<float>()),
        xft::xftVec<float>::dim_t{ln_bias->dims()[0]});
  }

  // prepare gn_scale
  for (auto* gn_scale : param.gn_scale) {
    xft_gn_weights_.emplace_back(
        const_cast<float*>(gn_scale->template data<float>()),
        xft::xftVec<float>::dim_t{gn_scale->dims()[0]});
  }
  // prepare gn_bias
  for (auto* gn_bias : param.gn_bias) {
    xft_gn_bias_.emplace_back(
        const_cast<float*>(gn_bias->template data<float>()),
        xft::xftVec<float>::dim_t{gn_bias->dims()[0]});
  }
  // prepare conv bias
  for (auto* conv_bias : param.conv_bias) {
    xft_conv_bias_.emplace_back(
        const_cast<float*>(conv_bias->template data<float>()),
        xft::xftVec<float>::dim_t{conv_bias->dims()[0]});
  }

  arg_fc_weight_int16_ = PrepareWeight<int16_t>(param.fc_weight);
  arg_conv_filter_int16_ = PrepareWeight<int16_t>(param.conv_weight);
  const int XPU_QUANT_SCALE_NUM = ctx.GetRawContext()->max_ptr_size();
  PrepareWeightMax(param.weight_max, XPU_QUANT_SCALE_NUM, &fc_weight_max_);
  PrepareFilterMax(param.conv_max, XPU_QUANT_SCALE_NUM, &conv_filter_max_);

  int channel = static_cast<int>(param.input->dims()[1]);
  int xh = static_cast<int>(param.input->dims()[2]);
  int xw = static_cast<int>(param.input->dims()[3]);
  int hidden_dim = xh * xw;
  int embedding_dim = static_cast<int>(param.embedding->dims()[2]);

  // xft fc weights
  xft_q_weights_.emplace_back(
      const_cast<int16_t*>(arg_fc_weight_int16_[0]),
      const_cast<float*>(fc_weight_max_[0]),
      xft::xftMat<int16_t>::dim_t{hidden_dim, hidden_dim});
  xft_q_weights_.emplace_back(
      const_cast<int16_t*>(arg_fc_weight_int16_[4]),
      const_cast<float*>(fc_weight_max_[4]),
      xft::xftMat<int16_t>::dim_t{hidden_dim, hidden_dim});
  xft_k_weights_.emplace_back(
      const_cast<int16_t*>(arg_fc_weight_int16_[1]),
      const_cast<float*>(fc_weight_max_[1]),
      xft::xftMat<int16_t>::dim_t{hidden_dim, hidden_dim});
  xft_k_weights_.emplace_back(
      const_cast<int16_t*>(arg_fc_weight_int16_[5]),
      const_cast<float*>(fc_weight_max_[5]),
      xft::xftMat<int16_t>::dim_t{hidden_dim, embedding_dim});
  xft_v_weights_.emplace_back(
      const_cast<int16_t*>(arg_fc_weight_int16_[2]),
      const_cast<float*>(fc_weight_max_[2]),
      xft::xftMat<int16_t>::dim_t{hidden_dim, hidden_dim});
  xft_v_weights_.emplace_back(
      const_cast<int16_t*>(arg_fc_weight_int16_[6]),
      const_cast<float*>(fc_weight_max_[6]),
      xft::xftMat<int16_t>::dim_t{hidden_dim, embedding_dim});
  xft_attn_fc_weights_.emplace_back(
      const_cast<int16_t*>(arg_fc_weight_int16_[3]),
      const_cast<float*>(fc_weight_max_[3]),
      xft::xftMat<int16_t>::dim_t{hidden_dim, hidden_dim});
  xft_attn_fc_weights_.emplace_back(
      const_cast<int16_t*>(arg_fc_weight_int16_[7]),
      const_cast<float*>(fc_weight_max_[7]),
      xft::xftMat<int16_t>::dim_t{hidden_dim, hidden_dim});
  xft_geglu_fc_weights_.emplace_back(
      const_cast<int16_t*>(arg_fc_weight_int16_[8]),
      const_cast<float*>(fc_weight_max_[8]),
      xft::xftMat<int16_t>::dim_t{param.geglu_dim * 2, hidden_dim});
  xft_geglu_fc_weights_.emplace_back(
      const_cast<int16_t*>(arg_fc_weight_int16_[9]),
      const_cast<float*>(fc_weight_max_[9]),
      xft::xftMat<int16_t>::dim_t{hidden_dim, param.geglu_dim});
  for (size_t i = 0; i < arg_conv_filter_int16_.size(); i++) {
    int kh = param.filter_dims[i][2];
    int kw = param.filter_dims[i][3];
    xft_conv_weights_.emplace_back(
        const_cast<int16_t*>(arg_conv_filter_int16_[i]),
        const_cast<float*>(conv_filter_max_[i]),
        xft::xftTensor<int16_t, 4>::dim_t{channel, hidden_dim, kh, kw});
  }
  st_param_.n_head = param.head_num;
  st_param_.size_per_head = param.size_per_head,
  st_param_.geglu_dim = param.geglu_dim;
  st_param_.add_res = true;
  st_param_.conv_groups = param.conv_groups;
  st_param_.kernel_dims = param.filter_dims;
  st_param_.dilations = param.dilations;
  st_param_.paddings = param.paddings;
  st_param_.strides = param.strides;
  st_param_.gn_groups.push_back(param.groups);
  st_param_.gn_eps.push_back(param.epsilon);
}

template <typename InType, PrecisionType PType>
void XPUSpatialTransformerCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const InType* in = param.input->template data<InType>();
  const InType* embedding = param.embedding->template data<InType>();
  InType* out = param.output->template mutable_data<InType>(TARGET(kXPU));
  int batch = static_cast<int>(param.input->dims()[0]);
  int hidden_dim = static_cast<int>(param.input->dims()[1]);
  int channel = hidden_dim;
  int xh = static_cast<int>(param.input->dims()[2]);
  int xw = static_cast<int>(param.input->dims()[3]);
  int embedding_seq = static_cast<int>(param.embedding->dims()[1]);
  int embedding_dim = static_cast<int>(param.embedding->dims()[2]);
  // input
  xft::xftTensor<InType, 4> in_tensor(
      const_cast<InType*>(in), nullptr, {batch, channel, xh, xw});
  xft::xftTensor<InType, 3> embedding_tensor(
      const_cast<InType*>(embedding),
      nullptr,
      {batch, embedding_seq, embedding_dim});
  // output
  xft::xftTensor<InType, 4> output_tensor(out, {batch, channel, xh, xw});
  int r = xft::st_spatial_transformer_fusion<InType, int16_t, int16_t>(
      ctx.GetRawContext(),
      in_tensor,
      embedding_tensor,
      xft_ln_weights_,
      xft_ln_bias_,
      xft_gn_weights_,
      xft_gn_bias_,
      xft_q_weights_,
      xft_k_weights_,
      xft_v_weights_,
      xft_attn_fc_weights_,
      xft_attn_fc_bias_,
      xft_geglu_fc_weights_,
      xft_geglu_fc_bias_,
      xft_conv_weights_,
      xft_conv_bias_,
      &output_tensor,
      st_param_);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

// using XPUSpatialTransformer_FP32 = xpu::XPUSpatialTransformerCompute<float,
// PRECISION(kFloat)>;
using XPUSpatialTransformer_FP16 =
    xpu::XPUSpatialTransformerCompute<float16, PRECISION(kFP16)>;

// REGISTER_LITE_KERNEL(
//     __xpu__spatial_transformer,
//     kXPU,
//     kFloat,
//     kNCHW,
//     XPUSpatialTransformer_FP32,
//     def)
//     .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
//     .BindInput("Embedding", {LiteType::GetTensorTy(TARGET(kXPU))})
//     .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
//     .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
//     .BindInput("LNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
//     .BindInput("LNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
//     .BindInput("ConvWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
//     .BindInput("ConvBias", {LiteType::GetTensorTy(TARGET(kXPU))})
//     .BindInput("GNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
//     .BindInput("GNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
//     .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
//     .Finalize();
REGISTER_LITE_KERNEL(__xpu__spatial_transformer,
                     kXPU,
                     kFP16,
                     kNCHW,
                     XPUSpatialTransformer_FP16,
                     def_fp16)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Embedding",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ConvWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ConvBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("GNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("GNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
