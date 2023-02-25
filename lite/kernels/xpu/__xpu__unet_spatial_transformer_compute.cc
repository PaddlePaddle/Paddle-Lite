// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/__xpu__unet_spatial_transformer_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

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


template <typename InType, PrecisionType PType>
void XPUUnetSpatialTransformerCompute<InType, PType>::prepare_weight_max(
    const std::vector<lite::Tensor*>& weight_max,
    int max_ptr_len,
    std::vector<const float*>& max_xpu_ptrs) {
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
    max_xpu_ptrs.push_back(cur_weight_max_ptr);
    offset += max_ptr_len;
  }
}

template <typename InType, PrecisionType PType>
void XPUUnetSpatialTransformerCompute<InType, PType>::prepare_filter_max(
    const std::vector<lite::Tensor*>& filter_max,
    int max_ptr_len,
    std::vector<const float*>& max_xpu_ptrs) {
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
    max_xpu_ptrs.push_back(cur_filter_max_ptr);
    offset += max_ptr_len;
  }
}

template <typename InType, PrecisionType PType>
void XPUUnetSpatialTransformerCompute<InType, PType>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();
  // prepare bias
  for (auto* fc_bias : param.fc_bias) {
    arg_fc_bias_.push_back(fc_bias->template data<float>());
  }
  // prepare scale
  for (auto* ln_scale : param.ln_scale) {
    arg_ln_scale_.push_back(ln_scale->template data<float>());
  }
  // prepare ln_bias
  for (auto* ln_bias : param.ln_bias) {
    arg_ln_bias_.push_back(ln_bias->template data<float>());
  }

  // prepare gn_scale
  for (auto* gn_scale : param.gn_scale) {
    arg_gn_scale_.push_back(gn_scale->template data<float>());
  }
  // prepare gn_bias
  for (auto* gn_bias : param.gn_bias) {
    arg_gn_bias_.push_back(gn_bias->template data<float>());
  }
  
  // prepare conv bias
  for (auto* conv_bias : param.conv_bias) {
    arg_conv_bias_.push_back(conv_bias->template data<float>());
  }

  arg_fc_weight_int16_ = prepare_weight<int16_t>(param.fc_weight);
  arg_conv_filter_int16_ = prepare_weight<int16_t>(param.conv_weight);
  const int XPU_QUANT_SCALE_NUM = ctx.GetRawContext()->max_ptr_size();
  prepare_weight_max(param.weight_max, XPU_QUANT_SCALE_NUM, fc_weight_max_);
  prepare_filter_max(param.conv_max, XPU_QUANT_SCALE_NUM, conv_filter_max_);
}

template <typename InType, PrecisionType PType>
void XPUUnetSpatialTransformerCompute<InType, PType>::Run() {
    auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const InType* in = param.input->template data<InType>();
  const InType* embedding = param.embedding->template data<InType>();
  InType* out = param.output->template mutable_data<InType>(TARGET(kXPU));
  int batch = static_cast<int>(param.input->dims()[0]);
  int hidden_dim = static_cast<int>(param.input->dims()[1]);
  int xh = static_cast<int>(param.input->dims()[2]);
  int xw = static_cast<int>(param.input->dims()[3]);
  int embedding_seq = static_cast<int>(param.embedding->dims()[1]);
  int r = xdnn::unet_spatial_transformer_fusion<InType, int16_t, InType, int16_t>(
        ctx.GetRawContext(),
        in,
        embedding,
        *(XPUUnetSpatialTransformerCompute::get_weight<int16_t>()),
        *(XPUUnetSpatialTransformerCompute::get_filter<int16_t>()),
        out,
        arg_fc_bias_,
        arg_conv_bias_,
        arg_ln_scale_,
        arg_ln_bias_,
        arg_gn_scale_,
        arg_gn_bias_,
        fc_weight_max_,
        conv_filter_max_,
        param.filter_dims,
        param.dilations,
        param.paddings,
        param.strides,
        param.conv_groups,
        batch,
        param.head_num,
        param.size_per_head,
        xh,
        xw,
        hidden_dim,
        embedding_seq,
        param.embedding_dim,
        param.groups,
        param.epsilon,
        param.geglu_dim);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

// using XPUUnetSpatialTransformer_FP32 = xpu::XPUUnetSpatialTransformerCompute<float, PRECISION(kFloat)>;
using XPUUnetSpatialTransformer_FP16 = xpu::XPUUnetSpatialTransformerCompute<float16, PRECISION(kFP16)>;

// REGISTER_LITE_KERNEL(
//     __xpu__unet_spatial_transformer,
//     kXPU,
//     kFloat,
//     kNCHW,
//     XPUUnetSpatialTransformer_FP32,
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
REGISTER_LITE_KERNEL(
    __xpu__unet_spatial_transformer,
    kXPU,
    kFP16,
    kNCHW,
    XPUUnetSpatialTransformer_FP16,
    def_fp16)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Embedding", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ConvWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ConvBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("GNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("GNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
