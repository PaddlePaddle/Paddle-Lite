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

#include "lite/kernels/xpu/__xpu__multihead_cross_attn_compute.h"
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
void XPUMhcaCompute<InType, PType>::PrepareWeightMax(
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
void XPUMhcaCompute<InType, PType>::PrepareForRun() {
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
  arg_fc_weight_int16_ = prepare_weight<int16_t>(param.fc_weight);
  const int XPU_QUANT_SCALE_NUM = ctx.GetRawContext()->max_ptr_size();
  PrepareWeightMax(param.weight_max, XPU_QUANT_SCALE_NUM, &fc_weight_max_);
}

template <typename InType, PrecisionType PType>
void XPUMhcaCompute<InType, PType>::Run() {
  // TODO(shenyijun): The compute of this op will be adapted to XFT interface
  // later on.
  //
  // auto& param = this->template Param<param_t>();
  // auto& ctx = this->ctx_->template As<XPUContext>();
  // const InType* in = param.input->template data<InType>();
  // const InType* embedding = param.embedding->template data<InType>();
  // InType* out = param.output->template mutable_data<InType>(TARGET(kXPU));
  // int batch = static_cast<int>(param.input->dims()[0]);
  // int seqlen = static_cast<int>(param.input->dims()[1]);
  // int embedding_seq = static_cast<int>(param.embedding->dims()[1]);
  // int r = xdnn::unet_mhca_fusion<InType, int16_t, InType, int16_t>(
  //        ctx.GetRawContext(),
  //        in,
  //        embedding,
  //        *(XPUMhcaCompute::GetWeight<int16_t>()),
  //        out,
  //        arg_fc_bias_,
  //        arg_ln_scale_,
  //        arg_ln_bias_,
  //        fc_weight_max_,
  //        batch,
  //        param.head_num,
  //        param.size_per_head,
  //        seqlen,
  //        param.hidden_dim,
  //        embedding_seq,
  //        param.embedding_dim);
  // CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using XPUMhca_FP32 = xpu::XPUMhcaCompute<float, PRECISION(kFloat)>;
using XPUMhca_FP16 = xpu::XPUMhcaCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(
    __xpu__multihead_cross_attn, kXPU, kFloat, kNCHW, XPUMhca_FP32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Embedding", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(
    __xpu__multihead_cross_attn, kXPU, kFP16, kNCHW, XPUMhca_FP16, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Embedding",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
