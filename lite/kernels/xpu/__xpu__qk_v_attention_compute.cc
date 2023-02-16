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

#include "lite/kernels/xpu/__xpu__qk_v_attention_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUQkVAttentionCompute::PrepareForRun() {}

void XPUQkVAttentionCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  int batch = static_cast<int>(param.v->dims()[0]);
  int max_seqlen = static_cast<int>(param.v->dims()[1]);
  // std::vector<int> mask_shape;
  // no vsl
  /*xdnn::QKVAttnParam qkv_attn_param(
      batch,  // TODO(TingShenXD):Use EncoderAttnParam
      max_seqlen,
      param.head_num,
      param.head_dim,
      mask_shape,  // unused
      xdnn::Activation_t::LINEAR,
      -1);                                           // no slice
      */
  xdnn::EncoderAttnParam qkv_attn_param(
      batch, max_seqlen, param.head_num, param.head_dim);
  int r = xdnn::qk_v_attention<float, float, float, int>(  // TODO(TingShenXD):
                                                           // support quant
      ctx.GetRawContext(),
      param.qk->data<float>(),
      param.v->data<float>(),
      param.output->mutable_data<float>(TARGET(kXPU)),
      nullptr,  // max_qk
      nullptr,  // max_v
      nullptr,  // max_qkv
      qkv_attn_param);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__qk_v_attention,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUQkVAttentionCompute,
                     def)
    .BindInput("qk", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("v", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
