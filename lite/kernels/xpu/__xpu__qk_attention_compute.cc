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

#include "lite/kernels/xpu/__xpu__qk_attention_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUQkAttentionCompute::PrepareForRun() {
  mask_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(XPU_MAX_LOD_SIZE_64 * sizeof(int));
  cpu_mask.reserve(XPU_MAX_LOD_SIZE_64);
}

void XPUQkAttentionCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  int batch = static_cast<int>(param.q->dims()[0]);
  int max_seqlen = static_cast<int>(param.q->dims()[1]);
  std::vector<int> mask_shape;
  if (param.mask == nullptr) {
    mask_shape = {static_cast<int>(param.q->dims()[0]),
                  param.head_num,
                  static_cast<int>(param.q->dims()[1]),
                  static_cast<int>(param.q->dims()[1])};
    int mask_size = param.q->dims()[0] * param.head_num * param.q->dims()[1] *
                    param.q->dims()[1];
    cpu_mask = std::vector<float>(mask_size, 0);
    mask_xpu_addr = reinterpret_cast<float*>(mask_xpu_guard_->addr_);
    XPU_CALL(xpu_memcpy(mask_xpu_addr,
                        cpu_mask.data(),
                        mask_size * sizeof(float),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  } else {
    std::vector<int64_t> mask_shape_int64 = param.mask->dims().Vectorize();
    mask_shape =
        std::vector<int>(mask_shape_int64.begin(), mask_shape_int64.end());
    mask_xpu_addr = const_cast<float*>(param.mask->data<float>());
  }
  // no vsl TODO(TingShen): Support vsl with input.lod
  /*xdnn::QKVAttnParam qkv_attn_param(
      batch,  // TODO(TingShenXD):Use EncoderAttnParam
      max_seqlen,
      param.head_num,
      param.head_dim,
      mask_shape,
      xdnn::Activation_t::LINEAR,
      -1);  // no slice
  */
  xdnn::NewEncoderAttnParam<int, int> qk_attn_param(batch,
                                                    max_seqlen,
                                                    param.head_num,
                                                    param.head_dim,
                                                    mask_shape,
                                                    false,  // do_fc_qkv_fusion
                                                    -1,     // no slice
                                                    0,      // qkv_shape
                                                    param.alpha);
  int r =
      xdnn::qk_attention<float, float, float, int, float>(  // TODO(TingShenXD):
                                                            // support quant
          ctx.GetRawContext(),
          param.q->data<float>(),
          param.k->data<float>(),
          param.output->mutable_data<float>(TARGET(kXPU)),
          nullptr,  // max_q
          nullptr,  // max_k
          nullptr,  // max_qk
          qk_attn_param,
          mask_xpu_addr);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__qk_attention,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUQkAttentionCompute,
                     def)
    .BindInput("q", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("k", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
