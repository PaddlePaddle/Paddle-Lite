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

#include "lite/kernels/xpu/__xpu__matmul_scale_softmax_v1_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename InType, PrecisionType PType>
void XpuMatmulScaleSoftmaxV1Compute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const InType* in_q = param.mat_q->template data<InType>();
  const InType* in_k = param.mat_k->template data<InType>();
  const InType* in_v = param.mat_v->template data<InType>();

  InType* out = param.output->template mutable_data<InType>(TARGET(kXPU));
  int batch = static_cast<int>(param.mat_q->dims()[0]);
  int head_num = static_cast<int>(param.mat_q->dims()[1]);
  int seqlen = static_cast<int>(param.mat_q->dims()[2]);
  int head_dim = static_cast<int>(param.mat_q->dims()[3]);
  // input
  xft::xftTensor<InType, 4> in_q_tensor(
      const_cast<InType*>(in_q), nullptr, {batch, head_num, seqlen, head_dim});
  xft::xftTensor<InType, 4> in_k_tensor(
      const_cast<InType*>(in_k), nullptr, {batch, head_num, seqlen, head_dim});
  xft::xftTensor<InType, 4> in_v_tensor(
      const_cast<InType*>(in_v), nullptr, {batch, head_num, seqlen, head_dim});
  // output
  xft::xftTensor<InType, 4> output_tensor(out,
                                          {batch, head_num, seqlen, head_dim});
  xft::STScaleSoftmaxParam st_param;
  st_param.alpha = param.alpha;
  st_param.matmul_trans_info = param.matmul_trans_info;
  int r = xft::st_fc_scale_softmax_fusion<InType, int16_t>(ctx.GetRawContext(),
                                                           in_q_tensor,
                                                           in_k_tensor,
                                                           in_v_tensor,
                                                           &output_tensor,
                                                           st_param);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using XPUSpatialTransformerResBlock_FP32 =
    xpu::XpuMatmulScaleSoftmaxV1Compute<float, PRECISION(kFloat)>;
using XPUSpatialTransformerResBlock_FP16 =
    xpu::XpuMatmulScaleSoftmaxV1Compute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(__xpu__matmul_scale_softmax_v1,
                     kXPU,
                     kFloat,
                     kNCHW,
                     XPUSpatialTransformerResBlock_FP32,
                     def)
    .BindInput("mat_q", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("mat_k", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("mat_v", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(__xpu__matmul_scale_softmax_v1,
                     kXPU,
                     kFP16,
                     kNCHW,
                     XPUSpatialTransformerResBlock_FP16,
                     def)
    .BindInput("mat_q", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("mat_k", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("mat_v", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
