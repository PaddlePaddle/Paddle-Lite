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

#include "lite/kernels/xpu/__xpu__search_attention_2_compute.h"
#include <algorithm>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUMmdnnSearchAttention2Compute::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  int maxptr_size = ctx.GetRawContext()->max_ptr_size();
  input_max_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(maxptr_size * sizeof(float));
  weight_max_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(maxptr_size * sizeof(float));
  output_max_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(maxptr_size * sizeof(float));
  auto& param = this->template Param<param_t>();
  float W_max = param.W_max;
  float weight_max_cpu[maxptr_size] = {W_max};
  XPU_CALL(xpu_memcpy(weight_max_xpu_guard_->addr_,
                      weight_max_cpu,
                      maxptr_size * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  dim_ = (param.W)->dims()[0];
}

void XPUMmdnnSearchAttention2Compute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* input = param.X;
  auto* weight = param.W;
  auto* bias = param.b;
  auto* output = param.Out;
  float alpha0 = param.alpha0;
  float alpha1 = param.alpha1;

  auto& input_lod = input->lod()[0];
  std::vector<int> m_lists;
  std::vector<int> k_lists;
  int batch = input_lod.size() - 1;
  int seqlen_sum = input->dims()[0];  // cap_l
  int seqlen_square_sum = 0;
  for (int i = 0; i < batch; i++) {
    int seqlen = input_lod[i + 1] - input_lod[i];
    seqlen_square_sum += seqlen * seqlen;
    m_lists.push_back(input_lod[i + 1] - input_lod[i]);
    k_lists.push_back(dim_);
  }

  XPUScratchPadGuard m_lists_xpu_guard =
      TargetWrapperXPU::MallocScratchPad(m_lists.size() * sizeof(int));
  int* m_lists_data = reinterpret_cast<int*>(m_lists_xpu_guard->addr_);
  XPU_CALL(xpu_memcpy(m_lists_data,
                      m_lists.data(),
                      m_lists.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPUScratchPadGuard k_lists_xpu_guard =
      TargetWrapperXPU::MallocScratchPad(k_lists.size() * sizeof(int));
  int* k_lists_data = reinterpret_cast<int*>(k_lists_xpu_guard->addr_);
  XPU_CALL(xpu_memcpy(k_lists_data,
                      k_lists.data(),
                      k_lists.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  LoD output_lod;
  output_lod.push_back(input_lod);
  output->set_lod(output_lod);
  output->Resize(input->dims());

  const float* input_data = input->data<float>();
  float* input_max_data = reinterpret_cast<float*>(input_max_xpu_guard_->addr_);
  const int16_t* weight_data = weight->data<int16_t>();
  float* weight_max_data =
      reinterpret_cast<float*>(weight_max_xpu_guard_->addr_);
  const float* bias_data = bias->data<float>();
  float* output_data = output->mutable_data<float>(TARGET(kXPU));
  float* output_max_data =
      reinterpret_cast<float*>(output_max_xpu_guard_->addr_);

  // fc_out: [cap_l, dim_], reuse of output
  // softmax_out: [seqlen_square_sum]
  // output/batchgemm1_out: [cap_l, dim_]
  XPUScratchPadGuard internal_result_xpu_guard =
      TargetWrapperXPU::MallocScratchPad(seqlen_square_sum * sizeof(float));
  float* fc_out = output_data;
  float* softmax_out =
      reinterpret_cast<float*>(internal_result_xpu_guard->addr_);
  float* batchgemm1_out = output_data;

  int r = 0;
  r = xdnn::findmax<float>(
      ctx.GetRawContext(), input_data, input_max_data, seqlen_sum * dim_);
  CHECK_EQ(r, 0);
  r = xdnn::fc_fusion<float, int16_t, float, int16_t>(
      ctx.GetRawContext(), /* context */
      input_data,          /* x */
      weight_data,
      fc_out,                      /* y */
      seqlen_sum,                  /* m */
      dim_,                        /* n */
      dim_,                        /* k */
      false,                       /* x_trans */
      true,                        /* w_trans */
      input_max_data,              /* x_max */
      weight_max_data,             /* w_max */
      output_max_data,             /* y_max */
      dim_,                        /* ldx */
      dim_,                        /* ldw */
      dim_,                        /* ldy */
      1.0f,                        /* alpha */
      0.0f,                        /* beta */
      bias_data,                   /* bias */
      xdnn::Activation_t::LINEAR); /* act_type */
  CHECK_EQ(r, 0);

  r = xdnn::fc_batched_vsl<float, float, float, int, int16_t>(
      ctx.GetRawContext(),
      input_data,
      fc_out,
      softmax_out,
      {m_lists.data(), static_cast<int>(m_lists.size()), m_lists_data},
      {m_lists.data(), static_cast<int>(m_lists.size()), m_lists_data},
      {k_lists.data(), static_cast<int>(k_lists.size()), k_lists_data},
      false,
      true,
      alpha0,
      true);
  CHECK_EQ(r, 0);
  r = xdnn::fc_batched_vsl<float, float, float, int, int16_t>(
      ctx.GetRawContext(),
      softmax_out,
      input_data,
      batchgemm1_out,
      {m_lists.data(), static_cast<int>(m_lists.size()), m_lists_data},
      {k_lists.data(), static_cast<int>(k_lists.size()), k_lists_data},
      {m_lists.data(), static_cast<int>(m_lists.size()), m_lists_data},
      false,
      false,
      alpha1,
      false);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    __xpu__mmdnn_search_attention2,
    kXPU,
    kFloat,
    kNCHW,
    paddle::lite::kernels::xpu::XPUMmdnnSearchAttention2Compute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
