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
  input_max_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(4 * sizeof(float));
  weight_max_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(4 * sizeof(float));
  output_max_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(4 * sizeof(float));
  auto& param = this->Param<param_t>();
  float W_max = param.W_max;
  float weight_max_cpu[4] = {W_max, W_max, W_max, W_max};
  XPU_CALL(xpu_memcpy(weight_max_xpu_guard_->addr_,
                      weight_max_cpu,
                      4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  dim_ = (param.W)->dims()[0];
}

void XPUMmdnnSearchAttention2Compute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* input = param.X;
  auto* weight = param.W;
  auto* bias = param.b;
  auto* output = param.Out;
  float alpha0 = param.alpha0;
  float alpha1 = param.alpha1;

  auto& input_lod = input->lod()[0];
  std::vector<int> lod_32;
  for (auto element : input_lod) {
    lod_32.push_back(element);
  }
  XPUScratchPadGuard lod_32_xpu_guard =
      TargetWrapperXPU::MallocScratchPad(lod_32.size() * sizeof(int));
  int* lod_32_data = reinterpret_cast<int*>(lod_32_xpu_guard->addr_);
  XPU_CALL(xpu_memcpy(lod_32_data,
                      lod_32.data(),
                      lod_32.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  int batch = input_lod.size() - 1;
  int seqlen_sum = input->dims()[0];  // cap_l
  int seqlen_max = 0;                 // max_width
  int seqlen_square_sum = 0;
  for (int i = 0; i < batch; i++) {
    int seqlen = input_lod[i + 1] - input_lod[i];
    seqlen_max = std::max(seqlen_max, seqlen);
    seqlen_square_sum += seqlen * seqlen;
  }

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
  // batchgemm0_out: [seqlen_square_sum]
  // softmax_out: [seqlen_square_sum], reuse of batchgemm0_out
  // output/batchgemm1_out: [cap_l, dim_]
  XPUScratchPadGuard internal_result_xpu_guard =
      TargetWrapperXPU::MallocScratchPad(seqlen_square_sum * sizeof(float));
  float* fc_out = output_data;
  float* batchgemm0_out =
      reinterpret_cast<float*>(internal_result_xpu_guard->addr_);
  float* softmax_out = batchgemm0_out;
  float* batchgemm1_out = output_data;

  int r = 0;
  r = xdnn::findmax<float>(
      ctx.GetRawContext(), input_data, seqlen_sum * dim_, input_max_data);
  CHECK_EQ(r, 0);
  r = xdnn::fc_int16(ctx.GetRawContext(),
                     false,
                     true,
                     seqlen_sum,
                     dim_,
                     dim_,
                     1.0f,
                     input_data,
                     input_max_data,
                     weight_data,
                     weight_max_data,
                     0.0f,
                     fc_out,
                     output_max_data,
                     bias_data,
                     xdnn::Activation_t::LINEAR);
  CHECK_EQ(r, 0);
  r = xdnn::search_noaligned_mat_mul(ctx.GetRawContext(),
                                     0,
                                     1,
                                     batch,
                                     lod_32_data,
                                     seqlen_max,
                                     dim_,
                                     alpha0,
                                     input_data,
                                     fc_out,
                                     batchgemm0_out);
  CHECK_EQ(r, 0);
  r = xdnn::search_seq_softmax(ctx.GetRawContext(),
                               batchgemm0_out,
                               softmax_out,
                               lod_32_data,
                               batch,
                               seqlen_max);
  CHECK_EQ(r, 0);
  r = xdnn::search_noaligned_mat_mul(ctx.GetRawContext(),
                                     0,
                                     0,
                                     batch,
                                     lod_32_data,
                                     seqlen_max,
                                     dim_,
                                     alpha1,
                                     softmax_out,
                                     input_data,
                                     batchgemm1_out);
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
