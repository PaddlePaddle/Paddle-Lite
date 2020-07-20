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

#include "lite/kernels/xpu/__xpu__search_attention_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUMmdnnSearchAttentionCompute::PrepareForRun() {
  offset_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(
      XPU_MAX_LOD_SIZE * sizeof(int), false /* use_l3 */);
  pad_begin_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(
      XPU_MAX_LOD_SIZE * sizeof(int), false /* use_l3 */);
  w_max_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(8 * sizeof(float), false /* use_l3 */);
  buffer_at_l3_guard_ = TargetWrapperXPU::MallocScratchPad(
      5 * L3_SLOT_SIZE * sizeof(float), false /* use_l3 */);
  buffer_at_gm_guard_ = TargetWrapperXPU::MallocScratchPad(
      5 * GM_SLOT_SIZE * sizeof(float), false /* use_l3 */);

  offset_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
  pad_begin_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
}

void XPUMmdnnSearchAttentionCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* X = param.X;
  auto* W = param.W;
  auto* b = param.b;
  float W_max = param.W_max;
  float alpha0 = param.alpha0;
  float alpha1 = param.alpha1;
  float mask = param.mask;

  const int16_t* w_data = W->data<int16_t>();
  const float* b_data = b->data<float>();

  int batch = X->lod()[0].size() - 1;
  int dim0 = X->dims()[0];
  int dim1 = X->dims()[1];
  const auto offset = X->lod()[0];
  int max_seq = 0;

  auto* top = param.Out;
  LoD top_lod;
  top_lod.push_back(X->lod()[0]);
  top->set_lod(top_lod);
  top->Resize({dim0, dim1});
  auto* top_data = top->mutable_data<float>(TARGET(kXPU));

  float maxs_cpu[8] = {0.0f, 0.0f, 0.0f, 0.0f, W_max, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i < batch; ++i) {
    offset_cpu[i] = offset[i];  // type of offset is int64, not supported by xpu
    pad_begin_cpu[i] = offset[i + 1] - offset[i];
    if (offset[i + 1] - offset[i] > max_seq) {
      max_seq = offset[i + 1] - offset[i];
    }
  }
  offset_cpu[batch] = offset[batch];

  XPU_CALL(xpu_memcpy(offset_xpu_guard_->addr_,
                      offset_cpu.get(),
                      offset.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(pad_begin_xpu_guard_->addr_,
                      pad_begin_cpu.get(),
                      batch * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(w_max_xpu_guard_->addr_,
                      maxs_cpu,
                      8 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  int* offset_xpu = reinterpret_cast<int*>(offset_xpu_guard_->addr_);
  int* pad_begin_xpu = reinterpret_cast<int*>(pad_begin_xpu_guard_->addr_);
  float* maxs_xpu = reinterpret_cast<float*>(w_max_xpu_guard_->addr_);
  float* buffer_at_l3 = reinterpret_cast<float*>(buffer_at_l3_guard_->addr_);
  float* buffer_at_gm = reinterpret_cast<float*>(buffer_at_gm_guard_->addr_);

  // when use l3, max_seq <= 128:
  // group_padding:           batch * max_seq * dim1;     at (slot0, slot1)
  // seq_fc:                  batch * max_seq * dim1;     at (slot2, slot3)
  // batchgemm0:              batch * max_seq * max_seq;  at slot4
  // attention_padding_mask:  batch * max_seq * max_seq;  at slot3
  // seq_softmax:             batch * max_seq * max_seq;  at slot4
  // batchgemm1:              batch * max_seq * dim1;     at (slot2, slot3)
  float* group_padding_output = buffer_at_l3;
  float* seq_fc_output = buffer_at_l3 + 2 * L3_SLOT_SIZE;
  float* batchgemm0_output = buffer_at_l3 + 4 * L3_SLOT_SIZE;
  float* attention_output = buffer_at_l3 + 3 * L3_SLOT_SIZE;
  float* seq_softmax_output = buffer_at_l3 + 4 * L3_SLOT_SIZE;
  float* batchgemm1_output = buffer_at_l3 + 2 * L3_SLOT_SIZE;

  if (max_seq > 128) {
    group_padding_output = buffer_at_gm;
    seq_fc_output = buffer_at_gm + 1 * GM_SLOT_SIZE;
    batchgemm0_output = buffer_at_gm + 2 * GM_SLOT_SIZE;
    attention_output = buffer_at_gm + 1 * GM_SLOT_SIZE;
    seq_softmax_output = buffer_at_gm + 3 * GM_SLOT_SIZE;
    batchgemm1_output = buffer_at_gm + 4 * GM_SLOT_SIZE;
  }

  const auto* bottom_data = X->data<float>();
  int r = 0;
  r = xdnn::search_sequence_pad_depad(ctx.GetRawContext(),
                                      const_cast<float*>(bottom_data),
                                      group_padding_output,
                                      offset_xpu,
                                      max_seq,
                                      batch,
                                      dim1,
                                      0);  // is_depad = 0
  CHECK_EQ(r, 0);
  // do-findmax
  r = xdnn::findmax<float>(ctx.GetRawContext(),
                           group_padding_output,
                           batch * max_seq * dim1,
                           maxs_xpu);
  CHECK_EQ(r, 0);
  r = xdnn::gemm_int16_maxptr<float, int16_t, float>(
      ctx.GetRawContext(),        /* ctx */
      false,                      /* trans_a */
      true,                       /* trans_b */
      batch * max_seq,            /* m */
      dim1,                       /* n */
      dim1,                       /* k */
      1.0f,                       /* alpha */
      group_padding_output,       /* data_a */
      dim1,                       /* lda */
      w_data,                     /* data_b */
      dim1,                       /* ldb */
      0.0f,                       /* beta */
      seq_fc_output,              /* data_c */
      dim1,                       /* ldc */
      b_data,                     /* bias */
      xdnn::Activation_t::LINEAR, /* act */
      maxs_xpu,                   /* max_a */
      maxs_xpu + 4,               /* max_b */
      nullptr /* max_c */);
  CHECK_EQ(r, 0);
  r = xdnn::search_aligned_mat_mul(ctx.GetRawContext(),
                                   0,
                                   1,
                                   batch,
                                   max_seq,
                                   max_seq,
                                   dim1,
                                   alpha0,
                                   group_padding_output,
                                   dim1,
                                   seq_fc_output,
                                   dim1,
                                   batchgemm0_output,
                                   max_seq);
  CHECK_EQ(r, 0);
  r = xdnn::search_pad_mask(ctx.GetRawContext(),
                            batchgemm0_output,
                            attention_output,
                            pad_begin_xpu,
                            batch,
                            max_seq,
                            max_seq,
                            batch,
                            mask);
  CHECK_EQ(r, 0);
  r = xdnn::softmax2d_forward(ctx.GetRawContext(),
                              attention_output,
                              seq_softmax_output,
                              batch * max_seq,
                              max_seq,
                              true);
  CHECK_EQ(r, 0);
  r = xdnn::search_aligned_mat_mul(ctx.GetRawContext(),
                                   0,
                                   0,
                                   batch,
                                   max_seq,
                                   dim1,
                                   max_seq,
                                   alpha1,
                                   seq_softmax_output,
                                   max_seq,
                                   group_padding_output,
                                   dim1,
                                   batchgemm1_output,
                                   dim1);
  CHECK_EQ(r, 0);
  r = xdnn::search_sequence_pad_depad(ctx.GetRawContext(),
                                      top_data,
                                      batchgemm1_output,
                                      offset_xpu,
                                      max_seq,
                                      batch,
                                      dim1,
                                      1);  // is_depad = 1
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__mmdnn_search_attention,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUMmdnnSearchAttentionCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
