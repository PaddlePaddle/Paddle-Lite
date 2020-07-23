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

#include "lite/kernels/xpu/match_matrix_tensor_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void MatchMatrixTensorCompute::PrepareForRun() {
  wx_max_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(
      XPU_MAX_LOD_SIZE * sizeof(int), false /* use_l3 */);
  offset_l_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(
      XPU_MAX_LOD_SIZE * sizeof(int), false /* use_l3 */);
  offset_r_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(
      XPU_MAX_LOD_SIZE * sizeof(int), false /* use_l3 */);

  offset_l_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
  offset_r_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
}

void MatchMatrixTensorCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* x = param.x;
  auto* y = param.y;
  auto* w = param.w;
  auto* out = param.out;
  auto* tmp = param.tmp;
  int dim_t = param.dim_t;
  float w_max = param.__xpu__w_max;
  bool fuse_relu = param.fuse_relu;
  bool float_to_fix = param.__xpu__float_to_fix;
  CHECK(float_to_fix) << "W should be fixed point";

  xdnn::Activation_t act = xdnn::Activation_t::LINEAR;
  if (fuse_relu) {
    act = xdnn::Activation_t::RELU;
  }

  int dim_in = x->dims()[1];
  const auto& offset_l = x->lod()[0];
  const auto& offset_r = y->lod()[0];

  std::vector<size_t> top_offset;
  int top_size = 0;
  top_offset.push_back(top_size);
  for (size_t b = 0; b < x->lod()[0].size() - 1; b++) {
    int len_l = offset_l[b + 1] - offset_l[b];
    int len_r = offset_r[b + 1] - offset_r[b];
    top_size += dim_t * len_l * len_r;
    top_offset.push_back(top_size);
  }
  auto* bottom_l_data = x->data<float>();
  auto* bottom_r_data = y->data<float>();
  auto* w_data = w->data<int16_t>();
  auto* out_data = out->mutable_data<float>(TARGET(kXPU));
  auto* bottom_l_trans_data = tmp->mutable_data<float>(TARGET(kXPU));
  int batch_size = x->lod()[0].size() - 1;

  float* wx_max = reinterpret_cast<float*>(wx_max_xpu_guard_->addr_);
  int* offset_l_xpu = reinterpret_cast<int*>(offset_l_xpu_guard_->addr_);
  int* offset_r_xpu = reinterpret_cast<int*>(offset_r_xpu_guard_->addr_);

  int r = xdnn::gemm_int16_tmp_api<float, int16_t, float>(
      ctx.GetRawContext(),        /* ctx */
      false,                      /* trans_a */
      false,                      /* trans_b */
      x->dims()[0],               /* m */
      dim_t * dim_in,             /* n */
      dim_in,                     /* k */
      1.0f,                       /* alpha */
      bottom_l_data,              /* data_a */
      dim_in,                     /* lda */
      w_data,                     /* data_b */
      dim_t * dim_in,             /* ldb */
      0.0f,                       /* beta */
      bottom_l_trans_data,        /* data_c */
      dim_t * dim_in,             /* ldc */
      nullptr,                    /* bias */
      xdnn::Activation_t::LINEAR, /* act */
      0.0f,                       /* max_a */
      w_max,                      /* max_b */
      wx_max /* max_c */);
  CHECK_EQ(r, 0);

  int max_width = 0;
  for (int i = 0; i < offset_l.size(); ++i) {
    offset_l_cpu[i] = offset_l[i];
    if (i != 0 && (offset_l_cpu[i] - offset_l_cpu[i - 1] > max_width)) {
      max_width = offset_l_cpu[i] - offset_l_cpu[i - 1];
    }
  }
  for (int i = 0; i < offset_r.size(); ++i) {
    offset_r_cpu[i] = offset_r[i];
    if (i != 0 && (offset_r_cpu[i] - offset_r_cpu[i - 1] > max_width)) {
      max_width = offset_r_cpu[i] - offset_r_cpu[i - 1];
    }
  }
  XPU_CALL(xpu_memcpy(offset_l_xpu,
                      offset_l_cpu.get(),
                      offset_l.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(offset_r_xpu,
                      offset_r_cpu.get(),
                      offset_r.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  r = xdnn::match_matrix_tensor(ctx.GetRawContext(),
                                batch_size,
                                bottom_l_trans_data,
                                bottom_r_data,
                                offset_l_xpu,
                                offset_r_xpu,
                                dim_t,
                                dim_in,
                                out_data,
                                wx_max,
                                act,
                                max_width);
  CHECK_EQ(r, 0);

  int lod_lv1_size = batch_size * dim_t;
  int lod_lv2_size = x->lod()[0].back() * dim_t;
  std::vector<size_t> out_lod0(batch_size + 1, 0);
  std::vector<size_t> out_lod1(lod_lv1_size + 1, 0);
  std::vector<size_t> out_lod2(lod_lv2_size + 1, 0);
  for (int i = 0; i < batch_size; i++) {
    out_lod0[i + 1] = out_lod0[i] + dim_t;
    int len_l = offset_l[i + 1] - offset_l[i];

    for (int j = 0; j < dim_t; j++) {
      out_lod1[i * dim_t + j + 1] = out_lod1[i * dim_t + j] + len_l;
      int len_r = offset_r[i + 1] - offset_r[i];

      for (int k = 0; k < len_l; k++) {
        out_lod2[offset_l[i] * dim_t + j * len_l + k + 1] =
            out_lod2[offset_l[i] * dim_t + j * len_l + k] + len_r;
      }
    }
  }

  paddle::lite::LoD out_lod;
  out_lod.push_back(top_offset);
  out_lod.push_back(offset_l);
  out_lod.push_back(offset_r);
  out->set_lod(out_lod);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(match_matrix_tensor,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::MatchMatrixTensorCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Tmp", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
