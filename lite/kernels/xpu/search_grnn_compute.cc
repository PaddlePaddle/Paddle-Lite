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

#include "lite/kernels/xpu/search_grnn_compute.h"
#include <algorithm>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SearchGrnnCompute::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  int maxptr_size = ctx.GetRawContext()->max_ptr_size();
  maxs_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(6 * maxptr_size * sizeof(float));
  offset_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
}

void SearchGrnnCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* bottom = param.x;
  auto* wi = param.wi;
  auto* wh = param.wh;
  auto* top = param.out;
  int cap_h = param.num_hidden;
  int cap_e = param.num_input;
  int cap_l = bottom->dims()[0];
  auto wi_max = param.__xpu__wi_max;
  auto wh_max = param.__xpu__wh_max;
  bool float_to_fix = param.__xpu__float_to_fix;
  CHECK(float_to_fix) << "W should be fixed point";

  int dim = 1;
  if (bottom->dims().size() > 1) {
    dim = bottom->dims()[1];
  }

  const auto& offset = bottom->lod()[0];
  LoD top_lod;
  top_lod.push_back(offset);
  top->set_lod(top_lod);
  std::vector<int64_t> top_dims_vec{cap_l, cap_h};
  top->Resize(top_dims_vec);
  auto* top_hidden = top->mutable_data<float>(TARGET(kXPU));
  const auto* dense_e2h = wi->data<int16_t>();
  const auto* dense_h2h = wh->data<int16_t>();

  CHECK_LE(offset.size(), 64);
  for (size_t i = 0; i < offset.size(); ++i) {
    offset_cpu[i] = offset[i];
  }
  int xpu_maxptr_size = ctx.GetRawContext()->max_ptr_size();
  std::vector<float> maxs_cpu(6 * xpu_maxptr_size, 0.0f);
  for (int idx = 0; idx < 3; idx++) {
    maxs_cpu[idx * xpu_maxptr_size] = wi_max[idx];
    maxs_cpu[(idx + 3) * xpu_maxptr_size] = wh_max[idx];
  }
  float* maxs_xpu = reinterpret_cast<float*>(maxs_xpu_guard_->addr_);
  XPU_CALL(xpu_memcpy(maxs_xpu,
                      maxs_cpu.data(),
                      6 * xpu_maxptr_size * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  float* weight_x_max = maxs_xpu;
  float* weight_w_max = maxs_xpu + 3 * xpu_maxptr_size;
  int r = xdnn::grnn_cell<float, int16_t, int, int16_t>(
      ctx.GetRawContext(),
      bottom->data<float>(),
      nullptr,
      {dense_e2h, dense_e2h + cap_e * cap_h, dense_e2h + cap_e * cap_h * 2},
      {dense_h2h, dense_h2h + cap_h * cap_h, dense_h2h + cap_h * cap_h * 2},
      top_hidden,
      cap_e,
      cap_h,
      {offset_cpu.get(), static_cast<int>(offset.size()), nullptr},
      nullptr,
      nullptr,
      {weight_x_max,
       weight_x_max + xpu_maxptr_size,
       weight_x_max + xpu_maxptr_size * 2},
      {weight_w_max,
       weight_w_max + xpu_maxptr_size,
       weight_w_max + xpu_maxptr_size * 2},
      nullptr);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(search_grnn,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SearchGrnnCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("tmp_buffer", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("idx_sorted_by_width",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("layout_input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
