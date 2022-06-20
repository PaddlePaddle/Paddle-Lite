// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/roi_align_compute.h"
#include <string>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {
static constexpr int kROISize = 4;

void RoiAlignCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* in = param.X;
  auto* rois = param.ROIs;
  auto* out = param.Out;
  float spatial_scale = param.spatial_scale;
  int pooled_height = param.pooled_height;
  int pooled_width = param.pooled_width;
  int sampling_ratio = param.sampling_ratio;

  auto in_dims = in->dims();
  int batch_size = in_dims[0];
  int channels = in_dims[1];
  int height = in_dims[2];
  int width = in_dims[3];
  auto rois_dims = rois->dims();
  int rois_num = rois_dims[0];
  auto out_dims = out->dims();
  if (rois_num == 0) {
    return;
  }
  auto rois_lod = rois->lod().back();
  std::vector<int> cpu_lod_data;
  cpu_lod_data.resize(batch_size + 1);
  for (int i = 0; i < rois_lod.size(); i++) {
    cpu_lod_data[i] = rois_lod[i];
  }

  XPUScratchPadGuard xpu_lod_grad_ =
      TargetWrapperXPU::MallocScratchPad(cpu_lod_data.size() * sizeof(int));
  int* xpu_lod = reinterpret_cast<int*>(xpu_lod_grad_->addr_);
  XPU_CALL(xpu_memcpy(xpu_lod,
                      cpu_lod_data.data(),
                      sizeof(int) * cpu_lod_data.size(),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  int r = xdnn::roi_align<float, int>(ctx.GetRawContext(),
                                      in->data<float>(),
                                      out->mutable_data<float>(TARGET(kXPU)),
                                      rois->data<float>(),
                                      static_cast<const int*>(xpu_lod),
                                      batch_size,
                                      channels,
                                      height,
                                      width,
                                      rois_num,
                                      pooled_height,
                                      pooled_width,
                                      spatial_scale,
                                      sampling_ratio,
                                      true);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(roi_align,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::RoiAlignCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ROIs", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
