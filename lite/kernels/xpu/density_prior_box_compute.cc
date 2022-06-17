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

#include "lite/kernels/xpu/density_prior_box_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void DensityPriorBoxCompute::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  std::vector<float> variance = param.variances_;
  CHECK_EQ(variance.size(), 4);
  variance_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(4 * sizeof(float));
  XPU_CALL(xpu_memcpy(variance_xpu_guard_->addr_,
                      variance.data(),
                      variance.size() * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
}

void DensityPriorBoxCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  bool is_clip = param.clip;
  std::vector<float> fixed_size = param.fixed_sizes;
  std::vector<float> fixed_ratio = param.fixed_ratios;
  auto density_size = param.density_sizes;
  float step_w = param.step_w;
  float step_h = param.step_h;
  float offset = param.offset;
  int feature_w = param.input->dims()[3];
  int feature_h = param.input->dims()[2];
  int img_w = param.image->dims()[3];
  int img_h = param.image->dims()[2];

  int num_priors = 0;
  for (size_t i = 0; i < density_size.size(); ++i) {
    num_priors += (fixed_ratio.size()) * (pow(density_size[i], 2));
  }

  param.boxes->Resize({feature_h, feature_w, num_priors, 4});
  param.variances->Resize({feature_h, feature_w, num_priors, 4});

  int r = xdnn::density_prior_box<float>(
      ctx.GetRawContext(),
      param.boxes->mutable_data<float>(TARGET(kXPU)),
      img_h,
      img_w,
      feature_h,
      feature_w,
      fixed_size,
      fixed_ratio,
      density_size,
      step_w,
      step_h,
      offset,
      is_clip);

  CHECK_EQ(r, 0);

  float* xpu_variances_in =
      reinterpret_cast<float*>(variance_xpu_guard_->addr_);
  r = xdnn::broadcast<float>(ctx.GetRawContext(),
                             xpu_variances_in,
                             param.variances->mutable_data<float>(TARGET(kXPU)),
                             {1, 4, 1},
                             {feature_h * feature_w * num_priors, 4, 1});

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(density_prior_box,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::DensityPriorBoxCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Image", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Variances", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
