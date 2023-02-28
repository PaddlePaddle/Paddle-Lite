// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/anchor_generator_compute.h"
#include <string>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void AnchorGeneratorCompute::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  var_guard_ = TargetWrapperXPU::MallocScratchPad(param.variances.size() *
                                                  sizeof(float));
  XPU_CALL(xpu_memcpy(var_guard_->addr_,
                      param.variances.data(),
                      param.variances.size() * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
}

void AnchorGeneratorCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto* anchors = param.Anchors;
  auto* variances = param.Variances;
  auto* input = param.Input;
  float* anchors_data = anchors->mutable_data<float>(TARGET(kXPU));
  float* variances_data = variances->mutable_data<float>(TARGET(kXPU));
  auto input_dims = input->dims();
  int feature_height = input_dims[2];
  int feature_width = input_dims[3];
  int num_anchors = param.aspect_ratios.size() * param.anchor_sizes.size();
  int r = xdnn::anchor_generator<float>(ctx.GetRawContext(),
                                        anchors_data,
                                        feature_height,
                                        feature_width,
                                        param.aspect_ratios,
                                        param.anchor_sizes,
                                        param.stride,
                                        param.offset);
  CHECK_EQ(r, 0);
  std::vector<int> xshape{1, 1, 1, static_cast<int>(param.variances.size())};
  std::vector<int> yshape{feature_height, feature_width, num_anchors, 4};
  r = xdnn::broadcast<float>(ctx.GetRawContext(),
                             reinterpret_cast<float*>(var_guard_->addr_),
                             variances_data,
                             xshape,
                             yshape);
  CHECK_EQ(r, 0);

  return;
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(anchor_generator,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::AnchorGeneratorCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Anchors", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Variances", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
