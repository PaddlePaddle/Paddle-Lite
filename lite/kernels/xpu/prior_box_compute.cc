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

#include "lite/kernels/xpu/prior_box_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

inline void ExpandAspectRatios(const std::vector<float>& input_aspect_ratior,
                               bool flip,
                               std::vector<float>* output_aspect_ratior) {
  constexpr float epsilon = 1e-6;
  output_aspect_ratior->clear();
  output_aspect_ratior->push_back(1.0f);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
      if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior->push_back(ar);
      if (flip) {
        output_aspect_ratior->push_back(1.0f / ar);
      }
    }
  }
}

void PriorBoxCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  std::vector<float> min_size = param.min_sizes;
  std::vector<float> max_size = param.max_sizes;
  std::vector<float> aspect_ratio = param.aspect_ratios;
  std::vector<float> variance = param.variances_;
  std::vector<float> aspect_ratios_vec;
  bool is_flip = param.flip;

  ExpandAspectRatios(aspect_ratio, is_flip, &aspect_ratios_vec);
  prior_num = aspect_ratios_vec.size() * min_size.size();
  prior_num += max_size.size();

  CHECK_LE(aspect_ratios_vec.size(), 16);
  xpu_aspect_ratios_guard_ = TargetWrapperXPU::MallocScratchPad(
      16 * sizeof(float), false /* use_l3 */);
  XPU_CALL(xpu_memcpy(xpu_aspect_ratios_guard_->addr_,
                      aspect_ratios_vec.data(),
                      aspect_ratios_vec.size() * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  ar_num = aspect_ratios_vec.size();

  CHECK_LE(min_size.size(), 8);
  xpu_min_sizes_guard_ =
      TargetWrapperXPU::MallocScratchPad(8 * sizeof(float), false /* use_l3 */);
  XPU_CALL(xpu_memcpy(xpu_min_sizes_guard_->addr_,
                      min_size.data(),
                      min_size.size() * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  min_size_num = min_size.size();

  CHECK_LE(max_size.size(), 8);
  xpu_max_sizes_guard_ =
      TargetWrapperXPU::MallocScratchPad(8 * sizeof(float), false /* use_l3 */);
  XPU_CALL(xpu_memcpy(xpu_max_sizes_guard_->addr_,
                      max_size.data(),
                      max_size.size() * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  max_size_num = max_size.size();

  CHECK_EQ(variance.size(), 4);
  variance_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(4 * sizeof(float), false /* use_l3 */);
  XPU_CALL(xpu_memcpy(variance_xpu_guard_->addr_,
                      variance.data(),
                      variance.size() * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
}

void PriorBoxCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  bool is_clip = param.clip;
  auto image_dims = param.image->dims();
  int im_width = static_cast<int>(image_dims[3]);
  int im_height = static_cast<int>(image_dims[2]);
  auto feature_dims = param.input->dims();
  int height = static_cast<int>(feature_dims[2]);
  int width = static_cast<int>(feature_dims[3]);

  float step_width = 0.0;
  float step_height = 0.0;
  float step_w = param.step_w;
  float step_h = param.step_h;
  float offset = param.offset;

  if (step_w == 0 || step_h == 0) {
    step_width = static_cast<float>(im_width) / static_cast<float>(width);
    step_height = static_cast<float>(im_height) / static_cast<float>(height);
  } else {
    step_width = step_w;
    step_height = step_h;
  }

  param.boxes->Resize({height, width, prior_num, 4});
  param.variances->Resize({height, width, prior_num, 4});

  bool min_max_aspect_ratios_order = param.min_max_aspect_ratios_order;
  float* xpu_aspect_ratios =
      reinterpret_cast<float*>(xpu_aspect_ratios_guard_->addr_);
  float* xpu_min_sizes = reinterpret_cast<float*>(xpu_min_sizes_guard_->addr_);
  float* xpu_max_sizes = reinterpret_cast<float*>(xpu_max_sizes_guard_->addr_);

  int r = xdnn::prior_box_gen(ctx.GetRawContext(),
                              param.boxes->mutable_data<float>(TARGET(kXPU)),
                              xpu_aspect_ratios,
                              height,
                              width,
                              im_height,
                              im_width,
                              ar_num,
                              offset,
                              step_width,
                              step_height,
                              xpu_min_sizes,
                              xpu_max_sizes,
                              min_size_num,
                              max_size_num,
                              is_clip,
                              min_max_aspect_ratios_order);
  CHECK_EQ(r, 0);

  float* xpu_variances_in =
      reinterpret_cast<float*>(variance_xpu_guard_->addr_);
  r = xdnn::broadcast_ew(ctx.GetRawContext(),
                         xpu_variances_in,
                         param.variances->mutable_data<float>(TARGET(kXPU)),
                         height * width * prior_num,
                         4,
                         1,
                         xdnn::ElementwiseOp::ASSIGN);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(prior_box,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::PriorBoxCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Image", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Variances", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
