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

#include "lite/kernels/host/prior_box_compute.h"
#include <string>
#include <vector>
#include "lite/backends/host/math/prior_box.h"

namespace paddle {
namespace lite_metal {
namespace kernels {
namespace host {

void PriorBoxCompute::ReInitWhenNeeded() {
  auto& param = this->template Param<param_t>();
  auto input_dims = param.input->dims();
  auto image_dims = param.image->dims();
  if (last_input_shape_ == input_dims && last_image_shape_ == image_dims) {
    return;
  }
  bool is_flip = param.flip;
  bool is_clip = param.clip;
  std::vector<float> min_size = param.min_sizes;
  std::vector<float> max_size = param.max_sizes;
  std::vector<float> aspect_ratio = param.aspect_ratios;
  std::vector<float> variance = param.variances_;
  int img_w = param.img_w;
  int img_h = param.img_h;
  float step_w = param.step_w;
  float step_h = param.step_h;
  float offset = param.offset;
  std::vector<float> aspect_ratios_vec;
  lite_metal::host::math::ExpandAspectRatios(
      aspect_ratio, is_flip, &aspect_ratios_vec);
  size_t prior_num = aspect_ratios_vec.size() * min_size.size();
  prior_num += max_size.size();
  std::vector<std::string> order = param.order;
  bool min_max_aspect_ratios_order = param.min_max_aspect_ratios_order;
  lite_metal::host::math::DensityPriorBox(param.input,
                                    param.image,
                                    &boxes_tmp_,
                                    &variances_tmp_,
                                    min_size,
                                    std::vector<float>(),
                                    std::vector<float>(),
                                    std::vector<int>(),
                                    max_size,
                                    aspect_ratios_vec,
                                    variance,
                                    img_w,
                                    img_h,
                                    step_w,
                                    step_h,
                                    offset,
                                    prior_num,
                                    is_flip,
                                    is_clip,
                                    order,
                                    min_max_aspect_ratios_order);
  last_input_shape_ = input_dims;
  last_image_shape_ = image_dims;
}

void PriorBoxCompute::Run() {
  auto& param = this->template Param<param_t>();
  param.boxes->CopyDataFrom(boxes_tmp_);
  param.variances->CopyDataFrom(variances_tmp_);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(prior_box,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite_metal::kernels::host::PriorBoxCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Image", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Variances", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
