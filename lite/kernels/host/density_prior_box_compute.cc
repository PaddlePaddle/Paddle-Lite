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

#include "lite/kernels/host/density_prior_box_compute.h"
#include <cmath>
#include <string>
#include <vector>
#include "lite/backends/host/math/prior_box.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void DensityPriorBoxCompute::Run() {
  auto& param = Param<operators::DensityPriorBoxParam>();
  bool is_flip = param.flip;
  bool is_clip = param.clip;
  std::vector<float> min_size = param.min_sizes;
  std::vector<float> fixed_size = param.fixed_sizes;
  std::vector<float> fixed_ratio = param.fixed_ratios;
  auto density_size = param.density_sizes;
  std::vector<float> max_size = param.max_sizes;
  std::vector<float> aspect_ratio = param.aspect_ratios;
  std::vector<float> variance = param.variances_;
  int img_w = param.img_w;
  int img_h = param.img_h;
  float step_w = param.step_w;
  float step_h = param.step_h;
  float offset = param.offset;
  std::vector<float> aspect_ratios_vec;
  lite::host::math::ExpandAspectRatios(
      aspect_ratio, is_flip, &aspect_ratios_vec);
  size_t prior_num = aspect_ratios_vec.size() * min_size.size();
  prior_num += max_size.size();
  if (fixed_size.size() > 0) {
    prior_num = fixed_size.size() * fixed_ratio.size();
  }
  if (density_size.size() > 0) {
    for (size_t i = 0; i < density_size.size(); ++i) {
      if (fixed_ratio.size() > 0) {
        prior_num += (fixed_ratio.size() * ((pow(density_size[i], 2)) - 1));
      } else {
        prior_num +=
            ((fixed_ratio.size() + 1) * ((pow(density_size[i], 2)) - 1));
      }
    }
  }
  std::vector<std::string> order = param.order;

  lite::host::math::DensityPriorBox(param.input,
                                    param.image,
                                    param.boxes,
                                    param.variances,
                                    min_size,
                                    fixed_size,
                                    fixed_ratio,
                                    density_size,
                                    max_size,
                                    aspect_ratio,
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
                                    false);
  if (param.flatten_to_2d) {
    auto out_dims = param.boxes->dims();
    int64_t sum = 1;
    for (int i = 0; i < out_dims.size() - 1; i++) {
      sum *= out_dims[i];
    }
    param.boxes->Resize({sum, 4});
    param.variances->Resize({sum, 4});
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(density_prior_box,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::DensityPriorBoxCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Image", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Variances", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
