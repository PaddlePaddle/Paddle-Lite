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

#include "lite/kernels/fpga/density_prior_box_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

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
  ExpandAspectRatios(aspect_ratio, is_flip, &aspect_ratios_vec);
  size_t prior_num = aspect_ratios_vec.size() * min_size.size();
  prior_num += max_size.size();
  if (fixed_size.size() > 0) {
    prior_num = fixed_size.size() * fixed_ratio.size();
  }
  if (density_size.size() > 0) {
    for (int i = 0; i < density_size.size(); ++i) {
      if (fixed_ratio.size() > 0) {
        prior_num += (fixed_ratio.size() * ((pow(density_size[i], 2)) - 1));
      } else {
        prior_num +=
            ((fixed_ratio.size() + 1) * ((pow(density_size[i], 2)) - 1));
      }
    }
  }
  std::vector<std::string> order = param.order;

  lite::arm::math::density_prior_box(param.input,
                                     param.image,
                                     &param.boxes,
                                     &param.variances,
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
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(density_prior_box,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::DensityPriorBoxCompute,
                     fpga)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Image",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Variances", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
