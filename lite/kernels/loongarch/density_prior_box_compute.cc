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

#include "lite/kernels/loongarch/density_prior_box_compute.h"
#include <string>
#include <vector>
#include "lite/backends/loongarch/math/prior_box.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace loongarch {

void DensityPriorBoxCompute::Run() {
  auto& param = *param_.get_mutable<operators::DensityPriorBoxParam>();
  // required inputs
  auto* input = param.input;  // 4D tensor NCHW
  auto* image = param.image;  // 4D tensor NCHW
  // outputs
  auto* boxes = param.boxes;     // [H, W, num_priors, 4]
  auto* vars = param.variances;  // [H, W, num_priors, 4]
  // required attributes
  bool clip = param.clip;
  std::vector<float> variances = param.variances_;
  std::vector<float> fixed_sizes = param.fixed_sizes;
  std::vector<float> fixed_ratios = param.fixed_ratios;
  std::vector<int> densities = param.density_sizes;
  // optional attributes
  float step_w = param.step_w;
  float step_h = param.step_h;
  float offset = param.offset;

  auto img_width = image->dims()[3];
  auto img_height = image->dims()[2];

  auto feature_width = input->dims()[3];
  auto feature_height = input->dims()[2];

  float step_width, step_height;
  if (step_w == 0 || step_h == 0) {
    step_width = static_cast<float>(img_width) / feature_width;
    step_height = static_cast<float>(img_height) / feature_height;
  } else {
    step_width = step_w;
    step_height = step_h;
  }
  int num_priors = 0;

#pragma omp parallel for reduction(+ : num_priors)
  for (int i = 0; i < densities.size(); ++i) {
    num_priors += (fixed_ratios.size()) * (pow(densities[i], 2));
  }

  boxes->Resize({feature_height, feature_width, num_priors, 4});
  vars->Resize({feature_height, feature_width, num_priors, 4});

  auto* boxes_data = boxes->mutable_data<float>();
  auto* vars_data = vars->mutable_data<float>();

  const float* input_data = input->data<float>();
  const float* image_data = image->data<float>();

  lite::loongarch::math::density_prior_box(img_width,
                                     img_height,
                                     feature_width,
                                     feature_height,
                                     input_data,
                                     image_data,
                                     clip,
                                     variances,
                                     fixed_sizes,
                                     fixed_ratios,
                                     densities,
                                     step_width,
                                     step_height,
                                     offset,
                                     num_priors,
                                     boxes_data,
                                     vars_data);
  if (param.flatten_to_2d) {
    auto out_dims = boxes->dims();
    int64_t sum = 1;
    for (int i = 0; i < out_dims.size() - 1; i++) {
      sum *= out_dims[i];
    }
    boxes->Resize({sum, 4});
    vars->Resize({sum, 4});
  }
}

}  // namespace loongarch
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(density_prior_box,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::DensityPriorBoxCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("Image", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Variances", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();
