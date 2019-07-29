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

#include "lite/kernels/arm/prior_box_compute.h"
#include <string>
#include <vector>
#include "lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void PriorBoxCompute::Run() {
  auto& param = Param<operators::PriorBoxParam>();

  CHECK_EQ(param.ins.size(), 2);  // inputs[0]-feature_map  inputs[1]-image_data

  bool is_flip = param.is_flip;
  bool is_clip = param.is_clip;
  std::vector<float> min_size = param.min_size;
  std::vector<float> max_size = param.max_size;
  std::vector<float> aspect_ratio = param.aspect_ratio;
  std::vector<float> variance = param.variance;
  int img_w = param.img_w;
  int img_h = param.img_h;
  float step_w = param.step_w;
  float step_h = param.step_h;
  float offset = param.offset;
  int prior_num = param.prior_num;
  std::vector<std::string> order = param.order;

  lite::arm::math::prior_box(param.ins,
                             &param.outs,
                             min_size,
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
                             order);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(prior_box,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::PriorBoxCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
