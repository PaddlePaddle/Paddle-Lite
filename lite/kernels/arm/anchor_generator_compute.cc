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

#include "lite/kernels/arm/anchor_generator_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void AnchorGeneratorCompute::Run() {
  auto& param = Param<operators::AnchorGeneratorParam>();
  auto* anchors = param.Anchors;
  auto* variances = param.Variances;
  auto* input = param.Input;

  float* anchors_data = anchors->mutable_data<float>();
  float* variances_data = variances->mutable_data<float>();
  auto input_dims = input->dims();
  int feature_height = input_dims[2];
  int feature_width = input_dims[3];

  lite::arm::math::anchor_generator_func(feature_height,
                                         feature_width,
                                         param.anchor_sizes,
                                         param.aspect_ratios,
                                         param.stride,
                                         param.variances,
                                         param.offset,
                                         anchors_data,
                                         variances_data);
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(anchor_generator,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::AnchorGeneratorCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Anchors", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Variances", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
