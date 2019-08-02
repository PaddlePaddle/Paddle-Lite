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

#include "lite/kernels/arm/box_coder_compute.h"
#include <string>
#include <vector>
#include "lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void BoxCoderCompute::Run() {
  auto& param = Param<operators::BoxCoderParam>();
  int axis = param.axis;
  bool box_normalized = param.box_normalized;
  std::string code_type = param.code_type;

  lite::arm::math::box_coder(param.proposals,
                             param.prior_box,
                             param.prior_box_var,
                             param.target_box,
                             code_type,
                             box_normalized,
                             axis);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(box_coder,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::BoxCoderCompute,
                     def)
    .BindInput("PriorBox", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("PriorBoxVar", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("TargetBox", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("OutputBox", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
