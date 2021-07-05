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
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite_metal {
namespace kernels {
namespace arm {

void BoxCoderCompute::Run() {
  auto& param = Param<operators::BoxCoderParam>();
  auto* prior_box = param.prior_box;
  auto* prior_box_var = param.prior_box_var;
  auto* target_box = param.target_box;
  auto* output_box = param.proposals;
  std::vector<float> variance = param.variance;
  std::string code_type = param.code_type;
  bool normalized = param.box_normalized;

  auto row = target_box->dims()[0];
  auto col = prior_box->dims()[0];
  if (code_type == "decode_center_size") {
    col = target_box->dims()[1];
  }
  auto len = prior_box->dims()[1];
  output_box->Resize({row, col, len});
  auto* output = output_box->mutable_data<float>();

  const float* loc_data = target_box->data<float>();
  const float* prior_data = prior_box->data<float>();
  const float* variance_data;
  bool var_len4 = false;
  if (param.prior_box_var != nullptr) {
    variance_data = prior_box_var->data<float>();
    var_len4 = false;
  } else {
    variance_data = param.variance.data();
    var_len4 = true;
  }

  lite_metal::arm::math::decode_bboxes(row,
                                 param.axis,
                                 loc_data,
                                 prior_data,
                                 variance_data,
                                 var_len4,
                                 code_type,
                                 normalized,
                                 col,
                                 output);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(box_coder,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite_metal::kernels::arm::BoxCoderCompute,
                     def)
    .BindInput("PriorBox", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("PriorBoxVar", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("TargetBox", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("OutputBox", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
