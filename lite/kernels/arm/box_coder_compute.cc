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
#include <algorithm>

#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/box_coder_fp16.h"
#endif

namespace paddle {
namespace lite {
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

  lite::arm::math::decode_bboxes(row,
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

#ifdef ENABLE_ARM_FP16
void BoxCoderFp16Compute::Run() {
  auto& param = Param<operators::BoxCoderParam>();
  auto* prior_box = param.prior_box;
  auto* prior_box_var = param.prior_box_var;
  auto* target_box = param.target_box;
  auto* output_box = param.proposals;
  std::vector<float16_t> variance(param.variance.size());
  std::transform(param.variance.begin(),
                 param.variance.end(),
                 variance.begin(),
                 [](float x) { return static_cast<float16_t>(x); });
  std::string code_type = param.code_type;
  bool normalized = param.box_normalized;

  auto row = target_box->dims()[0];
  auto col = prior_box->dims()[0];
  if (code_type == "decode_center_size") {
    col = target_box->dims()[1];
  }
  auto len = prior_box->dims()[1];
  output_box->Resize({row, col, len});
  auto* output = output_box->mutable_data<float16_t>();

  const float16_t* loc_data = target_box->data<float16_t>();
  const float16_t* prior_data = prior_box->data<float16_t>();
  const float16_t* variance_data;
  bool var_len4 = false;
  if (param.prior_box_var != nullptr) {
    variance_data = prior_box_var->data<float16_t>();
    var_len4 = false;
  } else {
    variance_data = variance.data();
    var_len4 = true;
  }

  lite::arm::math::fp16::decode_bboxes(row,
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
#endif

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
REGISTER_LITE_KERNEL(box_coder,
                     kARM,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::arm::BoxCoderFp16Compute,
                     def)
    .BindInput("PriorBox",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("PriorBoxVar",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("TargetBox",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("OutputBox",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

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
