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
  auto len = prior_box->dims()[1];      // 4
  output_box->Resize({row, col, len});  // N x M x 4
  auto* output = output_box->mutable_data<float>();
  auto axis = param.axis;

  const float* target_box_data = target_box->data<float>();
  const float* prior_box_data = prior_box->data<float>();

  bool var_len4 = false;
  int var_size = 0;
  const float* variance_data;
  if (prior_box_var != nullptr) {
    var_size = 2;
    variance_data = prior_box_var->data<float>();
    var_len4 = false;
  } else {
    var_size = 1;
    variance_data = param.variance.data();
    var_len4 = true;
  }
  if (code_type == "encode_center_size") {
    lite::arm::math::encode_bbox_center_kernel(row,
                                               target_box_data,
                                               prior_box_data,
                                               variance_data,
                                               var_len4,
                                               normalized,
                                               col,
                                               output);
  } else if (code_type == "decode_center_size") {
    if (axis == 0) {
      lite::arm::math::decode_bbox_center_kernel(row,
                                                 target_box_data,
                                                 prior_box_data,
                                                 variance_data,
                                                 var_len4,
                                                 col,
                                                 normalized,
                                                 output);
    } else {
      auto prior_box_var_data =
          prior_box_var ? prior_box_var->data<float>() : nullptr;
      lite::arm::math::decode_center_size_axis_1(var_size,
                                                 row,
                                                 col,
                                                 len,
                                                 target_box_data,
                                                 prior_box_data,
                                                 prior_box_var_data,
                                                 normalized,
                                                 variance,
                                                 output);
    }
  } else {
    LOG(FATAL) << "box_coder don't support this code_type: " << code_type;
  }
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
  auto axis = param.axis;

  const float16_t* target_box_data = target_box->data<float16_t>();
  const float16_t* prior_box_data = prior_box->data<float16_t>();

  bool var_len4 = false;
  int var_size = 0;
  const float16_t* variance_data;
  if (prior_box_var != nullptr) {
    var_size = 2;
    variance_data = prior_box_var->data<float16_t>();
    var_len4 = false;
  } else {
    var_size = 1;
    variance_data = variance.data();
    var_len4 = true;
  }
  if (code_type == "encode_center_size") {
    lite::arm::math::fp16::encode_bbox_center_kernel(row,
                                                     target_box_data,
                                                     prior_box_data,
                                                     variance_data,
                                                     var_len4,
                                                     normalized,
                                                     col,
                                                     output);
  } else if (code_type == "decode_center_size") {
    if (axis == 0) {
      lite::arm::math::fp16::decode_bbox_center_kernel(row,
                                                       target_box_data,
                                                       prior_box_data,
                                                       variance_data,
                                                       var_len4,
                                                       col,
                                                       normalized,
                                                       output);
    } else {
      auto prior_box_var_data =
          prior_box_var ? prior_box_var->data<float16_t>() : nullptr;
      lite::arm::math::fp16::decode_center_size_axis_1(var_size,
                                                       row,
                                                       col,
                                                       len,
                                                       target_box_data,
                                                       prior_box_data,
                                                       prior_box_var_data,
                                                       normalized,
                                                       variance,
                                                       output);
    }
  } else {
    LOG(FATAL) << "box_coder don't support this code_type: " << code_type;
  }
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
