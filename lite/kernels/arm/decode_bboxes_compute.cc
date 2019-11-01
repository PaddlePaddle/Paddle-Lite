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

#include "lite/kernels/arm/decode_bboxes_compute.h"
#include <string>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void DecodeBboxesCompute::Run() {
  auto& param = Param<operators::DecodeBboxesParam>();
  const float* loc_data = param.loc_data->data<float>();
  const float* prior_data = param.prior_data->data<float>();
  float* bbox_data = param.bbox_data->mutable_data<float>();

  // CHECK_EQ(param.loc_data->dims(), 2); // loc_data {N, boxes * 4}
  // CHECK_EQ(param.prior_data->dims(), 3); // prior_data {1, 2, boxes * 4(xmin,
  // ymin, xmax, ymax)}

  int batch_num = param.batch_num;
  int num_priors = param.num_priors;
  int num_loc_classes = param.num_loc_classes;
  int background_label_id = param.background_label_id;
  bool share_location = param.share_location;
  bool variance_encoded_in_target = param.variance_encoded_in_target;
  std::string code_type = param.code_type;

  lite::arm::math::decode_bboxes(batch_num,
                                 loc_data,
                                 prior_data,
                                 code_type,
                                 variance_encoded_in_target,
                                 num_priors,
                                 share_location,
                                 num_loc_classes,
                                 background_label_id,
                                 bbox_data);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(decode_bboxes,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::DecodeBboxesCompute,
                     def)
    .BindInput("Loc", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Prior", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Bbox", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
