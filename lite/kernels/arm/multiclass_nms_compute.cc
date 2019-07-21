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

#include "lite/kernels/arm/multiclass_nms_compute.h"
#include <string>
#include "lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void MulticlassNmsCompute::Run() {
  auto& param = Param<operators::MulticlassNmsParam>();
  const float* bbox_data = param.bbox_data->data<float>();
  const float* conf_data = param.conf_data->data<float>();
  // float* Out = param.out->mutable_data<float>();

  CHECK_EQ(param.bbox_data->dims().production() % 4, 0);

  std::vector<float> result;
  std::vector<int> priors = param.priors;
  int class_num = param.class_num;
  int background_id = param.background_id;
  int keep_topk = param.keep_topk;
  int nms_topk = param.nms_topk;
  float conf_thresh = param.conf_thresh;
  float nms_thresh = param.nms_thresh;
  float nms_eta = param.nms_eta;
  bool share_location = param.share_location;

  lite::arm::math::multiclass_nms(bbox_data,
                                  conf_data,
                                  &result,
                                  priors,
                                  class_num,
                                  background_id,
                                  keep_topk,
                                  nms_topk,
                                  conf_thresh,
                                  nms_thresh,
                                  nms_eta,
                                  share_location);
  param.out->Resize({static_cast<int64_t>(result.size() / 7), 7});
  float* out = param.out->mutable_data<float>();

  std::memcpy(out, result.data(), sizeof(float) * result.size());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(multiclass_nms,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::MulticlassNmsCompute,
                     def)
    .BindInput("Bbox", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Conf", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
