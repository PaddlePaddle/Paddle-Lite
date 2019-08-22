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
  // bbox shape : N, M, 4
  // scores shape : N, C, M
  const float* bbox_data = param.bbox_data->data<float>();
  const float* conf_data = param.conf_data->data<float>();

  CHECK_EQ(param.bbox_data->dims().production() % 4, 0);

  std::vector<float> result;
  int N = param.bbox_data->dims()[0];
  int M = param.bbox_data->dims()[1];
  std::vector<int> priors(N, M);
  int class_num = param.conf_data->dims()[1];
  int background_label = param.background_label;
  int keep_top_k = param.keep_top_k;
  int nms_top_k = param.nms_top_k;
  float score_threshold = param.score_threshold;
  float nms_threshold = param.nms_threshold;
  float nms_eta = param.nms_eta;
  bool share_location = param.share_location;

  lite::arm::math::multiclass_nms(bbox_data,
                                  conf_data,
                                  &result,
                                  priors,
                                  class_num,
                                  background_label,
                                  keep_top_k,
                                  nms_top_k,
                                  score_threshold,
                                  nms_threshold,
                                  nms_eta,
                                  share_location);
  lite::LoD* lod = param.out->mutable_lod();
  std::vector<uint64_t> lod_info;
  lod_info.push_back(0);
  std::vector<float> result_corrected;
  int tmp_batch_id;
  uint64_t num = 0;
  for (int i = 0; i < result.size(); ++i) {
    if (i == 0) {
      tmp_batch_id = result[i];
    }
    if (i % 7 == 0) {
      if (result[i] == tmp_batch_id) {
        ++num;
      } else {
        lod_info.push_back(num);
        ++num;
        tmp_batch_id = result[i];
      }
    } else {
      result_corrected.push_back(result[i]);
    }
  }
  lod_info.push_back(num);
  (*lod).push_back(lod_info);

  if (result_corrected.empty()) {
    (*lod).clear();
    (*lod).push_back(std::vector<uint64_t>({0, 1}));
    param.out->Resize({static_cast<int64_t>(1)});
    param.out->mutable_data<float>()[0] = -1.;
  } else {
    param.out->Resize({static_cast<int64_t>(result_corrected.size() / 6), 6});
    float* out = param.out->mutable_data<float>();
    std::memcpy(
        out, result_corrected.data(), sizeof(float) * result_corrected.size());
  }
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
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
