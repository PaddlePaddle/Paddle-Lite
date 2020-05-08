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

#include "lite/kernels/arm/collect_fpn_proposals_compute.h"
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

struct ScoreWithID {
  float score;
  int batch_id;
  int index;
  int level;
  ScoreWithID() {
    batch_id = -1;
    index = -1;
    level = -1;
  }
  ScoreWithID(float score_, int batch_id_, int index_, int level_) {
    score = score_;
    batch_id = batch_id_;
    index = index_;
    level = level_;
  }
};

static inline bool CompareByScore(ScoreWithID a, ScoreWithID b) {
  return a.score >= b.score;
}

static inline bool CompareByBatchid(ScoreWithID a, ScoreWithID b) {
  return a.batch_id < b.batch_id;
}

void CollectFpnProposalsCompute::Run() {
  auto& param = Param<operators::CollectFpnProposalsParam>();
  auto multi_layer_rois = param.multi_level_rois;
  auto multi_layer_scores = param.multi_level_scores;
  auto* fpn_rois = param.fpn_rois;
  int post_nms_topN = param.post_nms_topN;

  if (multi_layer_rois.size() != multi_layer_scores.size()) {
    LOG(FATAL) << "multi_layer_rois.size() should be equan to "
                  "multi_layer_scores.size()";
  }

  size_t num_fpn_level = multi_layer_rois.size();
  std::vector<int> integral_of_all_rois(num_fpn_level + 1, 0);
  for (size_t i = 0; i < num_fpn_level; ++i) {
    auto cur_rois_lod = multi_layer_rois[i]->lod().back();
    integral_of_all_rois[i + 1] = static_cast<int>(
        integral_of_all_rois[i] + cur_rois_lod[cur_rois_lod.size() - 1]);
  }

  std::vector<ScoreWithID> scores_of_all_rois(
      integral_of_all_rois[num_fpn_level], ScoreWithID());
  for (int i = 0; i < num_fpn_level; ++i) {
    const float* cur_level_scores = multi_layer_scores[i]->data<float>();
    int cur_level_num = integral_of_all_rois[i + 1] - integral_of_all_rois[i];
    auto cur_scores_lod = multi_layer_scores[i]->lod().back();
    int cur_batch_id = 0;
    for (int j = 0; j < cur_level_num; ++j) {
      if (j >= cur_scores_lod[cur_batch_id + 1]) {
        cur_batch_id++;
      }
      int cur_index = j + integral_of_all_rois[i];
      scores_of_all_rois[cur_index].score = cur_level_scores[j];
      scores_of_all_rois[cur_index].index = j;
      scores_of_all_rois[cur_index].level = i;
      scores_of_all_rois[cur_index].batch_id = cur_batch_id;
    }
  }

  // keep top post_nms_topN rois, sort the rois by the score
  if (post_nms_topN > integral_of_all_rois[num_fpn_level]) {
    post_nms_topN = integral_of_all_rois[num_fpn_level];
  }
  std::stable_sort(
      scores_of_all_rois.begin(), scores_of_all_rois.end(), CompareByScore);
  scores_of_all_rois.resize(post_nms_topN);
  // sort by batch id
  std::stable_sort(
      scores_of_all_rois.begin(), scores_of_all_rois.end(), CompareByBatchid);
  // create a pointer array
  std::vector<const float*> multi_fpn_rois_data(num_fpn_level);
  for (int i = 0; i < num_fpn_level; ++i) {
    multi_fpn_rois_data[i] = multi_layer_rois[i]->data<float>();
  }

  // initialize the outputs
  const int kBoxDim = 4;
  auto fpn_rois_data = fpn_rois->mutable_data<float>();
  std::vector<uint64_t> lod0(1, 0);
  int cur_batch_id = 0;
  for (int i = 0; i < post_nms_topN; ++i) {
    int cur_fpn_level = scores_of_all_rois[i].level;
    int cur_level_index = scores_of_all_rois[i].index;
    std::memcpy(fpn_rois_data,
                multi_fpn_rois_data[cur_fpn_level] + cur_level_index * kBoxDim,
                kBoxDim * sizeof(float));
    fpn_rois_data += kBoxDim;
    if (scores_of_all_rois[i].batch_id != cur_batch_id) {
      cur_batch_id = scores_of_all_rois[i].batch_id;
      lod0.emplace_back(i);
    }
  }
  lod0.emplace_back(post_nms_topN);
  lite::LoD lod;
  lod.emplace_back(lod0);
  fpn_rois->set_lod(lod);
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(collect_fpn_proposals,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::CollectFpnProposalsCompute,
                     def)
    .BindInput("MultiLevelRois", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("MultiLevelScores", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("FpnRois", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
