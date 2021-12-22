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

#include "lite/kernels/host/collect_fpn_proposals_compute.h"
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

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
  int num_size = param.multi_rois_num.size();
  for (size_t i = 0; i < num_fpn_level; ++i) {
    int all_rois = 0;
    if (num_size == 0) {
      auto cur_rois_lod = multi_layer_rois[i]->lod().back();
      all_rois = cur_rois_lod[cur_rois_lod.size() - 1];
    } else {
      const int* cur_rois_num = param.multi_rois_num[i]->data<int>();
      all_rois = std::accumulate(
          cur_rois_num, cur_rois_num + param.multi_rois_num[i]->numel(), 0);
    }
    integral_of_all_rois[i + 1] = integral_of_all_rois[i] + all_rois;
  }
  const int batch_size = (num_size == 0)
                             ? multi_layer_rois[0]->lod().back().size() - 1
                             : param.multi_rois_num[0]->numel();
  std::vector<ScoreWithID> scores_of_all_rois(
      integral_of_all_rois[num_fpn_level], ScoreWithID());
  for (int i = 0; i < num_fpn_level; ++i) {
    const float* cur_level_scores = multi_layer_scores[i]->data<float>();
    int cur_level_num = integral_of_all_rois[i + 1] - integral_of_all_rois[i];
    auto cur_scores_lod = multi_layer_scores[i]->lod().back();
    int cur_batch_id = 0;
    int pre_num = 0;
    for (int j = 0; j < cur_level_num; ++j) {
      if (num_size == 0) {
        auto cur_scores_lod = multi_layer_scores[i]->lod().back();
        if (static_cast<size_t>(j) >= cur_scores_lod[cur_batch_id + 1]) {
          cur_batch_id++;
        }
      } else {
        const int* rois_num_data = param.multi_rois_num[i]->data<int>();
        if (j >= pre_num + rois_num_data[cur_batch_id]) {
          pre_num += rois_num_data[cur_batch_id];
          cur_batch_id++;
        }
      }
      int cur_index = j + integral_of_all_rois[i];
      scores_of_all_rois[cur_index].score = cur_level_scores[j];
      scores_of_all_rois[cur_index].index = j;
      scores_of_all_rois[cur_index].level = i;
      scores_of_all_rois[cur_index].batch_id = cur_batch_id;
    }
  }

  // keep top post_nms_topN rois, sort the rois by the score
  post_nms_topN = std::min(post_nms_topN, integral_of_all_rois[num_fpn_level]);
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
  fpn_rois->Resize({post_nms_topN, kBoxDim});
  auto fpn_rois_data = fpn_rois->mutable_data<float>();
  std::vector<uint64_t> lod0(1, 0);
  int cur_batch_id = 0;
  std::vector<int64_t> num_per_batch;
  int pre_idx = 0;
  int cur_num = 0;
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
      cur_num = i - pre_idx;
      pre_idx = i;
      num_per_batch.emplace_back(cur_num);
    }
  }
  num_per_batch.emplace_back(post_nms_topN - pre_idx);
  if (param.rois_num) {
    int* rois_num_data = param.rois_num->mutable_data<int>();
    for (int i = 0; i < batch_size; i++) {
      rois_num_data[i] = num_per_batch[i];
    }
  }
  lod0.emplace_back(post_nms_topN);
  lite::LoD lod;
  lod.emplace_back(lod0);
  fpn_rois->set_lod(lod);

  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(collect_fpn_proposals,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::CollectFpnProposalsCompute,
                     def)
    .BindInput("MultiLevelRois", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("MultiLevelScores", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("MultiLevelRoIsNum", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("FpnRois", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("RoisNum", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindPaddleOpVersion("collect_fpn_proposals", 1)
    .Finalize();
