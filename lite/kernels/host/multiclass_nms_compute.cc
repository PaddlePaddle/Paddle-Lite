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

#include "lite/kernels/host/multiclass_nms_compute.h"
#include <map>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename dtype>
static bool sort_score_pair_descend(const std::pair<float, dtype>& pair1,
                                    const std::pair<float, dtype>& pair2) {
  return pair1.first > pair2.first;
}

template <typename dtype>
void get_max_score_index(const dtype* scores,
                         int num,
                         float threshold,
                         int top_k,
                         std::vector<std::pair<dtype, int>>* score_index_vec) {
  //! Generate index score pairs.
  for (int i = 0; i < num; ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  //! Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec->begin(),
                   score_index_vec->end(),
                   sort_score_pair_descend<int>);

  //! Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

template <typename dtype>
dtype bbox_size(const dtype* bbox, bool normalized = true) {
  if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return dtype(0.);
  } else {
    const dtype width = bbox[2] - bbox[0];
    const dtype height = bbox[3] - bbox[1];

    if (normalized) {
      return width * height;
    } else {
      // If bbox is not within range [0, 1].
      return (width + 1) * (height + 1);
    }
  }
}

template <typename dtype>
dtype jaccard_overlap(const dtype* bbox1, const dtype* bbox2) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] || bbox2[1] > bbox1[3] ||
      bbox2[3] < bbox1[1]) {
    return dtype(0.);
  } else {
    const dtype inter_xmin = std::max(bbox1[0], bbox2[0]);
    const dtype inter_ymin = std::max(bbox1[1], bbox2[1]);
    const dtype inter_xmax = std::min(bbox1[2], bbox2[2]);
    const dtype inter_ymax = std::min(bbox1[3], bbox2[3]);

    const dtype inter_width = inter_xmax - inter_xmin;
    const dtype inter_height = inter_ymax - inter_ymin;
    const dtype inter_size = inter_width * inter_height;

    const dtype bbox1_size = bbox_size(bbox1);
    const dtype bbox2_size = bbox_size(bbox2);

    return inter_size / (bbox1_size + bbox2_size - inter_size);
  }
}

template <typename dtype>
void apply_nms_fast(const dtype* bboxes,
                    const dtype* scores,
                    int num,
                    float score_threshold,
                    float nms_threshold,
                    float eta,
                    int top_k,
                    std::vector<int>* indices) {
  // Get top_k scores (with corresponding indices).
  std::vector<std::pair<dtype, int>> score_index_vec;
  get_max_score_index(scores, num, score_threshold, top_k, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices->clear();

  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    bool keep = true;

    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        float overlap =
            jaccard_overlap(bboxes + idx * 4, bboxes + kept_idx * 4);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }

    if (keep) {
      indices->push_back(idx);
    }

    score_index_vec.erase(score_index_vec.begin());

    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

template <typename dtype>
void multiclass_nms(const dtype* bbox_cpu_data,
                    const dtype* conf_cpu_data,
                    std::vector<dtype>* result,
                    const std::vector<int>& priors,
                    int class_num,
                    int background_id,
                    int keep_topk,
                    int nms_topk,
                    float conf_thresh,
                    float nms_thresh,
                    float nms_eta,
                    bool share_location) {
  int num_kept = 0;
  std::vector<std::map<int, std::vector<int>>> all_indices;
  int64_t conf_offset = 0;
  int64_t bbox_offset = 0;
  for (int i = 0; i < priors.size(); ++i) {
    std::map<int, std::vector<int>> indices;
    int num_det = 0;
    int num_priors = priors[i];

    int conf_idx = class_num * conf_offset;
    int bbox_idx =
        share_location ? bbox_offset * 4 : bbox_offset * 4 * class_num;

    for (int c = 0; c < class_num; ++c) {
      if (c == background_id) {
        // Ignore background class
        continue;
      }

      const dtype* cur_conf_data = conf_cpu_data + conf_idx + c * num_priors;
      const dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;

      if (!share_location) {
        cur_bbox_data += c * num_priors * 4;
      }

      apply_nms_fast(cur_bbox_data,
                     cur_conf_data,
                     num_priors,
                     conf_thresh,
                     nms_thresh,
                     nms_eta,
                     nms_topk,
                     &(indices[c]));
      num_det += indices[c].size();
    }

    if (keep_topk > -1 && num_det > keep_topk) {
      std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;

      for (auto it = indices.begin(); it != indices.end(); ++it) {
        int label = it->first;
        const std::vector<int>& label_indices = it->second;

        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          float score = conf_cpu_data[conf_idx + label * num_priors + idx];
          score_index_pairs.push_back(
              std::make_pair(score, std::make_pair(label, idx)));
        }
      }

      // Keep top k results per image.
      std::stable_sort(score_index_pairs.begin(),
                       score_index_pairs.end(),
                       sort_score_pair_descend<std::pair<int, int>>);
      score_index_pairs.resize(keep_topk);
      // Store the new indices.
      std::map<int, std::vector<int>> new_indices;

      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }

      all_indices.push_back(new_indices);
      num_kept += keep_topk;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
    conf_offset += num_priors;
    bbox_offset += num_priors;
  }

  if (num_kept == 0) {
    (*result).clear();
    return;
  } else {
    (*result).resize(num_kept * 7);
  }

  int count = 0;

  conf_offset = 0;
  bbox_offset = 0;
  for (int i = 0; i < priors.size(); ++i) {
    int num_priors = priors[i];
    int conf_idx = class_num * conf_offset;
    int bbox_idx =
        share_location ? bbox_offset * 4 : bbox_offset * 4 * class_num;

    for (auto it = all_indices[i].begin(); it != all_indices[i].end(); ++it) {
      int label = it->first;
      std::vector<int>& indices = it->second;
      const dtype* cur_conf_data =
          conf_cpu_data + conf_idx + label * num_priors;
      const dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;

      if (!share_location) {
        cur_bbox_data += label * num_priors * 4;
      }

      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        (*result)[count * 7] = i;
        (*result)[count * 7 + 1] = label;
        (*result)[count * 7 + 2] = cur_conf_data[idx];

        for (int k = 0; k < 4; ++k) {
          (*result)[count * 7 + 3 + k] = cur_bbox_data[idx * 4 + k];
        }

        ++count;
      }
    }
    conf_offset += num_priors;
    bbox_offset += num_priors;
  }
}

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

  multiclass_nms(bbox_data,
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

  lite::LoD lod;
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
  lod.push_back(lod_info);
  if (result_corrected.empty()) {
    lod.clear();
    lod.push_back(std::vector<uint64_t>({0, 1}));
    param.out->Resize({static_cast<int64_t>(1)});
    param.out->mutable_data<float>()[0] = -1.;
    param.out->set_lod(lod);
  } else {
    param.out->Resize({static_cast<int64_t>(result_corrected.size() / 6), 6});
    float* out = param.out->mutable_data<float>();
    std::memcpy(
        out, result_corrected.data(), sizeof(float) * result_corrected.size());
    param.out->set_lod(lod);
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(multiclass_nms,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::MulticlassNmsCompute,
                     def)
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
