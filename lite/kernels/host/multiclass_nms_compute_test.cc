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
#include <gtest/gtest.h>
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
void multiclass_nms_compute_ref(const operators::MulticlassNmsParam& param,
                                int class_num,
                                const std::vector<int>& priors,
                                bool share_location,
                                std::vector<float>* result) {
  int background_id = param.background_label;
  int keep_topk = param.keep_top_k;
  int nms_topk = param.nms_top_k;
  float conf_thresh = param.score_threshold;
  float nms_thresh = param.nms_threshold;
  float nms_eta = param.nms_eta;
  const dtype* bbox_data = param.bboxes->data<const dtype>();
  const dtype* conf_data = param.scores->data<const dtype>();
  dtype* out = param.out->mutable_data<dtype>();
  (*result).clear();

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

      const dtype* cur_conf_data = conf_data + conf_idx + c * num_priors;
      const dtype* cur_bbox_data = bbox_data + bbox_idx;

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
          float score = conf_data[conf_idx + label * num_priors + idx];
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
    (*result).resize(1);
    (*result)[0] = -1;
    return;
  } else {
    (*result).resize(num_kept * 6);
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
      const dtype* cur_conf_data = conf_data + conf_idx + label * num_priors;
      const dtype* cur_bbox_data = bbox_data + bbox_idx;

      if (!share_location) {
        cur_bbox_data += label * num_priors * 4;
      }

      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        (*result)[count * 6] = label;
        (*result)[count * 6 + 1] = cur_conf_data[idx];

        for (int k = 0; k < 4; ++k) {
          (*result)[count * 6 + 2 + k] = cur_bbox_data[idx * 4 + k];
        }

        ++count;
      }
    }
    conf_offset += num_priors;
    bbox_offset += num_priors;
  }
}

TEST(multiclass_nms_host, init) {
  MulticlassNmsCompute multiclass_nms;
  ASSERT_EQ(multiclass_nms.precision(), PRECISION(kFloat));
  ASSERT_EQ(multiclass_nms.target(), TARGET(kHost));
}

TEST(multiclass_nms_host, retrive_op) {
  auto multiclass_nms =
      KernelRegistry::Global().Create<TARGET(kHost), PRECISION(kFloat)>(
          "multiclass_nms");
  ASSERT_FALSE(multiclass_nms.empty());
  ASSERT_TRUE(multiclass_nms.front());
}

TEST(multiclass_nms_host, compute) {
  MulticlassNmsCompute multiclass_nms;
  operators::MulticlassNmsParam param;
  lite::Tensor bbox, conf, out;
  std::vector<float> out_ref;

  for (std::vector<int> priors : {std::vector<int>({2, 2, 2})}) {
    int N = priors.size();
    for (bool share_location : {true}) {
      for (int class_num : {1, 4, 10}) {
        DDim* bbox_dim;
        DDim* conf_dim;
        int M = priors[0];
        if (share_location) {
          bbox_dim = new DDim({N, M, 4});
        } else {
          bbox_dim = new DDim({class_num, M, 4});
        }
        conf_dim = new DDim({N, class_num, M});
        bbox.Resize(*bbox_dim);
        conf.Resize(*conf_dim);
        for (int background_id : {0}) {
          for (int keep_topk : {1, 5, 10}) {
            for (int nms_topk : {1, 5, 10}) {
              for (float nms_eta : {1.0, 0.99, 0.9}) {
                for (float nms_thresh : {0.5, 0.7}) {
                  for (float conf_thresh : {0.5, 0.7}) {
                    auto* conf_data = conf.mutable_data<float>();
                    auto* bbox_data = bbox.mutable_data<float>();
                    for (int i = 0; i < bbox_dim->production(); ++i) {
                      bbox_data[i] = i * 1. / bbox_dim->production();
                    }
                    for (int i = 0; i < conf_dim->production(); ++i) {
                      conf_data[i] = i * 1. / conf_dim->production();
                    }
                    param.bboxes = &bbox;
                    param.scores = &conf;
                    param.out = &out;
                    param.background_label = background_id;
                    param.keep_top_k = keep_topk;
                    param.nms_top_k = nms_topk;
                    param.score_threshold = conf_thresh;
                    param.nms_threshold = nms_thresh;
                    param.nms_eta = nms_eta;
                    multiclass_nms.SetParam(param);
                    multiclass_nms.Run();
                    auto* out_data = out.mutable_data<float>();
                    out_ref.clear();
                    multiclass_nms_compute_ref<float>(
                        param, class_num, priors, share_location, &out_ref);
                    EXPECT_EQ(out.dims().production(), out_ref.size());
                    if (out.dims().production() == out_ref.size()) {
                      auto* out_ref_data = out_ref.data();
                      for (int i = 0; i < out.dims().production(); i++) {
                        EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
                      }
                    }
                  }
                }
              }
            }
          }
        }
        delete bbox_dim;
        delete conf_dim;
      }
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(multiclass_nms, kHost, kFloat, kNCHW, def);
