/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef MULTICLASSNMS_OP
#pragma once

#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include "framework/tensor.h"
#include "operators/math/poly_util.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <class T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

template <class T>
static inline void GetMaxScoreIndex(
    const std::vector<T>& scores, const T threshold, int top_k,
    std::vector<std::pair<T, int>>* sorted_indices) {
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      sorted_indices->push_back(std::make_pair(scores[i], i));
    }
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices->begin(), sorted_indices->end(),
                   SortScorePairDescend<int>);
  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < static_cast<int>(sorted_indices->size())) {
    sorted_indices->resize(top_k);
  }
}

template <class T>
static inline T BBoxArea(const T* box, const bool normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return static_cast<T>(0.);
  } else {
    const T w = box[2] - box[0];
    const T h = box[3] - box[1];
    if (normalized) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}

template <class T>
static inline T JaccardOverlap(const T* box1, const T* box2,
                               const bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<T>(0.);
  } else {
    const T inter_xmin = std::max(box1[0], box2[0]);
    const T inter_ymin = std::max(box1[1], box2[1]);
    const T inter_xmax = std::min(box1[2], box2[2]);
    const T inter_ymax = std::min(box1[3], box2[3]);
    const T inter_w = inter_xmax - inter_xmin;
    const T inter_h = inter_ymax - inter_ymin;
    const T inter_area = inter_w * inter_h;
    const T bbox1_area = BBoxArea<T>(box1, normalized);
    const T bbox2_area = BBoxArea<T>(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <class T>
static inline T PolyIoU(const T* box1, const T* box2, const size_t box_size,
                        const bool normalized) {
  T bbox1_area = math::PolyArea<T>(box1, box_size, normalized);
  T bbox2_area = math::PolyArea<T>(box2, box_size, normalized);
  T inter_area = math::PolyOverlapArea<T>(box1, box2, box_size, normalized);
  if (bbox1_area == 0 || bbox2_area == 0 || inter_area == 0) {
    // If coordinate values are is invalid
    // if area size <= 0,  return 0.
    return static_cast<T>(0.);
  } else {
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <typename T>
static inline void NMSFast(const framework::Tensor& bbox,
                           const framework::Tensor& scores,
                           const T score_threshold, const T nms_threshold,
                           const T eta, const int64_t top_k,
                           std::vector<int>* selected_indices) {
  // The total boxes for each instance.
  int64_t num_boxes = bbox.dims()[0];
  // 4: [xmin ymin xmax ymax]
  int64_t box_size = bbox.dims()[1];

  std::vector<T> scores_data(num_boxes);
  std::copy_n(scores.data<T>(), num_boxes, scores_data.begin());
  std::vector<std::pair<T, int>> sorted_indices;
  GetMaxScoreIndex(scores_data, score_threshold, top_k, &sorted_indices);

  selected_indices->clear();
  T adaptive_threshold = nms_threshold;
  const T* bbox_data = bbox.data<T>();

  while (sorted_indices.size() != 0) {
    const int idx = sorted_indices.front().second;
    bool keep = true;
    for (size_t k = 0; k < selected_indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*selected_indices)[k];
        T overlap = T(0.);
        if (box_size == 4) {
          overlap = JaccardOverlap<T>(bbox_data + idx * box_size,
                                      bbox_data + kept_idx * box_size, true);
        } else {
          overlap = PolyIoU<T>(bbox_data + idx * box_size,
                               bbox_data + kept_idx * box_size, box_size, true);
        }
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      selected_indices->push_back(idx);
    }
    sorted_indices.erase(sorted_indices.begin());
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

template <typename T>
void MultiClassNMS(const framework::Tensor& scores,
                   const framework::Tensor& bboxes,
                   std::map<int, std::vector<int>>* indices, int* num_nmsed_out,
                   const int& background_label, const int& nms_top_k,
                   const int& keep_top_k, const T& nms_threshold,
                   const T& nms_eta, const T& score_threshold) {
  int64_t class_num = scores.dims()[0];
  int64_t predict_dim = scores.dims()[1];
  int num_det = 0;
  for (int64_t c = 0; c < class_num; ++c) {
    if (c == background_label) continue;
    framework::Tensor score = scores.Slice(c, c + 1);
    /// [c] is key
    NMSFast<float>(bboxes, score, score_threshold, nms_threshold, nms_eta,
                   nms_top_k, &((*indices)[c]));
    num_det += (*indices)[c].size();
  }

  *num_nmsed_out = num_det;
  const T* scores_data = scores.data<T>();
  if (keep_top_k > -1 && num_det > keep_top_k) {
    std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
    for (const auto& it : *indices) {
      int label = it.first;
      const T* sdata = scores_data + label * predict_dim;
      const std::vector<int>& label_indices = it.second;
      for (size_t j = 0; j < label_indices.size(); ++j) {
        int idx = label_indices[j];
        // PADDLE_ENFORCE_LT(idx, predict_dim);
        score_index_pairs.push_back(
            std::make_pair(sdata[idx], std::make_pair(label, idx)));
      }
    }
    // Keep top k results per image.
    std::stable_sort(score_index_pairs.begin(), score_index_pairs.end(),
                     SortScorePairDescend<std::pair<int, int>>);
    score_index_pairs.resize(keep_top_k);

    // Store the new indices.
    std::map<int, std::vector<int>> new_indices;
    for (size_t j = 0; j < score_index_pairs.size(); ++j) {
      int label = score_index_pairs[j].second.first;
      int idx = score_index_pairs[j].second.second;
      new_indices[label].push_back(idx);
    }
    new_indices.swap(*indices);
    *num_nmsed_out = keep_top_k;
  }
}

template <typename T>
void MultiClassOutput(const framework::Tensor& scores,
                      const framework::Tensor& bboxes,
                      const std::map<int, std::vector<int>>& selected_indices,
                      framework::Tensor* outs) {
  int predict_dim = scores.dims()[1];
  int box_size = bboxes.dims()[1];
  int out_dim = bboxes.dims()[1] + 2;
  auto* scores_data = scores.data<T>();
  auto* bboxes_data = bboxes.data<T>();
  auto* odata = outs->data<T>();

  int count = 0;
  for (const auto& it : selected_indices) {
    /// one batch
    int label = it.first;
    const T* sdata = scores_data + label * predict_dim;
    const std::vector<int>& indices = it.second;
    for (size_t j = 0; j < indices.size(); ++j) {
      int idx = indices[j];
      const T* bdata = bboxes_data + idx * box_size;
      odata[count * out_dim] = label;           // label
      odata[count * out_dim + 1] = sdata[idx];  // score
      // xmin, ymin, xmax, ymax
      std::memcpy(odata + count * out_dim + 2, bdata, box_size * sizeof(T));
      count++;
    }
  }
}

template <typename P>
void MultiClassNMSCompute(const MultiClassNMSParam<CPU>& param) {
  const auto* input_bboxes = param.InputBBoxes();
  const auto& input_bboxes_dims = input_bboxes->dims();

  const auto* input_scores = param.InputScores();
  const auto& input_scores_dims = input_scores->dims();

  auto* outs = param.Out();
  auto background_label = param.BackGroundLabel();
  auto nms_top_k = param.NMSTopK();
  auto keep_top_k = param.KeepTopK();
  auto nms_threshold = param.NMSThreshold();
  auto nms_eta = param.NMSEta();
  auto score_threshold = param.ScoreThreshold();

  int64_t batch_size = input_scores_dims[0];
  int64_t class_num = input_scores_dims[1];
  int64_t predict_dim = input_scores_dims[2];
  int64_t box_dim = input_bboxes_dims[2];

  std::vector<std::map<int, std::vector<int>>> all_indices;
  std::vector<size_t> batch_starts = {0};
  for (int64_t i = 0; i < batch_size; ++i) {
    framework::Tensor ins_score = input_scores->Slice(i, i + 1);
    ins_score.Resize({class_num, predict_dim});

    framework::Tensor ins_boxes = input_bboxes->Slice(i, i + 1);
    ins_boxes.Resize({predict_dim, box_dim});

    std::map<int, std::vector<int>> indices;
    int num_nmsed_out = 0;
    MultiClassNMS<float>(ins_score, ins_boxes, &indices, &num_nmsed_out,
                         background_label, nms_top_k, keep_top_k, nms_threshold,
                         nms_eta, score_threshold);
    all_indices.push_back(indices);
    batch_starts.push_back(batch_starts.back() + num_nmsed_out);
  }

  int num_kept = batch_starts.back();
  if (num_kept == 0) {
    float* od = outs->mutable_data<float>({1});
    od[0] = -1;
  } else {
    int64_t out_dim = box_dim + 2;
    outs->mutable_data<float>({num_kept, out_dim});
    for (int64_t i = 0; i < batch_size; ++i) {
      framework::Tensor ins_score = input_scores->Slice(i, i + 1);
      ins_score.Resize({class_num, predict_dim});

      framework::Tensor ins_boxes = input_bboxes->Slice(i, i + 1);
      ins_boxes.Resize({predict_dim, box_dim});

      int64_t s = batch_starts[i];
      int64_t e = batch_starts[i + 1];
      if (e > s) {
        framework::Tensor out = outs->Slice(s, e);
        MultiClassOutput<float>(ins_score, ins_boxes, all_indices[i], &out);
      }
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
