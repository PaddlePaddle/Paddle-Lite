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

#pragma once
#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include "lite/backends/host/math/nms_util.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace host {

inline std::vector<uint64_t> GetNmsLodFromRoisNum(const Tensor* rois_num) {
  std::vector<uint64_t> rois_lod;
  auto* rois_num_data = rois_num->data<int>();
  rois_lod.push_back(static_cast<uint64_t>(0));
  for (int i = 0; i < rois_num->numel(); ++i) {
    rois_lod.push_back(rois_lod.back() +
                       static_cast<uint64_t>(rois_num_data[i]));
  }
  return rois_lod;
}

template <class T>
void SliceOneClass(const Tensor& items,
                   const int class_id,
                   Tensor* one_class_item) {
  T* item_data = one_class_item->template mutable_data<T>();
  const T* items_data = items.data<T>();
  const int64_t num_item = items.dims()[0];
  const int64_t class_num = items.dims()[1];
  if (items.dims().size() == 3) {
    int64_t item_size = items.dims()[2];
    for (int i = 0; i < num_item; ++i) {
      std::memcpy(item_data + i * item_size,
                  items_data + i * class_num * item_size + class_id * item_size,
                  sizeof(T) * item_size);
    }
  } else {
    for (int i = 0; i < num_item; ++i) {
      item_data[i] = items_data[i * class_num + class_id];
    }
  }
}

template <typename T>
void NMSFast(const Tensor& bbox,
             const Tensor& scores,
             const T score_threshold,
             const T nms_threshold,
             const T eta,
             const int64_t top_k,
             std::vector<int>* selected_indices,
             const bool normalized) {
  // The total boxes for each instance.
  int64_t num_boxes = bbox.dims()[0];
  // 4: [xmin ymin xmax ymax]
  // 8: [x1 y1 x2 y2 x3 y3 x4 y4]
  // 16, 24, or 32: [x1 y1 x2 y2 ...  xn yn], n = 8, 12 or 16
  int64_t box_size = bbox.dims()[1];

  std::vector<T> scores_data(num_boxes);
  std::copy_n(scores.data<T>(), num_boxes, scores_data.begin());
  std::vector<std::pair<T, int>> sorted_indices;
  lite::host::math::GetMaxScoreIndex(
      scores_data, score_threshold, top_k, &sorted_indices);

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
        // 4: [xmin ymin xmax ymax]
        if (box_size == 4) {
          overlap = lite::host::math::JaccardOverlap<T>(
              bbox_data + idx * box_size,
              bbox_data + kept_idx * box_size,
              normalized);
        }
        // 8: [x1 y1 x2 y2 x3 y3 x4 y4] or 16, 24, 32
        if (box_size == 8 || box_size == 16 || box_size == 24 ||
            box_size == 32) {
          overlap =
              lite::host::math::PolyIoU<T>(bbox_data + idx * box_size,
                                           bbox_data + kept_idx * box_size,
                                           box_size,
                                           normalized);
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
void MultiClassNMS(const operators::MulticlassNmsParam& param,
                   const Tensor& scores,
                   const Tensor& bboxes,
                   const int scores_size,
                   std::map<int, std::vector<int>>* indices,
                   int* num_nmsed_out) {
  int64_t background_label = param.background_label;
  int64_t nms_top_k = param.nms_top_k;
  int64_t keep_top_k = param.keep_top_k;
  bool normalized = param.normalized;
  T nms_threshold = static_cast<T>(param.nms_threshold);
  T nms_eta = static_cast<T>(param.nms_eta);
  T score_threshold = static_cast<T>(param.score_threshold);

  int num_det = 0;

  int64_t class_num = scores_size == 3 ? scores.dims()[0] : scores.dims()[1];
  Tensor bbox_slice, score_slice;
  for (int64_t c = 0; c < class_num; ++c) {
    if (c == background_label) continue;
    if (scores_size == 3) {
      score_slice = scores.Slice<T>(c, c + 1);
      bbox_slice = bboxes;
    } else {
      score_slice.Resize({scores.dims()[0], 1});
      bbox_slice.Resize({scores.dims()[0], 4});
      SliceOneClass<T>(scores, c, &score_slice);
      SliceOneClass<T>(bboxes, c, &bbox_slice);
    }
    NMSFast(bbox_slice,
            score_slice,
            score_threshold,
            nms_threshold,
            nms_eta,
            nms_top_k,
            &((*indices)[c]),
            normalized);
    if (scores_size == 2) {
      std::stable_sort((*indices)[c].begin(), (*indices)[c].end());
    }
    num_det += (*indices)[c].size();
  }

  *num_nmsed_out = num_det;
  const T* scores_data = scores.data<T>();
  if (keep_top_k > -1 && num_det > keep_top_k) {
    const T* sdata;
    std::vector<std::pair<T, std::pair<int, int>>> score_index_pairs;
    for (const auto& it : *indices) {
      int label = it.first;
      if (scores_size == 3) {
        sdata = scores_data + label * scores.dims()[1];
      } else {
        score_slice.Resize({scores.dims()[0], 1});
        SliceOneClass<T>(scores, label, &score_slice);
        sdata = score_slice.data<T>();
      }
      const std::vector<int>& label_indices = it.second;
      for (size_t j = 0; j < label_indices.size(); ++j) {
        int idx = label_indices[j];
        score_index_pairs.push_back(
            std::make_pair(sdata[idx], std::make_pair(label, idx)));
      }
    }
    // Keep top k results per image.
    std::stable_sort(
        score_index_pairs.begin(),
        score_index_pairs.end(),
        lite::host::math::SortScorePairDescend<std::pair<int, int>>);
    score_index_pairs.resize(keep_top_k);

    // Store the new indices.
    std::map<int, std::vector<int>> new_indices;
    for (size_t j = 0; j < score_index_pairs.size(); ++j) {
      int label = score_index_pairs[j].second.first;
      int idx = score_index_pairs[j].second.second;
      new_indices[label].push_back(idx);
    }
    if (scores_size == 2) {
      for (const auto& it : new_indices) {
        int label = it.first;
        std::stable_sort(new_indices[label].begin(), new_indices[label].end());
      }
    }
    new_indices.swap(*indices);
    *num_nmsed_out = keep_top_k;
  }
}

template <typename T>
void MultiClassOutput(const Tensor& scores,
                      const Tensor& bboxes,
                      const std::map<int, std::vector<int>>& selected_indices,
                      const int scores_size,
                      Tensor* outs,
                      int* oindices = nullptr,
                      const int offset = 0) {
  int64_t class_num = scores.dims()[1];
  int64_t predict_dim = scores.dims()[1];
  int64_t box_size = bboxes.dims()[1];
  if (scores_size == 2) {
    box_size = bboxes.dims()[2];
  }
  int64_t out_dim = box_size + 2;
  auto* scores_data = scores.data<T>();
  auto* bboxes_data = bboxes.data<T>();
  auto* odata = outs->template mutable_data<T>();
  const T* sdata;
  Tensor bbox;
  bbox.Resize({scores.dims()[0], box_size});
  int count = 0;
  for (const auto& it : selected_indices) {
    int label = it.first;
    const std::vector<int>& indices = it.second;
    if (scores_size == 2) {
      SliceOneClass<T>(bboxes, label, &bbox);
    } else {
      sdata = scores_data + label * predict_dim;
    }
    for (size_t j = 0; j < indices.size(); ++j) {
      int idx = indices[j];
      odata[count * out_dim] = label;  // label
      const T* bdata;
      if (scores_size == 3) {
        bdata = bboxes_data + idx * box_size;
        odata[count * out_dim + 1] = sdata[idx];  // score
        if (oindices != nullptr) {
          oindices[count] = offset + idx;
        }
      } else {
        bdata = bbox.data<T>() + idx * box_size;
        odata[count * out_dim + 1] = *(scores_data + idx * class_num + label);
        if (oindices != nullptr) {
          oindices[count] = offset + idx * class_num + label;
        }
      }
      // xmin, ymin, xmax, ymax or multi-points coordinates
      std::memcpy(odata + count * out_dim + 2, bdata, box_size * sizeof(T));
      count++;
    }
  }
}

template <typename T, TargetType TType, PrecisionType PType>
class MulticlassNmsCompute : public KernelLite<TType, PType> {
 public:
  void Run() {
    auto& param = this->template Param<operators::MulticlassNmsParam>();
    auto* boxes = param.bboxes;
    auto* scores = param.scores;
    auto* outs = param.out;
    bool return_index = param.index ? true : false;
    auto* index = param.index;
    auto score_dims = scores->dims();
    auto score_size = score_dims.size();
    auto has_roissum = param.rois_num != nullptr;
    auto return_rois_num = param.nms_rois_num != nullptr;
    auto rois_num = param.rois_num;

    std::vector<std::map<int, std::vector<int>>> all_indices;
    std::vector<uint64_t> batch_starts = {0};
    int64_t batch_size = score_dims[0];
    int64_t box_dim = boxes->dims()[2];
    int64_t out_dim = box_dim + 2;
    int num_nmsed_out = 0;
    Tensor boxes_slice, scores_slice;
    int n;
    if (has_roissum) {
      n = score_size == 3 ? batch_size : rois_num->numel();
    } else {
      n = score_size == 3 ? batch_size : boxes->lod().back().size() - 1;
    }
    for (int i = 0; i < n; ++i) {
      if (score_size == 3) {
        scores_slice = scores->template Slice<T>(i, i + 1);
        scores_slice.Resize({score_dims[1], score_dims[2]});
        boxes_slice = boxes->template Slice<T>(i, i + 1);
        boxes_slice.Resize({score_dims[2], box_dim});
      } else {
        std::vector<uint64_t> boxes_lod;
        if (has_roissum) {
          boxes_lod = GetNmsLodFromRoisNum(rois_num);
        } else {
          boxes_lod = boxes->lod().back();
        }
        scores_slice =
            scores->template Slice<T>(boxes_lod[i], boxes_lod[i + 1]);
        boxes_slice = boxes->template Slice<T>(boxes_lod[i], boxes_lod[i + 1]);
      }
      std::map<int, std::vector<int>> indices;
      MultiClassNMS<T>(param,
                       scores_slice,
                       boxes_slice,
                       score_size,
                       &indices,
                       &num_nmsed_out);
      all_indices.push_back(indices);
      batch_starts.push_back(batch_starts.back() + num_nmsed_out);
    }

    uint64_t num_kept = batch_starts.back();
    if (num_kept == 0) {
      if (return_index) {
        outs->Resize({0, out_dim});
        index->Resize({0, 1});
      } else {
        outs->Resize({1, 1});
        T* od = outs->template mutable_data<T>();
        od[0] = -1;
        batch_starts = {0, 1};
      }
    } else {
      outs->Resize({static_cast<int64_t>(num_kept), out_dim});
      outs->template mutable_data<T>();
      int offset = 0;
      int* oindices = nullptr;
      for (int i = 0; i < n; ++i) {
        if (score_size == 3) {
          scores_slice = scores->template Slice<T>(i, i + 1);
          boxes_slice = boxes->template Slice<T>(i, i + 1);
          scores_slice.Resize({score_dims[1], score_dims[2]});
          boxes_slice.Resize({score_dims[2], box_dim});
          if (return_index) {
            offset = i * score_dims[2];
          }
        } else {
          std::vector<uint64_t> boxes_lod;
          if (has_roissum) {
            boxes_lod = GetNmsLodFromRoisNum(rois_num);
          } else {
            boxes_lod = boxes->lod().back();
          }
          scores_slice =
              scores->template Slice<T>(boxes_lod[i], boxes_lod[i + 1]);
          boxes_slice =
              boxes->template Slice<T>(boxes_lod[i], boxes_lod[i + 1]);
          if (return_index) {
            offset = boxes_lod[i] * score_dims[1];
          }
        }
        int64_t s = static_cast<int64_t>(batch_starts[i]);
        int64_t e = static_cast<int64_t>(batch_starts[i + 1]);
        if (e > s) {
          Tensor out = outs->template Slice<T>(s, e);
          if (return_index) {
            index->Resize({static_cast<int64_t>(num_kept), 1});
            int* output_idx = index->template mutable_data<int>();
            oindices = output_idx + s;
          }
          MultiClassOutput<T>(scores_slice,
                              boxes_slice,
                              all_indices[i],
                              score_dims.size(),
                              &out,
                              oindices,
                              offset);
        }
      }
    }

    if (return_rois_num) {
      auto* nms_rois_num = param.nms_rois_num;
      nms_rois_num->Resize({n});
      int* num_data = nms_rois_num->template mutable_data<int>();

      for (int i = 1; i <= n; i++) {
        num_data[i - 1] = batch_starts[i] - batch_starts[i - 1];
      }
    }

    LoD lod;
    lod.emplace_back(batch_starts);
    if (return_index) {
      index->set_lod(lod);
    }
    outs->set_lod(lod);
  }

  virtual ~MulticlassNmsCompute() = default;
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
