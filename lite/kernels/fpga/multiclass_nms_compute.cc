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

#include "lite/kernels/fpga/multiclass_nms_compute.h"
#include <map>
#include <utility>
#include <vector>

#include "lite/backends/fpga/KD/debugger.hpp"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

template <class T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

template <class T>
static void GetMaxScoreIndex(const std::vector<T>& scores,
                             const T threshold,
                             int top_k,
                             std::vector<std::pair<T, int>>* sorted_indices) {
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      sorted_indices->push_back(std::make_pair(scores[i], i));
    }
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices->begin(),
                   sorted_indices->end(),
                   SortScorePairDescend<int>);
  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < static_cast<int>(sorted_indices->size())) {
    sorted_indices->resize(top_k);
  }
}

template <class T>
static T BBoxArea(const T* box, const bool normalized) {
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
static T JaccardOverlap(const T* box1, const T* box2, const bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<T>(0.);
  } else {
    const T inter_xmin = std::max(box1[0], box2[0]);
    const T inter_ymin = std::max(box1[1], box2[1]);
    const T inter_xmax = std::min(box1[2], box2[2]);
    const T inter_ymax = std::min(box1[3], box2[3]);
    T norm = normalized ? static_cast<T>(0.) : static_cast<T>(1.);
    T inter_w = inter_xmax - inter_xmin + norm;
    T inter_h = inter_ymax - inter_ymin + norm;
    const T inter_area = inter_w * inter_h;
    const T bbox1_area = BBoxArea<T>(box1, normalized);
    const T bbox2_area = BBoxArea<T>(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <class T>
T PolyIoU(const T* box1,
          const T* box2,
          const size_t box_size,
          const bool normalized) {
  LOG(FATAL) << "PolyIoU not implement.";
  return *box1;
}

template <class T>
void SliceOneClass(const Tensor& items,
                   const int class_id,
                   Tensor* one_class_item) {
  T* item_data = one_class_item->mutable_data<T>();
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
        // 4: [xmin ymin xmax ymax]
        if (box_size == 4) {
          overlap = JaccardOverlap<T>(bbox_data + idx * box_size,
                                      bbox_data + kept_idx * box_size,
                                      normalized);
        }
        // 8: [x1 y1 x2 y2 x3 y3 x4 y4] or 16, 24, 32
        if (box_size == 8 || box_size == 16 || box_size == 24 ||
            box_size == 32) {
          overlap = PolyIoU<T>(bbox_data + idx * box_size,
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
      scores.Slice<T>(score_slice, c, c + 1);
    } else {
      score_slice.Resize({scores.dims()[0], 1});
      bbox_slice.Resize({scores.dims()[0], 4});
      SliceOneClass<T>(scores, c, &score_slice);
      SliceOneClass<T>(bboxes, c, &bbox_slice);
    }
    NMSFast(bboxes,
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
    std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
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
    std::stable_sort(score_index_pairs.begin(),
                     score_index_pairs.end(),
                     SortScorePairDescend<std::pair<int, int>>);
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
  auto* odata = outs->mutable_data<T>();
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

void MulticlassNmsCompute::Run() {
  auto& param = Param<operators::MulticlassNmsParam>();
  auto* boxes = param.bboxes;
  auto* scores = param.scores;
  auto* outs = param.out;
  bool return_index = param.index ? true : false;
  auto* index = param.index;
  auto score_dims = scores->dims();
  auto score_size = score_dims.size();

  std::vector<std::map<int, std::vector<int>>> all_indices;
  std::vector<uint64_t> batch_starts = {0};
  int64_t batch_size = score_dims[0];
  int64_t box_dim = boxes->dims()[2];
  int64_t out_dim = box_dim + 2;
  int num_nmsed_out = 0;
  Tensor boxes_slice, scores_slice;
  int n = score_size == 3 ? batch_size : boxes->lod().back().size() - 1;
  for (int i = 0; i < n; ++i) {
    if (score_size == 3) {
      scores->Slice<float>(scores_slice, i, i + 1);
      scores_slice.Resize({score_dims[1], score_dims[2]});
      boxes->Slice<float>(boxes_slice, i, i + 1);
      boxes_slice.Resize({score_dims[2], box_dim});
    } else {
      auto boxes_lod = boxes->lod().back();
      scores->Slice<float>(scores_slice, boxes_lod[i], boxes_lod[i + 1]);
      boxes->Slice<float>(boxes_slice, boxes_lod[i], boxes_lod[i + 1]);
    }
    std::map<int, std::vector<int>> indices;
    MultiClassNMS<float>(
        param, scores_slice, boxes_slice, score_size, &indices, &num_nmsed_out);
    all_indices.push_back(indices);
    batch_starts.push_back(batch_starts.back() + num_nmsed_out);
  }

  uint64_t num_kept = batch_starts.back();
  if (num_kept == 0) {
    if (return_index) {
      outs->Resize({0, out_dim});
      index->Resize({0, 1});
      outs->mutable_data<float>();
      index->mutable_data<int64_t>();
    } else {
      outs->Resize({1, 1});
      float* od = outs->mutable_data<float>();
      od[0] = -1;
      batch_starts = {0, 1};
    }
  } else {
    outs->Resize({static_cast<int64_t>(num_kept), out_dim});
    outs->mutable_data<float>();
    int offset = 0;
    int* oindices = nullptr;
    for (int i = 0; i < n; ++i) {
      if (score_size == 3) {
        scores->Slice<float>(scores_slice, i, i + 1);
        boxes->Slice<float>(boxes_slice, i, i + 1);
        scores_slice.Resize({score_dims[1], score_dims[2]});
        boxes_slice.Resize({score_dims[2], box_dim});
        if (return_index) {
          offset = i * score_dims[2];
        }
      } else {
        auto boxes_lod = boxes->lod().back();
        scores->Slice<float>(scores_slice, boxes_lod[i], boxes_lod[i + 1]);
        boxes->Slice<float>(boxes_slice, boxes_lod[i], boxes_lod[i + 1]);
        if (return_index) {
          offset = boxes_lod[i] * score_dims[1];
        }
      }
      int64_t s = static_cast<int64_t>(batch_starts[i]);
      int64_t e = static_cast<int64_t>(batch_starts[i + 1]);
      if (e > s) {
        Tensor out;
        outs->Slice<float>(out, s, e);
        Tensor index_int32;
        if (return_index) {
          index->Resize({static_cast<int64_t>(num_kept), 1});

          index_int32.Resize(index->dims());
          int32_t* output_idx = index_int32.mutable_data<int32_t>();

          oindices = output_idx + s;
        }
        MultiClassOutput<float>(scores_slice,
                                boxes_slice,
                                all_indices[i],
                                score_dims.size(),
                                &out,
                                oindices,
                                offset);
        outs->ZynqTensor()->copyFrom(out.ZynqTensor());
        outs->ZynqTensor()->flush();

        if (return_index) {
          int64_t* index_int64_data = index->mutable_data<int64_t>();
          const int32_t* index_int32_data = index_int32.data<int32_t>();
          for (int i = 0; i < index_int32.numel(); ++i) {
            index_int64_data[i] = index_int32_data[i];
          }
        }
      }
    }
  }

  LoD lod;
  lod.emplace_back(batch_starts);
  if (return_index) {
    index->set_lod(lod);
  }
  outs->set_lod(lod);
}
}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(multiclass_nms,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::MulticlassNmsCompute,
                     def)
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(multiclass_nms2,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::MulticlassNmsCompute,
                     def2)
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Index",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
