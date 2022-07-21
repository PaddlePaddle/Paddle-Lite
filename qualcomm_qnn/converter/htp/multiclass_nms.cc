// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/qualcomm_qnn/converter/htp/multiclass_nms.h"
#include "driver/qualcomm_qnn/converter/htp/poly_util.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace htp {

template <typename T>
inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                                 const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

template <typename T>
inline void GetMaxScoreIndex(const std::vector<T>& scores,
                             const float threshold,
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

template <typename T>
inline T BBoxArea(const T* box, const bool normalized) {
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

template <typename T>
inline T JaccardOverlap(const T* box1, const T* box2, const bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<T>(0.);
  } else {
    const T inter_xmin = (std::max)(box1[0], box2[0]);
    const T inter_ymin = (std::max)(box1[1], box2[1]);
    const T inter_xmax = (std::min)(box1[2], box2[2]);
    const T inter_ymax = (std::min)(box1[3], box2[3]);
    const T norm = normalized ? static_cast<T>(0.) : static_cast<T>(1.);
    const T inter_w = inter_xmax - inter_xmin + norm;
    const T inter_h = inter_ymax - inter_ymin + norm;
    const T inter_area = inter_w * inter_h;
    const T bbox1_area = BBoxArea<T>(box1, normalized);
    const T bbox2_area = BBoxArea<T>(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <typename T>
inline T PolyIoU(const T* box1,
                 const T* box2,
                 const size_t box_size,
                 const bool normalized) {
  T bbox1_area = PolyArea<T>(box1, box_size, normalized);
  T bbox2_area = PolyArea<T>(box2, box_size, normalized);
  T inter_area = PolyOverlapArea<T>(box1, box2, box_size, normalized);
  if (bbox1_area == 0 || bbox2_area == 0 || inter_area == 0) {
    // If coordinate values are invalid
    // if area size <= 0,  return 0.
    return T(0.);
  } else {
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <typename T>
inline void NMSFast(const std::vector<T>& bbox,
                    const std::vector<T>& scores,
                    const float score_threshold,
                    const float nms_threshold,
                    const float eta,
                    const int32_t top_k,
                    std::vector<int>* selected_indices,
                    const bool normalized,
                    const int32_t box_num,
                    const int32_t box_size) {
  std::vector<std::pair<T, int>> sorted_indices;
  GetMaxScoreIndex(scores, score_threshold, top_k, &sorted_indices);
  selected_indices->clear();
  T adaptive_threshold = nms_threshold;
  const T* bbox_data = bbox.data();
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
void MultiClassOutput(const std::vector<T>& scores,
                      const std::vector<T>& bboxes,
                      const std::map<int, std::vector<int>>& selected_indices,
                      std::vector<T>* outs,
                      int32_t class_num,
                      int32_t box_size,
                      int* oindices = nullptr,
                      const int offset = 0) {
  int32_t out_dim = box_size + 2;
  auto* scores_data = scores.data();
  auto* bboxes_data = bboxes.data();
  auto* odata = outs->data();
  const T* sdata;
  int count = 0;
  for (const auto& it : selected_indices) {
    int label = it.first;
    const std::vector<int>& indices = it.second;
    sdata = scores_data + label * class_num;
    for (size_t j = 0; j < indices.size(); ++j) {
      int idx = indices[j];
      odata[count * out_dim] = label;  // label
      const T* bdata;
      bdata = bboxes_data + idx * box_size;
      odata[count * out_dim + 1] = sdata[idx];  // score
      if (oindices != nullptr) {
        oindices[count] = offset + idx;
      }
      // xmin, ymin, xmax, ymax or multi-points coordinates
      std::memcpy(odata + count * out_dim + 2, bdata, box_size * sizeof(T));
      count++;
    }
  }
}

template <typename T>
void MultiClassNMS(const std::vector<T>& bboxes,
                   const std::vector<T>& scores,
                   const int32_t background_label,
                   const int32_t nms_top_k,
                   const int32_t keep_top_k,
                   const bool normalized,
                   const float nms_threshold,
                   const float nms_eta,
                   const float score_threshold,
                   const int32_t class_num,
                   const int32_t box_num,
                   const int32_t box_size,
                   std::map<int, std::vector<int>>* indices,
                   int* num_nmsed_out) {
  int num_det = 0;
  for (int32_t c = 0; c < class_num; ++c) {
    if (c == background_label) continue;
    std::vector<T> score_slice = std::vector<T>(
        scores.data() + c * box_num, scores.data() + (c + 1) * box_num);
    NMSFast(bboxes,
            score_slice,
            score_threshold,
            nms_threshold,
            nms_eta,
            nms_top_k,
            &((*indices)[c]),
            normalized,
            box_num,
            box_size);
    num_det += (*indices)[c].size();
  }

  *num_nmsed_out = num_det;
  const T* scores_data = scores.data();
  if (keep_top_k > -1 && num_det > keep_top_k) {
    const T* sdata;
    std::vector<std::pair<T, std::pair<int, int>>> score_index_pairs;
    for (const auto& it : *indices) {
      int label = it.first;
      sdata = scores_data + label * box_num;
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
    new_indices.swap(*indices);
    *num_nmsed_out = keep_top_k;
  }
}

template <typename T>
inline void MulticlassNmsKernel(T* output_box_data,
                                int32_t* output_nms_rois_num_data,
                                int32_t* output_index_data,
                                const T* bboxes_data,
                                const T* scores_data,
                                const int32_t background_label_data,
                                const float score_threshold_data,
                                const int32_t nms_top_k_data,
                                const float nms_threshold_data,
                                const float nms_eta_data,
                                const int32_t keep_top_k_data,
                                const bool normalized_data,
                                const int32_t batch_size,
                                const int32_t class_num,
                                const int32_t box_num,
                                const int32_t box_size,
                                const std::vector<uint32_t>& batch_starts) {
  std::vector<std::map<int, std::vector<int>>> all_indices;
  batch_starts = {0};
  int num_nmsed_out = 0;
  int32_t box_out_dim = box_size + 2;
  int32_t bboxes_batch_stride = box_num * box_size;
  int32_t scores_batch_stride = class_num * box_num;
  std::vector<float> boxes_slice, scores_slice;
  for (int i = 0; i < batch_size; ++i) {
    boxes_slice =
        std::vector<float>(bboxes_data + i * bboxes_batch_stride,
                           bboxes_data + (i + 1) * bboxes_batch_stride);
    scores_slice =
        std::vector<float>(scores_data + i * scores_batch_stride,
                           scores_data + (i + 1) * scores_batch_stride);
    std::map<int, std::vector<int>> indices;
    MultiClassNMS<float>(boxes_slice,
                         scores_slice,
                         background_label_data,
                         nms_top_k_data,
                         keep_top_k_data,
                         normalized_data,
                         nms_threshold_data,
                         nms_eta_data,
                         score_threshold_data,
                         class_num,
                         box_num,
                         box_size,
                         &indices,
                         &num_nmsed_out);
    all_indices.push_back(indices);
    batch_starts.push_back(batch_starts.back() + num_nmsed_out);
  }

  uint32_t num_kept = batch_starts.back();
  if (num_kept != 0) {
    int offset = 0;
    int* oindices = nullptr;
    for (int i = 0; i < batch_size; ++i) {
      boxes_slice =
          std::vector<float>(bboxes_data + i * bboxes_batch_stride,
                             bboxes_data + (i + 1) * bboxes_batch_stride);
      scores_slice =
          std::vector<float>(scores_data + i * scores_batch_stride,
                             scores_data + (i + 1) * scores_batch_stride);
      offset = i * box_num;
      int32_t s = static_cast<int32_t>(batch_starts[i]);
      int32_t e = static_cast<int32_t>(batch_starts[i + 1]);
      if (e > s) {
        std::vector<float> out = std::vector<float>(output_box_data + s * 6,
                                                    output_box_data + e * 6);
        oindices = output_index_data + s;
        MultiClassOutput<float>(scores_slice,
                                boxes_slice,
                                all_indices[i],
                                &out,
                                class_num,
                                box_size,
                                oindices,
                                offset);
        std::memcpy(
            output_box_data + s * 6, out.data(), (e - s) * 6 * sizeof(float));
      }
    }
  }
  for (int i = 1; i <= batch_size; i++) {
    output_nms_rois_num_data[i - 1] = batch_starts[i] - batch_starts[i - 1];
  }
}

template <typename TensorType>
int MulticlassNmsImpl(TensorType& output_box,           // NOLINT
                      TensorType& output_nms_rois_num,  // NOLINT
                      TensorType& output_index,         // NOLINT
                      const TensorType& bboxes,
                      const TensorType& scores,
                      const Tensor& background_label,
                      const Tensor& score_threshold,
                      const Tensor& nms_top_k,
                      const Tensor& nms_threshold,
                      const Tensor& nms_eta,
                      const Tensor& keep_top_k,
                      const Tensor& normalized) {
  infolog("MulticlassNmsImpl execute... ");
  // Input
  auto* bboxes_data = reinterpret_cast<const float*>(bboxes.raw_data_const());
  auto* scores_data = reinterpret_cast<const float*>(scores.raw_data_const());
  // Output
  auto* output_box_data = reinterpret_cast<float*>(output_box.raw_data());
  auto* output_nms_rois_num_data =
      reinterpret_cast<int*>(output_nms_rois_num.raw_data());
  auto* output_index_data = reinterpret_cast<int*>(output_index.raw_data());
  // Attr
  int32_t background_label_data = background_label(0, 0, 0, 0);
  float score_threshold_data = score_threshold(0, 0, 0, 0);
  int32_t nms_top_k_data = nms_top_k(0, 0, 0, 0);
  float nms_threshold_data = nms_threshold(0, 0, 0, 0);
  float nms_eta_data = nms_eta(0, 0, 0, 0);
  int32_t keep_top_k_data = keep_top_k(0, 0, 0, 0);
  bool normalized_data = normalized(0, 0, 0, 0);
  int32_t batch_size = scores.dim(1);
  int32_t class_num = scores.dim(2);
  // The total boxes for each instance.
  int32_t box_num = bboxes.dim(2);
  // 4: [xmin ymin xmax ymax]
  // 8: [x1 y1 x2 y2 x3 y3 x4 y4]
  // 16, 24, or 32: [x1 y1 x2 y2 ...  xn yn], n = 8, 12 or 16
  int32_t box_size = bboxes.dim(3);
  int32_t box_out_dim = box_size + 2;
  std::vector<uint32_t> batch_starts;

  MulticlassNmsKernel(output_box_data,
                      output_nms_rois_num_data,
                      output_index_data,
                      bboxes_data,
                      scores_data,
                      background_label_data,
                      score_threshold_data,
                      nms_top_k_data,
                      nms_threshold_data,
                      nms_eta_data,
                      keep_top_k_data,
                      normalized_data,
                      batch_size,
                      class_num,
                      box_num,
                      box_size,
                      batch_starts);

  size_t output_nms_rois_num_dims[] = {static_cast<size_t>(batch_size)};
  output_nms_rois_num.set_dims(output_nms_rois_num_dims);

  uint32_t num_kept = batch_starts.back();
  if (num_kept == 0) {
    size_t output_box_dims[] = {0, static_cast<size_t>(box_out_dim)};
    output_box.set_dims(output_box_dims);
    size_t output_index_dims[] = {0, 1};
    output_index.set_dims(output_index_dims);
  } else {
    size_t output_box_dims[] = {
        1, 1, static_cast<size_t>(num_kept), static_cast<size_t>(box_out_dim)};
    output_box.set_dims(output_box_dims);
    size_t output_index_dims[] = {static_cast<size_t>(num_kept), 1};
    output_index.set_dims(output_index_dims);
  }

  return GraphStatus::Success;
}

}  // namespace htp
}  // namespace qualcomm_qnn
}  // namespace nnadapter

using namespace nnadapter::qualcomm_qnn::htp;  // NOLINT
using namespace hnnx;                          // NOLINT

const char* op_name_multiclass_nms = "MulticlassNms";

DEF_PACKAGE_OP(MulticlassNmsImpl<Tensor>, op_name_multiclass_nms)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((MulticlassNmsImpl<PlainFloatTensor>),
                                  op_name_multiclass_nms,
                                  "SNAIL",
                                  Flags::RESOURCE_HVX)
