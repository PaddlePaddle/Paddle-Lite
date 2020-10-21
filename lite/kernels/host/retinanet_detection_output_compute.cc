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

#include "lite/kernels/host/retinanet_detection_output_compute.h"
#include <cmath>
#include <map>
#include <utility>
#include <vector>
#include "lite/operators/retinanet_detection_output_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

template <class T>
bool SortScoreTwoPairDescend(const std::pair<float, std::pair<T, T>>& pair1,
                             const std::pair<float, std::pair<T, T>>& pair2) {
  return pair1.first > pair2.first;
}

template <class T>
static inline void GetMaxScoreIndex(
    const std::vector<T>& scores,
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
static inline T BBoxArea(const std::vector<T>& box, const bool normalized) {
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
static inline T JaccardOverlap(const std::vector<T>& box1,
                               const std::vector<T>& box2,
                               const bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<T>(0.);
  } else {
    const T inter_xmin = (std::max)(box1[0], box2[0]);
    const T inter_ymin = (std::max)(box1[1], box2[1]);
    const T inter_xmax = (std::min)(box1[2], box2[2]);
    const T inter_ymax = (std::min)(box1[3], box2[3]);
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
void NMSFast(const std::vector<std::vector<T>>& cls_dets,
             const T nms_threshold,
             const T eta,
             std::vector<int>* selected_indices) {
  int64_t num_boxes = cls_dets.size();
  std::vector<std::pair<T, int>> sorted_indices;
  for (int64_t i = 0; i < num_boxes; ++i) {
    sorted_indices.push_back(std::make_pair(cls_dets[i][4], i));
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(
      sorted_indices.begin(), sorted_indices.end(), SortScorePairDescend<int>);
  selected_indices->clear();
  T adaptive_threshold = nms_threshold;

  while (sorted_indices.size() != 0) {
    const int idx = sorted_indices.front().second;
    bool keep = true;
    for (size_t k = 0; k < selected_indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*selected_indices)[k];
        T overlap = T(0.);

        overlap = JaccardOverlap<T>(cls_dets[idx], cls_dets[kept_idx], false);
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

template <class T>
void DeltaScoreToPrediction(
    const std::vector<T>& bboxes_data,
    const std::vector<T>& anchors_data,
    T im_height,
    T im_width,
    T im_scale,
    int class_num,
    const std::vector<std::pair<T, int>>& sorted_indices,
    std::map<int, std::vector<std::vector<T>>>* preds) {
  im_height = static_cast<T>(std::round(im_height / im_scale));
  im_width = static_cast<T>(std::round(im_width / im_scale));
  T zero(0);
  int i = 0;
  for (const auto& it : sorted_indices) {
    T score = it.first;
    int idx = it.second;
    int a = idx / class_num;
    int c = idx % class_num;

    int box_offset = a * 4;
    T anchor_box_width =
        anchors_data[box_offset + 2] - anchors_data[box_offset] + 1;
    T anchor_box_height =
        anchors_data[box_offset + 3] - anchors_data[box_offset + 1] + 1;
    T anchor_box_center_x = anchors_data[box_offset] + anchor_box_width / 2;
    T anchor_box_center_y =
        anchors_data[box_offset + 1] + anchor_box_height / 2;
    T target_box_center_x = 0, target_box_center_y = 0;
    T target_box_width = 0, target_box_height = 0;
    target_box_center_x =
        bboxes_data[box_offset] * anchor_box_width + anchor_box_center_x;
    target_box_center_y =
        bboxes_data[box_offset + 1] * anchor_box_height + anchor_box_center_y;
    target_box_width = std::exp(bboxes_data[box_offset + 2]) * anchor_box_width;
    target_box_height =
        std::exp(bboxes_data[box_offset + 3]) * anchor_box_height;
    T pred_box_xmin = target_box_center_x - target_box_width / 2;
    T pred_box_ymin = target_box_center_y - target_box_height / 2;
    T pred_box_xmax = target_box_center_x + target_box_width / 2 - 1;
    T pred_box_ymax = target_box_center_y + target_box_height / 2 - 1;
    pred_box_xmin = pred_box_xmin / im_scale;
    pred_box_ymin = pred_box_ymin / im_scale;
    pred_box_xmax = pred_box_xmax / im_scale;
    pred_box_ymax = pred_box_ymax / im_scale;

    pred_box_xmin = (std::max)((std::min)(pred_box_xmin, im_width - 1), zero);
    pred_box_ymin = (std::max)((std::min)(pred_box_ymin, im_height - 1), zero);
    pred_box_xmax = (std::max)((std::min)(pred_box_xmax, im_width - 1), zero);
    pred_box_ymax = (std::max)((std::min)(pred_box_ymax, im_height - 1), zero);

    std::vector<T> one_pred;
    one_pred.push_back(pred_box_xmin);
    one_pred.push_back(pred_box_ymin);
    one_pred.push_back(pred_box_xmax);
    one_pred.push_back(pred_box_ymax);
    one_pred.push_back(score);
    (*preds)[c].push_back(one_pred);
    i++;
  }
}

template <class T>
void MultiClassNMS(const std::map<int, std::vector<std::vector<T>>>& preds,
                   int class_num,
                   const int keep_top_k,
                   const T nms_threshold,
                   const T nms_eta,
                   std::vector<std::vector<T>>* nmsed_out,
                   int* num_nmsed_out) {
  std::map<int, std::vector<int>> indices;
  int num_det = 0;
  for (int c = 0; c < class_num; ++c) {
    if (static_cast<bool>(preds.count(c))) {
      const std::vector<std::vector<T>> cls_dets = preds.at(c);
      NMSFast(cls_dets, nms_threshold, nms_eta, &(indices[c]));
      num_det += indices[c].size();
    }
  }

  std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
  for (const auto& it : indices) {
    int label = it.first;
    const std::vector<int>& label_indices = it.second;
    for (size_t j = 0; j < label_indices.size(); ++j) {
      int idx = label_indices[j];
      score_index_pairs.push_back(
          std::make_pair(preds.at(label)[idx][4], std::make_pair(label, idx)));
    }
  }
  // Keep top k results per image.
  std::stable_sort(score_index_pairs.begin(),
                   score_index_pairs.end(),
                   SortScoreTwoPairDescend<int>);
  if (num_det > keep_top_k) {
    score_index_pairs.resize(keep_top_k);
  }

  // Store the new indices.
  std::map<int, std::vector<int>> new_indices;
  for (const auto& it : score_index_pairs) {
    int label = it.second.first;
    int idx = it.second.second;
    std::vector<T> one_pred;
    one_pred.push_back(label);
    one_pred.push_back(preds.at(label)[idx][4]);
    one_pred.push_back(preds.at(label)[idx][0]);
    one_pred.push_back(preds.at(label)[idx][1]);
    one_pred.push_back(preds.at(label)[idx][2]);
    one_pred.push_back(preds.at(label)[idx][3]);
    nmsed_out->push_back(one_pred);
  }

  *num_nmsed_out = (num_det > keep_top_k ? keep_top_k : num_det);
}

template <class T>
void RetinanetDetectionOutput(
    const operators::RetinanetDetectionOutputParam& param,
    const std::vector<Tensor>& scores,
    const std::vector<Tensor>& bboxes,
    const std::vector<Tensor>& anchors,
    const Tensor& im_info,
    std::vector<std::vector<T>>* nmsed_out,
    int* num_nmsed_out) {
  int64_t nms_top_k = param.nms_top_k;
  int64_t keep_top_k = param.keep_top_k;
  T nms_threshold = static_cast<T>(param.nms_threshold);
  T nms_eta = static_cast<T>(param.nms_eta);
  T score_threshold = static_cast<T>(param.score_threshold);

  int64_t class_num = scores[0].dims()[1];
  std::map<int, std::vector<std::vector<T>>> preds;
  for (size_t l = 0; l < scores.size(); ++l) {
    // Fetch per level score
    Tensor scores_per_level = scores[l];
    // Fetch per level bbox
    Tensor bboxes_per_level = bboxes[l];
    // Fetch per level anchor
    Tensor anchors_per_level = anchors[l];

    int64_t scores_num = scores_per_level.numel();
    int64_t bboxes_num = bboxes_per_level.numel();
    std::vector<T> scores_data(scores_num);
    std::vector<T> bboxes_data(bboxes_num);
    std::vector<T> anchors_data(bboxes_num);
    std::copy_n(scores_per_level.data<T>(), scores_num, scores_data.begin());
    std::copy_n(bboxes_per_level.data<T>(), bboxes_num, bboxes_data.begin());
    std::copy_n(anchors_per_level.data<T>(), bboxes_num, anchors_data.begin());
    std::vector<std::pair<T, int>> sorted_indices;

    // For the highest level, we take the threshold 0.0
    T threshold = (l < (scores.size() - 1) ? score_threshold : 0.0);
    GetMaxScoreIndex(scores_data, threshold, nms_top_k, &sorted_indices);
    auto* im_info_data = im_info.data<T>();
    auto im_height = im_info_data[0];
    auto im_width = im_info_data[1];
    auto im_scale = im_info_data[2];
    DeltaScoreToPrediction(bboxes_data,
                           anchors_data,
                           im_height,
                           im_width,
                           im_scale,
                           class_num,
                           sorted_indices,
                           &preds);
  }

  MultiClassNMS(preds,
                class_num,
                keep_top_k,
                nms_threshold,
                nms_eta,
                nmsed_out,
                num_nmsed_out);
}

template <class T>
void MultiClassOutput(const std::vector<std::vector<T>>& nmsed_out,
                      Tensor* outs) {
  auto* odata = outs->mutable_data<T>();
  int count = 0;
  int64_t out_dim = 6;
  for (size_t i = 0; i < nmsed_out.size(); ++i) {
    odata[count * out_dim] = nmsed_out[i][0] + 1;  // label
    odata[count * out_dim + 1] = nmsed_out[i][1];  // score
    odata[count * out_dim + 2] = nmsed_out[i][2];  // xmin
    odata[count * out_dim + 3] = nmsed_out[i][3];  // xmin
    odata[count * out_dim + 4] = nmsed_out[i][4];  // xmin
    odata[count * out_dim + 5] = nmsed_out[i][5];  // xmin
    count++;
  }
}

void RetinanetDetectionOutputCompute::Run() {
  auto& param = Param<operators::RetinanetDetectionOutputParam>();
  auto& boxes = param.bboxes;
  auto& scores = param.scores;
  auto& anchors = param.anchors;
  auto* im_info = param.im_info;
  auto* outs = param.out;

  std::vector<Tensor> boxes_list(boxes.size());
  std::vector<Tensor> scores_list(scores.size());
  std::vector<Tensor> anchors_list(anchors.size());
  for (size_t j = 0; j < boxes_list.size(); ++j) {
    boxes_list[j] = *boxes[j];
    scores_list[j] = *scores[j];
    anchors_list[j] = *anchors[j];
  }
  auto score_dims = scores_list[0].dims();
  int64_t batch_size = score_dims[0];
  auto box_dims = boxes_list[0].dims();
  int64_t box_dim = box_dims[2];
  int64_t out_dim = box_dim + 2;

  std::vector<std::vector<std::vector<float>>> all_nmsed_out;
  std::vector<uint64_t> batch_starts = {0};
  for (int i = 0; i < batch_size; ++i) {
    int num_nmsed_out = 0;
    std::vector<Tensor> box_per_batch_list(boxes_list.size());
    std::vector<Tensor> score_per_batch_list(scores_list.size());
    for (size_t j = 0; j < boxes_list.size(); ++j) {
      auto score_dims = scores_list[j].dims();
      score_per_batch_list[j] = scores_list[j].Slice<float>(i, i + 1);
      score_per_batch_list[j].Resize({score_dims[1], score_dims[2]});
      box_per_batch_list[j] = boxes_list[j].Slice<float>(i, i + 1);
      box_per_batch_list[j].Resize({score_dims[1], box_dim});
    }
    Tensor im_info_slice = im_info->Slice<float>(i, i + 1);

    std::vector<std::vector<float>> nmsed_out;
    RetinanetDetectionOutput(param,
                             score_per_batch_list,
                             box_per_batch_list,
                             anchors_list,
                             im_info_slice,
                             &nmsed_out,
                             &num_nmsed_out);
    all_nmsed_out.push_back(nmsed_out);
    batch_starts.push_back(batch_starts.back() + num_nmsed_out);
  }

  uint64_t num_kept = batch_starts.back();
  if (num_kept == 0) {
    outs->Resize({0, out_dim});
  } else {
    outs->Resize({static_cast<int64_t>(num_kept), out_dim});
    for (int i = 0; i < batch_size; ++i) {
      int64_t s = static_cast<int64_t>(batch_starts[i]);
      int64_t e = static_cast<int64_t>(batch_starts[i + 1]);
      if (e > s) {
        Tensor out = outs->Slice<float>(s, e);
        MultiClassOutput(all_nmsed_out[i], &out);
      }
    }
  }

  LoD lod;
  lod.emplace_back(batch_starts);
  outs->set_lod(lod);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    retinanet_detection_output,
    kHost,
    kFloat,
    kNCHW,
    paddle::lite::kernels::host::RetinanetDetectionOutputCompute,
    def)
    .BindInput("BBoxes",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Scores",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Anchors",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("ImInfo",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
