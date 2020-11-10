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

#include "lite/kernels/arm/generate_proposals_compute.h"
#include <string>
#include <utility>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

static const double kBBoxClipDefault = std::log(1000.0 / 16.0);

static void permute(const Tensor &input,
                    Tensor *output,
                    const std::vector<int> &orders) {
  auto in_dims = input.dims();
  auto out_dims = output->dims();
  int num_axes = in_dims.size();
  int count = in_dims.production();

  const float *din = input.data<float>();
  float *dout = output->mutable_data<float>();
  std::vector<int> old_steps(
      {static_cast<int>(in_dims[1] * in_dims[2] * in_dims[3]),
       static_cast<int>(in_dims[2] * in_dims[3]),
       static_cast<int>(in_dims[3]),
       1});
  std::vector<int> new_steps(
      {static_cast<int>(out_dims[1] * out_dims[2] * out_dims[3]),
       static_cast<int>(out_dims[2] * out_dims[3]),
       static_cast<int>(out_dims[3]),
       1});

  for (int i = 0; i < count; ++i) {
    int old_idx = 0;
    int idx = i;
    for (int j = 0; j < num_axes; ++j) {
      int order = orders[j];
      old_idx += (idx / new_steps[j]) * old_steps[order];
      idx %= new_steps[j];
    }
    dout[i] = din[old_idx];
  }
}

template <typename T, typename IndexT = int>
static void gather(const Tensor &src, const Tensor &index, Tensor *output) {
  auto *p_src = src.data<T>();
  auto *p_index = index.data<IndexT>();
  auto *p_output = output->mutable_data<T>();

  auto src_dims = src.dims();
  int slice_size = 1;
  for (int i = 1; i < src_dims.size(); i++) slice_size *= src_dims[i];
  size_t slice_bytes = slice_size * sizeof(T);

  int64_t index_size = index.numel();
  for (int64_t i = 0; i < index_size; i++) {
    IndexT index_ = p_index[i];
    memcpy(p_output + i * slice_size, p_src + index_ * slice_size, slice_bytes);
  }
}

template <class T>
static void BoxCoder(Tensor *all_anchors,
                     Tensor *bbox_deltas,
                     Tensor *variances,
                     Tensor *proposals) {
  T *proposals_data = proposals->mutable_data<T>();

  int64_t row = all_anchors->dims()[0];
  int64_t len = all_anchors->dims()[1];

  auto *bbox_deltas_data = bbox_deltas->data<T>();
  auto *anchor_data = all_anchors->data<T>();
  const T *variances_data = nullptr;
  if (variances) {
    variances_data = variances->data<T>();
  }

  for (int64_t i = 0; i < row; ++i) {
    T anchor_width = anchor_data[i * len + 2] - anchor_data[i * len] + 1.0;
    T anchor_height = anchor_data[i * len + 3] - anchor_data[i * len + 1] + 1.0;

    T anchor_center_x = anchor_data[i * len] + 0.5 * anchor_width;
    T anchor_center_y = anchor_data[i * len + 1] + 0.5 * anchor_height;

    T bbox_center_x = 0, bbox_center_y = 0;
    T bbox_width = 0, bbox_height = 0;

    if (variances) {
      bbox_center_x =
          variances_data[i * len] * bbox_deltas_data[i * len] * anchor_width +
          anchor_center_x;
      bbox_center_y = variances_data[i * len + 1] *
                          bbox_deltas_data[i * len + 1] * anchor_height +
                      anchor_center_y;
      bbox_width = std::exp(std::min<T>(variances_data[i * len + 2] *
                                            bbox_deltas_data[i * len + 2],
                                        kBBoxClipDefault)) *
                   anchor_width;
      bbox_height = std::exp(std::min<T>(variances_data[i * len + 3] *
                                             bbox_deltas_data[i * len + 3],
                                         kBBoxClipDefault)) *
                    anchor_height;
    } else {
      bbox_center_x =
          bbox_deltas_data[i * len] * anchor_width + anchor_center_x;
      bbox_center_y =
          bbox_deltas_data[i * len + 1] * anchor_height + anchor_center_y;
      bbox_width = std::exp(std::min<T>(bbox_deltas_data[i * len + 2],
                                        kBBoxClipDefault)) *
                   anchor_width;
      bbox_height = std::exp(std::min<T>(bbox_deltas_data[i * len + 3],
                                         kBBoxClipDefault)) *
                    anchor_height;
    }

    proposals_data[i * len] = bbox_center_x - bbox_width / 2;
    proposals_data[i * len + 1] = bbox_center_y - bbox_height / 2;
    proposals_data[i * len + 2] = bbox_center_x + bbox_width / 2 - 1;
    proposals_data[i * len + 3] = bbox_center_y + bbox_height / 2 - 1;
  }
  // return proposals;
}

template <class T>
static void ClipTiledBoxes(const Tensor &im_info, Tensor *boxes) {
  T *boxes_data = boxes->mutable_data<T>();
  const T *im_info_data = im_info.data<T>();
  T zero(0);
  for (int64_t i = 0; i < boxes->numel(); ++i) {
    if (i % 4 == 0) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[1] - 1), zero);
    } else if (i % 4 == 1) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[0] - 1), zero);
    } else if (i % 4 == 2) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[1] - 1), zero);
    } else {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[0] - 1), zero);
    }
  }
}

template <class T>
static void FilterBoxes(Tensor *boxes,
                        float min_size,
                        const Tensor &im_info,
                        Tensor *keep) {
  T *boxes_data = boxes->mutable_data<T>();
  const T *im_info_data = im_info.data<T>();
  T im_scale = im_info_data[2];
  min_size = std::max(min_size, 1.0f);
  keep->Resize(std::vector<int64_t>({boxes->dims()[0]}));
  int *keep_data = keep->mutable_data<int>();

  int keep_len = 0;
  for (int i = 0; i < boxes->dims()[0]; ++i) {
    T ws = boxes_data[4 * i + 2] - boxes_data[4 * i] + 1;
    T hs = boxes_data[4 * i + 3] - boxes_data[4 * i + 1] + 1;
    T ws_origin_scale =
        (boxes_data[4 * i + 2] - boxes_data[4 * i]) / im_scale + 1;
    T hs_origin_scale =
        (boxes_data[4 * i + 3] - boxes_data[4 * i + 1]) / im_scale + 1;
    T x_ctr = boxes_data[4 * i] + ws / 2;
    T y_ctr = boxes_data[4 * i + 1] + hs / 2;
    if (ws_origin_scale >= min_size && hs_origin_scale >= min_size &&
        x_ctr <= im_info_data[1] && y_ctr <= im_info_data[0]) {
      keep_data[keep_len++] = i;
    }
  }
  keep->Resize(std::vector<int64_t>({keep_len}));
}

template <class T>
static std::vector<std::pair<T, int>> GetSortedScoreIndex(
    const std::vector<T> &scores) {
  std::vector<std::pair<T, int>> sorted_indices;
  sorted_indices.reserve(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    sorted_indices.emplace_back(scores[i], i);
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices.begin(),
                   sorted_indices.end(),
                   [](const std::pair<T, int> &a, const std::pair<T, int> &b) {
                     return a.first < b.first;
                   });
  return sorted_indices;
}

template <class T>
static T BBoxArea(const T *box, bool normalized) {
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
static T JaccardOverlap(const T *box1, const T *box2, bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<T>(0.);
  } else {
    const T inter_xmin = std::max(box1[0], box2[0]);
    const T inter_ymin = std::max(box1[1], box2[1]);
    const T inter_xmax = std::min(box1[2], box2[2]);
    const T inter_ymax = std::min(box1[3], box2[3]);
    const T inter_w = std::max(T(0), inter_xmax - inter_xmin + 1);
    const T inter_h = std::max(T(0), inter_ymax - inter_ymin + 1);
    const T inter_area = inter_w * inter_h;
    const T bbox1_area = BBoxArea<T>(box1, normalized);
    const T bbox2_area = BBoxArea<T>(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <class T>
static Tensor VectorToTensor(const std::vector<T> &selected_indices,
                             int selected_num) {
  Tensor keep_nms;
  keep_nms.Resize(std::vector<int64_t>({selected_num}));
  auto *keep_data = keep_nms.mutable_data<T>();
  for (int i = 0; i < selected_num; ++i) {
    keep_data[i] = selected_indices[i];
  }
  return keep_nms;
}

template <class T>
static Tensor NMS(Tensor *bbox, Tensor *scores, T nms_threshold, float eta) {
  int64_t num_boxes = bbox->dims()[0];
  int64_t box_size = bbox->dims()[1];  // 4: [xmin ymin xmax ymax]

  std::vector<T> scores_data(num_boxes);
  std::copy_n(scores->data<T>(), num_boxes, scores_data.begin());
  std::vector<std::pair<T, int>> sorted_indices =
      GetSortedScoreIndex<T>(scores_data);

  std::vector<int> selected_indices;
  int selected_num = 0;
  T adaptive_threshold = nms_threshold;
  const T *bbox_data = bbox->data<T>();
  while (sorted_indices.size() != 0) {
    int idx = sorted_indices.back().second;
    bool flag = true;
    for (int kept_idx : selected_indices) {
      if (flag) {
        T overlap = JaccardOverlap<T>(
            bbox_data + idx * box_size, bbox_data + kept_idx * box_size, false);
        flag = (overlap <= adaptive_threshold);
      } else {
        break;
      }
    }
    if (flag) {
      selected_indices.push_back(idx);
      ++selected_num;
    }
    sorted_indices.erase(sorted_indices.end() - 1);
    if (flag && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
  return VectorToTensor(selected_indices, selected_num);
}

static std::pair<Tensor, Tensor> ProposalForOneImage(
    const Tensor &im_info_slice,
    const Tensor &anchors,
    const Tensor &variances,          // H * W * A * 4
    const Tensor &bbox_deltas_slice,  // [A, 4]
    const Tensor &scores_slice,       // [A, 1]
    int pre_nms_top_n,
    int post_nms_top_n,
    float nms_thresh,
    float min_size,
    float eta) {
  // sort scores_slice
  Tensor index_t;
  index_t.Resize(std::vector<int64_t>({scores_slice.numel()}));
  auto *index = index_t.mutable_data<int>();
  for (int i = 0; i < index_t.numel(); i++) {
    index[i] = i;
  }
  auto *scores_data = scores_slice.data<float>();
  auto compare_func = [scores_data](const int64_t &i, const int64_t &j) {
    return scores_data[i] > scores_data[j];
  };
  if (pre_nms_top_n <= 0 || pre_nms_top_n >= scores_slice.numel()) {
    std::stable_sort(index, index + scores_slice.numel(), compare_func);
  } else {
    std::nth_element(index,
                     index + pre_nms_top_n,
                     index + scores_slice.numel(),
                     compare_func);
    index_t.Resize({pre_nms_top_n});
  }

  Tensor scores_sel, bbox_sel, anchor_sel, var_sel;
  scores_sel.Resize(std::vector<int64_t>({index_t.numel(), 1}));
  bbox_sel.Resize(std::vector<int64_t>({index_t.numel(), 4}));
  anchor_sel.Resize(std::vector<int64_t>({index_t.numel(), 4}));
  var_sel.Resize(std::vector<int64_t>({index_t.numel(), 4}));
  gather<float>(scores_slice, index_t, &scores_sel);
  gather<float>(bbox_deltas_slice, index_t, &bbox_sel);
  gather<float>(anchors, index_t, &anchor_sel);
  gather<float>(variances, index_t, &var_sel);

  Tensor proposals;
  proposals.Resize(std::vector<int64_t>({index_t.numel(), 4}));
  BoxCoder<float>(&anchor_sel, &bbox_sel, &var_sel, &proposals);

  ClipTiledBoxes<float>(im_info_slice, &proposals);

  Tensor keep;
  FilterBoxes<float>(&proposals, min_size, im_info_slice, &keep);
  Tensor scores_filter;
  scores_filter.Resize(std::vector<int64_t>({keep.numel(), 1}));
  bbox_sel.Resize(std::vector<int64_t>({keep.numel(), 4}));
  gather<float>(scores_sel, keep, &scores_filter);
  gather<float>(proposals, keep, &bbox_sel);
  if (nms_thresh <= 0) {
    return std::make_pair(bbox_sel, scores_filter);
  }

  Tensor keep_nms = NMS<float>(&bbox_sel, &scores_filter, nms_thresh, eta);
  if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
    keep_nms.Resize(std::vector<int64_t>({post_nms_top_n}));
  }
  proposals.Resize(std::vector<int64_t>({keep_nms.numel(), 4}));
  scores_sel.Resize(std::vector<int64_t>({keep_nms.numel(), 1}));
  gather<float>(bbox_sel, keep_nms, &proposals);
  gather<float>(scores_filter, keep_nms, &scores_sel);
  return std::make_pair(proposals, scores_sel);
}

void AppendTensor(Tensor *dst, int64_t offset, const Tensor &src) {
  auto *out_data = static_cast<void *>(dst->mutable_data<float>());
  auto *to_add_data = static_cast<const void *>(src.data<float>());
  size_t size_of_t = sizeof(float);
  offset *= size_of_t;
  std::memcpy(
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(out_data) + offset),
      to_add_data,
      src.numel() * size_of_t);
}

void GenerateProposalsCompute::Run() {
  auto &ctx = this->ctx_->template As<ARMContext>();
  auto &param = Param<operators::GenerateProposalsParam>();
  auto *scores = param.Scores;              // N * A * H * W
  auto *bbox_deltas = param.BboxDeltas;     // N * 4A * H * W
  auto *im_info = param.ImInfo;             // N * 3
  auto *anchors = param.Anchors;            // H * W * A * 4
  auto *variances = param.Variances;        // H * W * A * 4
  auto *rpn_rois = param.RpnRois;           // A * 4
  auto *rpn_roi_probs = param.RpnRoiProbs;  // A * 1
  int pre_nms_top_n = param.pre_nms_topN;
  int post_nms_top_n = param.post_nms_topN;
  float nms_thresh = param.nms_thresh;
  float min_size = param.min_size;
  float eta = param.eta;

  auto &scores_dim = scores->dims();
  int64_t num = scores_dim[0];
  int64_t c_score = scores_dim[1];
  int64_t h_score = scores_dim[2];
  int64_t w_score = scores_dim[3];
  auto &bbox_dim = bbox_deltas->dims();
  int64_t c_bbox = bbox_dim[1];
  int64_t h_bbox = bbox_dim[2];
  int64_t w_bbox = bbox_dim[3];

  rpn_rois->Resize({scores->numel(), 4});
  rpn_roi_probs->Resize(std::vector<int64_t>({scores->numel(), 1}));

  Tensor bbox_deltas_swap, scores_swap;
  scores_swap.Resize(std::vector<int64_t>({num, h_score, w_score, c_score}));
  bbox_deltas_swap.Resize(std::vector<int64_t>({num, h_bbox, w_bbox, c_bbox}));
  std::vector<int> orders({0, 2, 3, 1});
  permute(*scores, &scores_swap, orders);
  permute(*bbox_deltas, &bbox_deltas_swap, orders);

  LoD lod;
  lod.resize(1);
  auto &lod0 = lod[0];
  lod0.push_back(0);
  anchors->Resize(std::vector<int64_t>({anchors->numel() / 4, 4}));
  variances->Resize(std::vector<int64_t>({variances->numel() / 4, 4}));
  std::vector<int64_t> tmp_lod;

  int64_t num_proposals = 0;
  for (int64_t i = 0; i < num; ++i) {
    Tensor im_info_slice = im_info->Slice<float>(i, i + 1);
    Tensor bbox_deltas_slice = bbox_deltas_swap.Slice<float>(i, i + 1);
    Tensor scores_slice = scores_swap.Slice<float>(i, i + 1);

    bbox_deltas_slice.Resize(
        std::vector<int64_t>({c_bbox * h_bbox * w_bbox / 4, 4}));
    scores_slice.Resize(std::vector<int64_t>({c_score * h_score * w_score, 1}));

    std::pair<Tensor, Tensor> tensor_pair =
        ProposalForOneImage(im_info_slice,
                            *anchors,
                            *variances,
                            bbox_deltas_slice,
                            scores_slice,
                            pre_nms_top_n,
                            post_nms_top_n,
                            nms_thresh,
                            min_size,
                            eta);
    Tensor &proposals = tensor_pair.first;
    Tensor &scores = tensor_pair.second;

    AppendTensor(rpn_rois, 4 * num_proposals, proposals);
    AppendTensor(rpn_roi_probs, num_proposals, scores);

    num_proposals += proposals.dims()[0];
    lod0.push_back(num_proposals);
    tmp_lod.push_back(num_proposals);
  }

  if (param.RpnRoisLod != nullptr) {
    param.RpnRoisLod->Resize(DDim(std::vector<DDim::value_type>({num})));
    int64_t *lod_data = param.RpnRoisLod->mutable_data<int64_t>();
    for (int i = 0; i < num; i++) {
      lod_data[i] = tmp_lod[i];
    }
  }

  rpn_rois->set_lod(lod);
  rpn_roi_probs->set_lod(lod);
  rpn_rois->Resize({num_proposals, 4});
  rpn_roi_probs->Resize({num_proposals, 1});

  /*
  auto* rpn_roi_probs_data = rpn_roi_probs->data<float>();
  LOG(INFO) << "rpn_roi_probs:" << rpn_roi_probs->dims();
  for (int i = 0; i < rpn_roi_probs->numel() - 4; i = i + 4) {
    LOG(INFO) << rpn_roi_probs_data[i] << " " << rpn_roi_probs_data[i+1]
      << " " << rpn_roi_probs_data[i+2] << " " << rpn_roi_probs_data[i+3];
  }
  auto* rpn_roi_data = rpn_rois->data<float>();
  LOG(INFO) << "rpn_roi:" << rpn_rois->dims();
  for (int i = 0; i < rpn_rois->numel() - 4; i = i + 4) {
    LOG(INFO) << rpn_roi_data[i] << " " << rpn_roi_data[i+1]
      << " " << rpn_roi_data[i+2] << " " << rpn_roi_data[i+3];
  }
  */
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(generate_proposals,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::GenerateProposalsCompute,
                     def)
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("BboxDeltas", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("ImInfo", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Anchors", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Variances", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("RpnRois", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("RpnRoiProbs", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("RpnRoisLod",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
