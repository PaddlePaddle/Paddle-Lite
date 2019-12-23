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

#ifdef PROPOSAL_OP

#include <algorithm>
#include <cmath>
#include <vector>
#include "operators/kernel/detection_kernel.h"

namespace paddle_mobile {
namespace operators {

static const double kBBoxClipDefault = std::log(1000.0 / 16.0);

template <>
bool ProposalKernel<FPGA, float>::Init(ProposalParam<FPGA> *param) {
  int post_nms_top_n = param->post_nms_topn_;
  int64_t batch = param->scores_->dims()[0];
  auto total = post_nms_top_n * batch;
  param->rpn_rois_->mutable_data<float>({total, 4});
  param->rpn_probs_->mutable_data<int8_t>({total, 1});

  param->float_bbox = std::make_shared<Tensor>();
  param->float_bbox->Resize(param->bbox_deltas_->dims());
  param->float_bbox->init(type_id<float>().hash_code());
  fpga::format_fp32_ofm(param->float_bbox.get());

  auto input = param->scores_;
  param->score_index_ = std::make_shared<Tensor>();
  param->score_index_->mutable_data<int32_t>({input->numel()});
  auto score_index = param->score_index_->data<int32_t>();
  for (int i = 0; i < input->numel(); ++i) {
    score_index[i] = i;
  }

  return true;
}
template <typename T>
void CPUGather(const Tensor &src, const Tensor &index, Tensor *output) {
  PADDLE_MOBILE_ENFORCE(index.dims().size() == 1 ||
                            (index.dims().size() == 2 && index.dims()[1] == 1),
                        "Dim not correct");
  int64_t index_size = index.dims()[0];

  auto src_dims = src.dims();

  const T *p_src = src.data<T>();
  const int *p_index = index.data<int>();
  T *p_output = output->data<T>();

  // slice size
  int slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const size_t slice_bytes = slice_size * sizeof(T);

  for (int64_t i = 0; i < index_size; ++i) {
    int index_ = p_index[i];
    memcpy(p_output + i * slice_size, p_src + index_ * slice_size, slice_bytes);
  }
}

void AppendProposals(Tensor *dst, int64_t offset, const Tensor &src) {
  auto *out_data = dst->data<void>();
  auto *to_add_data = src.data<void>();
  size_t size_of_t = framework::SizeOfType(src.type());
  offset *= size_of_t;
  std::memcpy(
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(out_data) + offset),
      to_add_data, src.numel() * size_of_t);
}

template <class T>
static inline void BoxCoder(Tensor *all_anchors, Tensor *bbox_deltas,
                            Tensor *proposals) {
  T *proposals_data = proposals->mutable_data<T>();

  int64_t row = all_anchors->dims()[0];
  int64_t len = all_anchors->dims()[1];

  auto *bbox_deltas_data = bbox_deltas->data<T>();
  auto *anchor_data = all_anchors->data<T>();

  for (int64_t i = 0; i < row; ++i) {
    T anchor_width = anchor_data[i * len + 2] - anchor_data[i * len] + 1.0;
    T anchor_height = anchor_data[i * len + 3] - anchor_data[i * len + 1] + 1.0;

    T anchor_center_x = anchor_data[i * len] + 0.5 * anchor_width;
    T anchor_center_y = anchor_data[i * len + 1] + 0.5 * anchor_height;

    T bbox_center_x = 0, bbox_center_y = 0;
    T bbox_width = 0, bbox_height = 0;

    bbox_center_x = bbox_deltas_data[i * len] * anchor_width + anchor_center_x;
    bbox_center_y =
        bbox_deltas_data[i * len + 1] * anchor_height + anchor_center_y;
    bbox_width = std::exp(bbox_deltas_data[i * len + 2]) * anchor_width;
    bbox_height = std::exp(bbox_deltas_data[i * len + 3]) * anchor_height;

    proposals_data[i * len] = bbox_center_x - bbox_width / 2;
    proposals_data[i * len + 1] = bbox_center_y - bbox_height / 2;
    proposals_data[i * len + 2] = bbox_center_x + bbox_width / 2;
    proposals_data[i * len + 3] = bbox_center_y + bbox_height / 2;
  }
}

template <class T>
static inline void ClipTiledBoxes(const Tensor &im_info, Tensor *boxes) {
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
static inline void FilterBoxes(Tensor *boxes, float min_size,
                               const Tensor &im_info, Tensor *keep) {
  const T *im_info_data = im_info.data<T>();
  T *boxes_data = boxes->mutable_data<T>();
  T im_scale = im_info_data[2];
  keep->Resize({boxes->dims()[0]});
  min_size = std::max(min_size, 1.0f);
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
  keep->Resize({keep_len});
}

template <class T>
static inline std::vector<std::pair<T, int>> GetSortedScoreIndex(
    const std::vector<T> &scores) {
  std::vector<std::pair<T, int>> sorted_indices;
  sorted_indices.reserve(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    sorted_indices.emplace_back(scores[i], i);
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices.begin(), sorted_indices.end(),
                   [](const std::pair<T, int> &a, const std::pair<T, int> &b) {
                     return a.first < b.first;
                   });
  return sorted_indices;
}

template <class T>
static inline T BBoxArea(const T *box, bool normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
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
static inline Tensor VectorToTensor(const std::vector<T> &selected_indices,
                                    int selected_num) {
  Tensor keep_nms;
  keep_nms.Resize({selected_num});
  auto *keep_data = keep_nms.mutable_data<T>();
  for (int i = 0; i < selected_num; ++i) {
    keep_data[i] = selected_indices[i];
  }
  return keep_nms;
}

template <class T>
static inline T JaccardOverlap(const T *box1, const T *box2, bool normalized) {
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
static inline Tensor NMS(Tensor *bbox, Tensor *scores, T nms_threshold,
                         float eta, int post_nms_num = 100) {
  int64_t num_boxes = bbox->dims()[0];
  // 4: [xmin ymin xmax ymax]
  int64_t box_size = bbox->dims()[1];

  std::vector<int8_t> scores_data(num_boxes);
  std::copy_n(scores->data<int8_t>(), num_boxes, scores_data.begin());
  std::vector<std::pair<int8_t, int>> sorted_indices =
      GetSortedScoreIndex<int8_t>(scores_data);

  std::vector<int> selected_indices;
  int selected_num = 0;
  T adaptive_threshold = nms_threshold;
  const T *bbox_data = bbox->data<T>();
  while ((sorted_indices.size() != 0) && (selected_num < post_nms_num)) {
    int idx = sorted_indices.back().second;
    bool flag = true;
    for (int kept_idx : selected_indices) {
      if (flag) {
        T overlap = JaccardOverlap<T>(bbox_data + idx * box_size,
                                      bbox_data + kept_idx * box_size, false);
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

template <typename T>
std::pair<Tensor, Tensor> ProposalForOneImage(
    const Tensor &im_info_slice, const Tensor &anchors, const Tensor &variances,
    const Tensor &bbox_deltas_slice,  // [M, 4]
    const Tensor &scores_slice,       // [N, 1]
    const Tensor &score_index, int pre_nms_top_n, int post_nms_top_n,
    float nms_thresh, float min_size, float eta) {
  auto *scores_data = scores_slice.data<int8_t>();
  // Sort index
  Tensor index_t;
  index_t.Resize({scores_slice.numel()});
  int *index = index_t.mutable_data<int>();
  std::memcpy(index, score_index.data<int32_t>(),
              scores_slice.numel() * sizeof(int));

  auto compare = [scores_data](const int64_t &i, const int64_t &j) {
    return scores_data[i] > scores_data[j];
  };

  if (pre_nms_top_n <= 0 || pre_nms_top_n >= scores_slice.numel()) {
    std::sort(index, index + scores_slice.numel(), compare);
  } else {
    std::nth_element(index, index + pre_nms_top_n, index + scores_slice.numel(),
                     compare);
    index_t.Resize({pre_nms_top_n});
  }

  Tensor scores_sel, bbox_sel, anchor_sel, var_sel;
  scores_sel.mutable_data<int8_t>({index_t.numel(), 1});
  bbox_sel.mutable_data<T>({index_t.numel(), 4});
  anchor_sel.mutable_data<T>({index_t.numel(), 4});
  var_sel.mutable_data<T>({index_t.numel(), 4});

  CPUGather<int8_t>(scores_slice, index_t, &scores_sel);
  CPUGather<T>(bbox_deltas_slice, index_t, &bbox_sel);
  CPUGather<T>(anchors, index_t, &anchor_sel);
  Tensor proposals;
  proposals.mutable_data<T>({index_t.numel(), 4});
  BoxCoder<T>(&anchor_sel, &bbox_sel, &proposals);

  ClipTiledBoxes<T>(im_info_slice, &proposals);

  Tensor keep;
  FilterBoxes<T>(&proposals, min_size, im_info_slice, &keep);

  Tensor scores_filter;
  bbox_sel.mutable_data<T>({keep.numel(), 4});
  scores_filter.mutable_data<int8_t>({keep.numel(), 1});

  CPUGather<T>(proposals, keep, &bbox_sel);
  CPUGather<int8_t>(scores_sel, keep, &scores_filter);
  if (nms_thresh <= 0) {
    return std::make_pair(bbox_sel, scores_filter);
  }

  Tensor keep_nms =
      NMS<T>(&bbox_sel, &scores_filter, nms_thresh, eta, post_nms_top_n);

  if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
    keep_nms.Resize({post_nms_top_n});
  }

  proposals.mutable_data<T>({keep_nms.numel(), 4});        // original
  scores_sel.mutable_data<int8_t>({keep_nms.numel(), 1});  // original

  CPUGather<T>(bbox_sel, keep_nms, &proposals);
  CPUGather<int8_t>(scores_filter, keep_nms, &scores_sel);
  return std::make_pair(proposals, scores_sel);
}

template <>
void ProposalKernel<FPGA, float>::Compute(const ProposalParam<FPGA> &param) {
  auto input_score = param.scores_;
  auto input_score_data = input_score->data<int8_t>();
  uint32_t score_n, score_height, score_width, score_channels;

  auto input_bbox = param.bbox_deltas_;
  auto input_bbox_data = input_bbox->data<int8_t>();
  uint32_t bbox_n, bbox_height, bbox_width, bbox_channels;

  score_n = (uint32_t)(input_score->dims()[0]);
  score_channels = (uint32_t)(input_score->dims()[1]);
  score_height = (uint32_t)(input_score->dims()[2]);
  score_width = (uint32_t)(input_score->dims()[3]);

  bbox_n = (uint32_t)(input_bbox->dims()[0]);
  bbox_channels = (uint32_t)(input_bbox->dims()[1]);
  bbox_height = (uint32_t)(input_bbox->dims()[2]);
  bbox_width = (uint32_t)(input_bbox->dims()[3]);

  int64_t amount_per_side = score_width * score_height;

  int alignedCW =
      fpga::align_to_x(score_width * score_channels, IMAGE_ALIGNMENT);
  int unalignedCW = score_width * score_channels;
  fpga::fpga_invalidate(input_score_data,
                        score_height * alignedCW * sizeof(int8_t));

  Tensor score_tensor = *input_score;
  for (int h = 0; h < score_height; h++) {
    for (int w = 0; w < score_width; w++) {
      for (int c = 0; c < score_channels; ++c) {
        int dstidx = h * unalignedCW + w * score_channels + c;
        int srcidx = h * alignedCW + w * score_channels + c;
        score_tensor.data<int8_t>()[dstidx] = input_score_data[srcidx];
      }
    }
  }

  amount_per_side = bbox_width * bbox_height;
  alignedCW = fpga::align_to_x(bbox_width * bbox_channels, IMAGE_ALIGNMENT);
  unalignedCW = bbox_width * bbox_channels;
  fpga::fpga_invalidate(input_bbox_data,
                        bbox_height * alignedCW * sizeof(int8_t));

  auto bbox_tensor = param.float_bbox.get();
  for (int h = 0; h < bbox_height; h++) {
    for (int w = 0; w < bbox_width; w++) {
      for (int c = 0; c < bbox_channels; ++c) {
        int dstidx = h * unalignedCW + w * bbox_channels + c;
        int srcidx = h * alignedCW + w * bbox_channels + c;
        bbox_tensor->data<float>()[dstidx] =
            (static_cast<int>(input_bbox_data[srcidx])) / 127.0 *
            input_bbox->scale[0];
      }
    }
  }
  auto *im_info = param.im_info_;
  auto anchors = *param.anchors_;
  auto variances = *param.variances_;

  auto *rpn_rois = param.rpn_rois_;
  auto *rpn_roi_probs = param.rpn_probs_;

  auto score_index = *(param.score_index_.get());

  int pre_nms_top_n = param.pre_nms_topn_;
  int post_nms_top_n = param.post_nms_topn_;

  float nms_thresh = param.nms_thresh_ / 2.0f;
  float min_size = param.min_size_;
  float eta = param.eta_;

  rpn_rois->mutable_data<float>({bbox_tensor->numel() / 4, 4});
  rpn_roi_probs->mutable_data<int8_t>({input_score->numel() / 4, 1});
  framework::LoD lod;
  lod.resize(1);
  auto &lod0 = lod[0];
  lod0.push_back(0);
  anchors.Resize({anchors.numel() / 4, 4});
  variances.Resize({variances.numel() / 4, 4});

  int64_t num_proposals = 0;
  for (int64_t i = 0; i < score_n; ++i) {
    Tensor im_info_slice = im_info->Slice(i, i + 1);
    Tensor bbox_deltas_slice = (*bbox_tensor).Slice(i, i + 1);
    Tensor scores_slice = score_tensor.Slice(i, i + 1);

    bbox_deltas_slice.Resize({bbox_height * bbox_width * bbox_channels / 4, 4});
    scores_slice.Resize({score_height * score_width * score_channels, 1});
    std::pair<Tensor, Tensor> tensor_pair = ProposalForOneImage<float>(
        im_info_slice, anchors, variances, bbox_deltas_slice, scores_slice,
        score_index, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size, eta);
    Tensor &proposals = tensor_pair.first;
    Tensor &scores = tensor_pair.second;

    AppendProposals(rpn_rois, 4 * num_proposals, proposals);
    AppendProposals(rpn_roi_probs, num_proposals, scores);
    num_proposals += proposals.dims()[0];
    lod0.push_back(num_proposals);
  }
  rpn_rois->set_lod(lod);
  rpn_roi_probs->set_lod(lod);
  rpn_rois->Resize({num_proposals, 4});
  rpn_roi_probs->Resize({num_proposals, 1});
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // PROPOSAL_OP
