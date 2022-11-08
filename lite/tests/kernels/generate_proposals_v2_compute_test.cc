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

#include <gtest/gtest.h>
#include <fstream>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"

// this function is modified from
// lite/tests/kernels/anchor_generator_compute_test.cc
void anchor_generatre(int n,
                      int c,
                      int h,
                      int w,
                      std::vector<float> anchor_sizes_,
                      std::vector<float> aspect_ratios_,
                      std::vector<float> stride_,
                      std::vector<float> variances_,
                      float offset_,
                      paddle::lite::Tensor* vars,
                      paddle::lite::Tensor* anchors) {
  auto input_dims_ = paddle::lite::DDim(std::vector<int64_t>({n, c, h, w}));
  int num_anchors = anchor_sizes_.size() * aspect_ratios_.size();
  std::vector<int64_t> output_shape(
      {input_dims_[2], input_dims_[3], num_anchors, 4});
  paddle::lite::DDim output_dims(output_shape);
  anchors->Resize(output_dims);
  vars->Resize(output_dims);
  auto* anchors_data = anchors->mutable_data<float>();
  auto* vars_data = vars->mutable_data<float>();

  int feature_height = input_dims_[2];
  int feature_width = input_dims_[3];
  float stride_width = stride_[0];
  float stride_height = stride_[1];
  for (int h_idx = 0; h_idx < feature_height; ++h_idx) {
    for (int w_idx = 0; w_idx < feature_width; ++w_idx) {
      float x_ctr = (w_idx * stride_width) + offset_ * (stride_width - 1);
      float y_ctr = (h_idx * stride_height) + offset_ * (stride_height - 1);
      float area, area_ratios;
      float base_w, base_h;
      float scale_w, scale_h;
      float anchor_width, anchor_height;
      auto* anchors_data_hw = anchors_data +
                              h_idx * feature_width * num_anchors * 4 +
                              w_idx * num_anchors * 4;
      for (size_t r = 0; r < aspect_ratios_.size(); ++r) {
        auto ar = aspect_ratios_[r];
        auto* anchors_data_r = anchors_data_hw + r * anchor_sizes_.size() * 4;
        for (size_t s = 0; s < anchor_sizes_.size(); ++s) {
          auto anchor_size = anchor_sizes_[s];
          area = stride_width * stride_height;
          area_ratios = area / ar;
          base_w = round(sqrt(area_ratios));
          base_h = round(base_w * ar);
          scale_w = anchor_size / stride_width;
          scale_h = anchor_size / stride_height;
          anchor_width = scale_w * base_w;
          anchor_height = scale_h * base_h;
          anchors_data_r[s * 4 + 0] = (x_ctr - 0.5 * (anchor_width - 1));
          anchors_data_r[s * 4 + 1] = (y_ctr - 0.5 * (anchor_height - 1));
          anchors_data_r[s * 4 + 2] = (x_ctr + 0.5 * (anchor_width - 1));
          anchors_data_r[s * 4 + 3] = (y_ctr + 0.5 * (anchor_height - 1));
        }
      }
    }
  }

  for (int h = 0; h < feature_height; h++) {
    for (int w = 0; w < feature_width; w++) {
      for (int n = 0; n < num_anchors; n++) {
        auto vars_data_i = vars_data + h * feature_width * num_anchors * 4 +
                           w * num_anchors * 4 + n * 4;
        for (int i = 0; i < 4; i++) {
          vars_data_i[i] = variances_[i];
        }
      }
    }
  }
}

namespace paddle {
namespace lite {
static const double kBBoxClipDefault = std::log(1000.0 / 16.0);

static void permute(const Tensor& input,
                    Tensor* output,
                    const std::vector<int>& orders) {
  auto in_dims = input.dims();
  auto out_dims = output->dims();
  int num_axes = in_dims.size();
  int count = in_dims.production();

  const float* din = input.data<float>();
  float* dout = output->mutable_data<float>();
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

class GenerateProposalsV2ComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string Scores_ = "Scores";
  std::string BboxDeltas_ = "BboxDeltas";
  std::string ImShape_ = "ImShape";
  std::string Anchors_ = "Anchors";
  std::string Variances_ = "Variances";
  int pre_nms_topN_ = 12000;
  int post_nms_topN_ = 5000;
  float nms_thresh_ = 0.7;
  float min_size_ = 0.5;
  float eta_ = 1;
  bool pixel_offset_ = true;
  std::string RpnRois_ = "RpnRois";
  std::string RpnRoiProbs_ = "RpnRoiProbs";
  std::string RpnRoisLod_ = "RpnRoisLod";
  bool test_v18_api_ = false;

 public:
  GenerateProposalsV2ComputeTester(const Place& place,
                                   const std::string& alias,
                                   bool test_v18_api)
      : TestCase(place, alias), test_v18_api_(test_v18_api) {}

  template <typename T, typename IndexT = int>
  void CPUGather(const Tensor& src, const Tensor& index, Tensor* output) {
    if (index.dims().size() == 2) {
      // index.dims()[1] should be 1 when index.dims().size() = 2
      CHECK_EQ(index.dims()[1], 1);
    } else {
      // index.dims().size() should be 1 or 2 in gather_op
      CHECK_EQ(index.dims().size(), 1);
    }
    int64_t index_size = index.dims()[0];

    auto src_dims = src.dims();

    const T* p_src = src.data<T>();
    const IndexT* p_index = index.data<IndexT>();
    T* p_output = output->template mutable_data<T>();

    // slice size
    int slice_size = 1;
    for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

    const size_t slice_bytes = slice_size * sizeof(T);

    for (int64_t i = 0; i < index_size; ++i) {
      IndexT index_ = p_index[i];
      memcpy(
          p_output + i * slice_size, p_src + index_ * slice_size, slice_bytes);
    }
  }

  template <class T>
  static inline void BoxCoder(Tensor* all_anchors,
                              Tensor* bbox_deltas,
                              Tensor* variances,
                              Tensor* proposals,
                              const bool pixel_offset) {
    T* proposals_data = proposals->mutable_data<T>();

    int64_t row = all_anchors->dims()[0];
    int64_t len = all_anchors->dims()[1];

    auto* bbox_deltas_data = bbox_deltas->data<T>();
    auto* anchor_data = all_anchors->data<T>();
    const T* variances_data = nullptr;
    if (variances) {
      variances_data = variances->data<T>();
    }

    T offset = pixel_offset ? static_cast<T>(1.0) : 0;
    for (int64_t i = 0; i < row; ++i) {
      T anchor_width = anchor_data[i * len + 2] - anchor_data[i * len] + offset;
      T anchor_height =
          anchor_data[i * len + 3] - anchor_data[i * len + 1] + offset;

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
      proposals_data[i * len + 2] = bbox_center_x + bbox_width / 2 - offset;
      proposals_data[i * len + 3] = bbox_center_y + bbox_height / 2 - offset;
    }
    // return proposals;
  }

  template <class T>
  static inline void ClipTiledBoxes(const Tensor& im_info,
                                    Tensor* boxes,
                                    const bool is_scale,
                                    const bool pixel_offset) {
    T* boxes_data = boxes->mutable_data<T>();
    const T* im_info_data = im_info.data<T>();
    T offset = pixel_offset ? static_cast<T>(1) : 0;
    T zero(0);
    for (int64_t i = 0; i < boxes->numel(); ++i) {
      if (i % 4 == 0) {
        boxes_data[i] =
            std::max(std::min(boxes_data[i], im_info_data[1] - offset), zero);
      } else if (i % 4 == 1) {
        boxes_data[i] =
            std::max(std::min(boxes_data[i], im_info_data[0] - offset), zero);
      } else if (i % 4 == 2) {
        boxes_data[i] =
            std::max(std::min(boxes_data[i], im_info_data[1] - offset), zero);
      } else {
        boxes_data[i] =
            std::max(std::min(boxes_data[i], im_info_data[0] - offset), zero);
      }
    }
  }

  template <class T>
  static inline void FilterBoxes(Tensor* boxes,
                                 float min_size,
                                 const Tensor& im_info,
                                 bool is_scale,
                                 Tensor* keep,
                                 bool pixel_offset) {
    const T* im_info_data = im_info.data<T>();
    T* boxes_data = boxes->mutable_data<T>();
    keep->Resize({boxes->dims()[0]});
    min_size = std::max(min_size, 1.0f);
    int* keep_data = keep->mutable_data<int>();
    T offset = pixel_offset ? static_cast<T>(1.0) : 0;

    int keep_len = 0;
    for (int i = 0; i < boxes->dims()[0]; ++i) {
      T ws = boxes_data[4 * i + 2] - boxes_data[4 * i] + offset;
      T hs = boxes_data[4 * i + 3] - boxes_data[4 * i + 1] + offset;
      if (pixel_offset) {
        T x_ctr = boxes_data[4 * i] + ws / 2;
        T y_ctr = boxes_data[4 * i + 1] + hs / 2;

        if (ws >= min_size && hs >= min_size && x_ctr <= im_info_data[1] &&
            y_ctr <= im_info_data[0]) {
          keep_data[keep_len++] = i;
        }
      } else {
        if (ws >= min_size && hs >= min_size) {
          keep_data[keep_len++] = i;
        }
      }
    }
    keep->Resize({keep_len});
  }

  template <class T>
  static inline std::vector<std::pair<T, int>> GetSortedScoreIndex(
      const std::vector<T>& scores) {
    std::vector<std::pair<T, int>> sorted_indices;
    sorted_indices.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
      sorted_indices.emplace_back(scores[i], i);
    }
    // Sort the score pair according to the scores in descending order
    std::stable_sort(
        sorted_indices.begin(),
        sorted_indices.end(),
        [](const std::pair<T, int>& a, const std::pair<T, int>& b) {
          return a.first < b.first;
        });
    return sorted_indices;
  }

  template <class T>
  static inline T BBoxArea(const T* box, bool normalized) {
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
  static inline T JaccardOverlap(const T* box1,
                                 const T* box2,
                                 bool normalized) {
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
  static inline Tensor VectorToTensor(const std::vector<T>& selected_indices,
                                      int selected_num) {
    Tensor keep_nms;
    keep_nms.Resize({selected_num});
    auto* keep_data = keep_nms.mutable_data<T>();
    for (int i = 0; i < selected_num; ++i) {
      keep_data[i] = selected_indices[i];
    }
    return keep_nms;
  }

  template <class T>
  Tensor NMS(Tensor* bbox,
             Tensor* scores,
             T nms_threshold,
             float eta,
             const bool pixel_offset = true) {
    int64_t num_boxes = bbox->dims()[0];
    // 4: [xmin ymin xmax ymax]
    int64_t box_size = bbox->dims()[1];

    std::vector<T> scores_data(num_boxes);
    std::copy_n(scores->data<T>(), num_boxes, scores_data.begin());
    std::vector<std::pair<T, int>> sorted_indices =
        GetSortedScoreIndex<T>(scores_data);

    std::vector<int> selected_indices;
    int selected_num = 0;
    T adaptive_threshold = nms_threshold;
    const T* bbox_data = bbox->data<T>();
    while (sorted_indices.size() != 0) {
      int idx = sorted_indices.back().second;
      bool flag = true;
      for (int kept_idx : selected_indices) {
        if (flag) {
          T overlap = JaccardOverlap<T>(bbox_data + idx * box_size,
                                        bbox_data + kept_idx * box_size,
                                        !pixel_offset);
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

  template <class T>
  std::pair<Tensor, Tensor> ProposalForOneImage(
      const Tensor& im_info_slice,
      const Tensor& anchors,
      const Tensor& variances,
      const Tensor& bbox_deltas_slice,  // [M, 4]
      const Tensor& scores_slice,       // [N, 1]
      int pre_nms_top_n,
      int post_nms_top_n,
      float nms_thresh,
      float min_size,
      bool pixel_offset,
      float eta) {
    auto* scores_data = scores_slice.data<T>();

    // Sort index
    Tensor index_t;
    index_t.Resize({scores_slice.numel()});
    int* index = index_t.mutable_data<int>();
    for (int i = 0; i < scores_slice.numel(); ++i) {
      index[i] = i;
    }
    auto compare = [scores_data](const int64_t& i, const int64_t& j) {
      return scores_data[i] > scores_data[j];
    };

    if (pre_nms_top_n <= 0 || pre_nms_top_n >= scores_slice.numel()) {
      std::sort(index, index + scores_slice.numel(), compare);
    } else {
      std::nth_element(
          index, index + pre_nms_top_n, index + scores_slice.numel(), compare);
      index_t.Resize({pre_nms_top_n});
    }

    Tensor scores_sel, bbox_sel, anchor_sel, var_sel;
    scores_sel.Resize({index_t.numel(), 1});
    scores_sel.mutable_data<T>();
    bbox_sel.Resize({index_t.numel(), 4});
    bbox_sel.mutable_data<T>();
    anchor_sel.Resize({index_t.numel(), 4});
    anchor_sel.mutable_data<T>();
    var_sel.Resize({index_t.numel(), 4});
    var_sel.mutable_data<T>();

    CPUGather<T>(scores_slice, index_t, &scores_sel);
    CPUGather<T>(bbox_deltas_slice, index_t, &bbox_sel);
    CPUGather<T>(anchors, index_t, &anchor_sel);
    CPUGather<T>(variances, index_t, &var_sel);

    Tensor proposals;
    proposals.Resize({index_t.numel(), 4});
    proposals.mutable_data<T>();
    BoxCoder<T>(&anchor_sel, &bbox_sel, &var_sel, &proposals, pixel_offset);

    ClipTiledBoxes<T>(im_info_slice, &proposals, false, pixel_offset);

    Tensor keep;
    FilterBoxes<T>(
        &proposals, min_size, im_info_slice, false, &keep, pixel_offset);

    if (keep.numel() == 0) {
      Tensor scores_filter;
      scores_filter.Resize(std::vector<int64_t>({1, 1}));
      bbox_sel.Resize(std::vector<int64_t>({1, 4}));
      auto* scores_filter_data = scores_filter.mutable_data<float>();
      for (size_t i = 0; i < scores_filter.numel(); i++) {
        scores_filter_data[i] = 0;
      }
      auto* bbox_sel_data = bbox_sel.mutable_data<float>();
      for (size_t i = 0; i < scores_filter.numel(); i++) {
        bbox_sel_data[i] = 0;
      }
      return std::make_pair(bbox_sel, scores_filter);
    }

    Tensor scores_filter;
    bbox_sel.Resize({keep.numel(), 4});
    bbox_sel.mutable_data<T>();
    scores_filter.Resize({keep.numel(), 1});
    scores_filter.mutable_data<T>();
    CPUGather<T>(proposals, keep, &bbox_sel);
    CPUGather<T>(scores_sel, keep, &scores_filter);
    if (nms_thresh <= 0) {
      return std::make_pair(bbox_sel, scores_filter);
    }

    Tensor keep_nms =
        NMS<T>(&bbox_sel, &scores_filter, nms_thresh, eta, pixel_offset);

    if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
      keep_nms.Resize({post_nms_top_n});
    }
    proposals.Resize({keep_nms.numel(), 4});
    proposals.mutable_data<T>();
    scores_sel.Resize({keep_nms.numel(), 1});
    scores_sel.mutable_data<T>();
    CPUGather<T>(bbox_sel, keep_nms, &proposals);
    CPUGather<T>(scores_filter, keep_nms, &scores_sel);

    return std::make_pair(proposals, scores_sel);
  }

  template <class T>
  void AppendProposals(Tensor* dst, int64_t offset, const Tensor& src) {
    auto* out_data = dst->data<void>();
    auto* to_add_data = src.data<void>();
    size_t size_of_t = sizeof(T);
    offset *= size_of_t;
    std::memcpy(
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(out_data) + offset),
        to_add_data,
        src.numel() * size_of_t);
  }

  template <class T>
  void Compute(Scope* scope) {
    auto* scores = scope->FindTensor("Scores");
    auto* bbox_deltas = scope->FindTensor("BboxDeltas");
    auto* im_shape = scope->FindTensor("ImShape");
    auto anchors = scope->FindMutableTensor("Anchors");
    auto variances = scope->FindMutableTensor("Variances");

    auto* rpn_rois = scope->NewTensor("RpnRois");
    auto* rpn_roi_probs = scope->NewTensor("RpnRoiProbs");

    int pre_nms_top_n = pre_nms_topN_;
    int post_nms_top_n = post_nms_topN_;
    float nms_thresh = nms_thresh_;
    float min_size = min_size_;
    float eta = eta_;
    bool pixel_offset = pixel_offset_;

    auto& scores_dim = scores->dims();
    int64_t num = scores_dim[0];
    int64_t c_score = scores_dim[1];
    int64_t h_score = scores_dim[2];
    int64_t w_score = scores_dim[3];

    auto& bbox_dim = bbox_deltas->dims();
    int64_t c_bbox = bbox_dim[1];
    int64_t h_bbox = bbox_dim[2];
    int64_t w_bbox = bbox_dim[3];

    rpn_rois->Resize({bbox_deltas->numel() / 4, 4});
    rpn_rois->mutable_data<T>();
    rpn_roi_probs->Resize({scores->numel(), 1});
    rpn_roi_probs->mutable_data<T>();

    Tensor bbox_deltas_swap, scores_swap;
    bbox_deltas_swap.Resize({num, h_bbox, w_bbox, c_bbox});
    bbox_deltas_swap.mutable_data<T>();
    scores_swap.Resize({num, h_score, w_score, c_score});
    scores_swap.mutable_data<T>();

    std::vector<int> axis = {0, 2, 3, 1};
    permute(*bbox_deltas, &bbox_deltas_swap, axis);
    permute(*scores, &scores_swap, axis);

    LoD lod;
    lod.resize(1);
    auto& lod0 = lod[0];
    lod0.push_back(0);
    anchors->Resize({anchors->numel() / 4, 4});
    variances->Resize({variances->numel() / 4, 4});
    std::vector<int64_t> tmp_lod;

    int64_t num_proposals = 0;
    for (int64_t i = 0; i < num; ++i) {
      Tensor im_shape_slice = im_shape->Slice<float>(i, i + 1);
      Tensor bbox_deltas_slice = bbox_deltas_swap.Slice<float>(i, i + 1);
      Tensor scores_slice = scores_swap.Slice<float>(i, i + 1);

      bbox_deltas_slice.Resize({h_bbox * w_bbox * c_bbox / 4, 4});
      scores_slice.Resize({h_score * w_score * c_score, 1});

      std::pair<Tensor, Tensor> tensor_pair =
          ProposalForOneImage<float>(im_shape_slice,
                                     *anchors,
                                     *variances,
                                     bbox_deltas_slice,
                                     scores_slice,
                                     pre_nms_top_n,
                                     post_nms_top_n,
                                     nms_thresh,
                                     min_size,
                                     pixel_offset,
                                     eta);
      Tensor& proposals = tensor_pair.first;
      Tensor& scores = tensor_pair.second;

      AppendProposals<float>(rpn_rois, 4 * num_proposals, proposals);
      AppendProposals<float>(rpn_roi_probs, num_proposals, scores);
      num_proposals += proposals.dims()[0];
      lod0.push_back(num_proposals);
      tmp_lod.push_back(num_proposals);
    }
    if (test_v18_api_) {
      auto* rpn_rois_lod = scope->NewTensor(RpnRoisLod_);
      rpn_rois_lod->Resize({num});
      int64_t* lod_data = rpn_rois_lod->mutable_data<int64_t>();
      for (int i = 0; i < num; i++) {
        lod_data[i] = tmp_lod[i];
      }
    }
    rpn_rois->set_lod(lod);
    rpn_roi_probs->set_lod(lod);
    rpn_rois->Resize({num_proposals, 4});
    rpn_roi_probs->Resize({num_proposals, 1});
  }

  void RunBaseline(Scope* scope) override { Compute<float>(scope); }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("generate_proposals_v2");

    op_desc->SetInput("Scores", {Scores_});
    op_desc->SetInput("BboxDeltas", {BboxDeltas_});
    op_desc->SetInput("ImShape", {ImShape_});
    op_desc->SetInput("Anchors", {Anchors_});
    op_desc->SetInput("Variances", {Variances_});

    op_desc->SetAttr("pre_nms_topN", pre_nms_topN_);
    op_desc->SetAttr("post_nms_topN", post_nms_topN_);
    op_desc->SetAttr("nms_thresh", nms_thresh_);
    op_desc->SetAttr("min_size", min_size_);
    op_desc->SetAttr("eta", eta_);
    op_desc->SetAttr("pixel_offset", pixel_offset_);

    op_desc->SetOutput("RpnRois", {RpnRois_});
    op_desc->SetOutput("RpnRoiProbs", {RpnRoiProbs_});

    if (test_v18_api_) {
      op_desc->SetOutput("RpnRoisLod", {RpnRoisLod_});
    }
  }

  void PrepareData() override {
    int batch_size = 1;
    int input_channels = 20;
    int layer_h = 16;
    int layer_w = 16;

    Tensor variances;
    Tensor anchors;

    anchor_generatre(batch_size,
                     input_channels,
                     layer_h,
                     layer_w,
                     {16.0, 32.0},
                     {0.5, 1.0},
                     {16.0, 16.0},
                     {1.0, 1.0, 1.0, 1.0},
                     0.5,
                     &variances,
                     &anchors);

    // Anchors
    SetCommonTensor(Anchors_, anchors.dims(), anchors.data<float>());

    // Variances
    SetCommonTensor(Variances_, variances.dims(), variances.data<float>());

    // Scores
    {
      int num_anchors = anchors.dims()[2];
      DDim dims = DDim({batch_size, num_anchors, layer_h, layer_w});
      std::vector<float> data(dims.production(), 0);
      std::generate(data.begin(), data.end(), std::rand);
      SetCommonTensor(Scores_, dims, data.data());
    }

    // BboxDeltas
    {
      int num_anchors = anchors.dims()[2];
      DDim dims = DDim({batch_size, num_anchors * 4, layer_h, layer_w});
      std::vector<float> data(dims.production(), 0);
      std::generate(data.begin(), data.end(), std::rand);
      SetCommonTensor(BboxDeltas_, dims, data.data());
    }

    // ImInfo
    {
      DDim dims = DDim({3});
      std::vector<float> data{64, 64, 8};
      std::generate(data.begin(), data.end(), std::rand);
      SetCommonTensor(ImShape_, dims, data.data());
    }
  }
};

TEST(GenerateProposalsV2, precision) {
  Place place;
#if defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  for (bool test_v18_api : {true, false}) {
    std::unique_ptr<arena::TestCase> tester(
        new GenerateProposalsV2ComputeTester(place, "def", test_v18_api));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }
}

}  // namespace lite
}  // namespace paddle
