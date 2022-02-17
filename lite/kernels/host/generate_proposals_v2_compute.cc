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

#include "lite/kernels/host/generate_proposals_v2_compute.h"
#include "lite/backends/host/math/bbox_util.h"
#include "lite/backends/host/math/gather.h"
#include "lite/backends/host/math/nms_util.h"
#include "lite/backends/host/math/transpose.h"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

static std::pair<Tensor, Tensor> ProposalForOneImage(
    const Tensor &im_shape_slice,
    const Tensor &anchors,
    const Tensor &variances,          // H * W * A * 4
    const Tensor &bbox_deltas_slice,  // [A, 4]
    const Tensor &scores_slice,       // [A, 1]
    int pre_nms_top_n,
    int post_nms_top_n,
    float nms_thresh,
    float min_size,
    float eta,
    bool pixel_offset = true) {
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
  lite::host::math::Gather<float>(scores_slice, index_t, &scores_sel);
  lite::host::math::Gather<float>(bbox_deltas_slice, index_t, &bbox_sel);
  lite::host::math::Gather<float>(anchors, index_t, &anchor_sel);
  lite::host::math::Gather<float>(variances, index_t, &var_sel);

  Tensor proposals;
  proposals.Resize(std::vector<int64_t>({index_t.numel(), 4}));
  lite::host::math::BoxCoder<float>(
      &anchor_sel, &bbox_sel, &var_sel, &proposals, pixel_offset);

  lite::host::math::ClipTiledBoxes<float>(
      im_shape_slice, proposals, &proposals, false, pixel_offset);

  Tensor keep;
  lite::host::math::FilterBoxes<float>(
      &proposals, min_size, im_shape_slice, false, &keep, pixel_offset);
  // Handle the case when there is no keep index left
  if (keep.numel() == 0) {
    Tensor scores_filter;
    scores_filter.Resize(std::vector<int64_t>({1, 1}));
    bbox_sel.Resize(std::vector<int64_t>({1, 4}));
    auto *scores_filter_data = scores_filter.mutable_data<float>();
    for (size_t i = 0; i < scores_filter.numel(); i++) {
      scores_filter_data[i] = 0;
    }
    auto *bbox_sel_data = bbox_sel.mutable_data<float>();
    for (size_t i = 0; i < scores_filter.numel(); i++) {
      bbox_sel_data[i] = 0;
    }
    return std::make_pair(bbox_sel, scores_filter);
  }

  Tensor scores_filter;
  scores_filter.Resize(std::vector<int64_t>({keep.numel(), 1}));
  bbox_sel.Resize(std::vector<int64_t>({keep.numel(), 4}));
  lite::host::math::Gather<float>(scores_sel, keep, &scores_filter);
  lite::host::math::Gather<float>(proposals, keep, &bbox_sel);
  if (nms_thresh <= 0) {
    return std::make_pair(bbox_sel, scores_filter);
  }

  Tensor keep_nms = lite::host::math::NMS<float>(
      &bbox_sel, &scores_filter, nms_thresh, eta, pixel_offset);
  if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
    keep_nms.Resize(std::vector<int64_t>({post_nms_top_n}));
  }
  proposals.Resize(std::vector<int64_t>({keep_nms.numel(), 4}));
  scores_sel.Resize(std::vector<int64_t>({keep_nms.numel(), 1}));
  lite::host::math::Gather<float>(bbox_sel, keep_nms, &proposals);
  lite::host::math::Gather<float>(scores_filter, keep_nms, &scores_sel);
  return std::make_pair(proposals, scores_sel);
}

void GenerateProposalsV2Compute::Run() {
  auto &param = Param<operators::GenerateProposalsV2Param>();
  auto *scores = param.Scores;              // N * A * H * W
  auto *bbox_deltas = param.BboxDeltas;     // N * 4A * H * W
  auto *im_shape = param.ImShape;           // N * 3
  auto *anchors = param.Anchors;            // H * W * A * 4
  auto *variances = param.Variances;        // H * W * A * 4
  auto *rpn_rois = param.RpnRois;           // A * 4
  auto *rpn_roi_probs = param.RpnRoiProbs;  // A * 1
  int pre_nms_top_n = param.pre_nms_topN;
  int post_nms_top_n = param.post_nms_topN;
  float nms_thresh = param.nms_thresh;
  float min_size = param.min_size;
  float eta = param.eta;
  bool pixel_offset = param.pixel_offset;

  auto &scores_dim = scores->dims();
  int64_t num = scores_dim[0];
  int64_t c_score = scores_dim[1];
  int64_t h_score = scores_dim[2];
  int64_t w_score = scores_dim[3];
  auto &bbox_dim = bbox_deltas->dims();
  int64_t c_bbox = bbox_dim[1];
  int64_t h_bbox = bbox_dim[2];
  int64_t w_bbox = bbox_dim[3];

  rpn_rois->Resize({bbox_deltas->numel() / 4, 4});
  rpn_roi_probs->Resize(std::vector<int64_t>({scores->numel(), 1}));

  Tensor bbox_deltas_swap, scores_swap;
  scores_swap.Resize(std::vector<int64_t>({num, h_score, w_score, c_score}));
  bbox_deltas_swap.Resize(std::vector<int64_t>({num, h_bbox, w_bbox, c_bbox}));
  std::vector<int> orders({0, 2, 3, 1});
  lite::host::math::Transpose<float>(*scores, &scores_swap, orders);
  lite::host::math::Transpose<float>(*bbox_deltas, &bbox_deltas_swap, orders);
  LoD lod;
  lod.resize(1);
  auto &lod0 = lod[0];
  lod0.push_back(0);
  anchors->Resize(std::vector<int64_t>({anchors->numel() / 4, 4}));
  variances->Resize(std::vector<int64_t>({variances->numel() / 4, 4}));
  std::vector<int64_t> tmp_lod;
  std::vector<int64_t> tmp_num;

  int64_t num_proposals = 0;
  for (int64_t i = 0; i < num; ++i) {
    Tensor im_shape_slice = im_shape->Slice<float>(i, i + 1);
    Tensor bbox_deltas_slice = bbox_deltas_swap.Slice<float>(i, i + 1);
    Tensor scores_slice = scores_swap.Slice<float>(i, i + 1);

    bbox_deltas_slice.Resize(
        std::vector<int64_t>({c_bbox * h_bbox * w_bbox / 4, 4}));
    scores_slice.Resize(std::vector<int64_t>({c_score * h_score * w_score, 1}));
    std::pair<Tensor, Tensor> tensor_pair =
        ProposalForOneImage(im_shape_slice,
                            *anchors,
                            *variances,
                            bbox_deltas_slice,
                            scores_slice,
                            pre_nms_top_n,
                            post_nms_top_n,
                            nms_thresh,
                            min_size,
                            eta,
                            pixel_offset);
    Tensor &proposals = tensor_pair.first;
    Tensor &scores = tensor_pair.second;
    lite::host::math::AppendTensor<float>(
        rpn_rois, 4 * num_proposals, proposals);
    lite::host::math::AppendTensor<float>(rpn_roi_probs, num_proposals, scores);

    num_proposals += proposals.dims()[0];
    lod0.push_back(num_proposals);
    tmp_lod.push_back(num_proposals);
    tmp_num.push_back(proposals.dims()[0]);
  }

  if (param.RpnRoisLod != nullptr) {
    param.RpnRoisLod->Resize(DDim(std::vector<DDim::value_type>({num})));
    int64_t *lod_data = param.RpnRoisLod->mutable_data<int64_t>();
    for (int i = 0; i < num; i++) {
      lod_data[i] = tmp_lod[i];
    }
  }

  if (param.RpnRoisNum != nullptr) {
    param.RpnRoisNum->Resize(DDim(std::vector<DDim::value_type>({num})));
    int32_t *num_data = param.RpnRoisNum->mutable_data<int32_t>();
    for (int i = 0; i < num; i++) {
      num_data[i] = tmp_num[i];
    }
  }
  rpn_rois->set_lod(lod);
  rpn_roi_probs->set_lod(lod);
  rpn_rois->Resize({num_proposals, 4});
  rpn_roi_probs->Resize({num_proposals, 1});
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(generate_proposals_v2,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::GenerateProposalsV2Compute,
                     def)
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("BboxDeltas", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("ImShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Anchors", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Variances", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("RpnRois", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("RpnRoiProbs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("RpnRoisLod",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("RpnRoisNum",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
