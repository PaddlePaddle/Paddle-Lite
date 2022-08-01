// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/generate_proposals_compute.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void GenerateProposalsCompute::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto anchors_numel = param.Anchors->numel();
  num_guard_ = TargetWrapperXPU::MallocScratchPad(2 * sizeof(int));
  box_sel_guard_ =
      TargetWrapperXPU::MallocScratchPad(anchors_numel * 6 * sizeof(int));
  scores_sel_guard_ =
      TargetWrapperXPU::MallocScratchPad(anchors_numel / 2 * sizeof(float));
  index_sel_guard_ =
      TargetWrapperXPU::MallocScratchPad(anchors_numel / 2 * sizeof(float));
  trans_scores_guard_ =
      TargetWrapperXPU::MallocScratchPad(param.Scores->numel() * sizeof(float));
  trans_deltas_guard_ = TargetWrapperXPU::MallocScratchPad(
      param.BboxDeltas->numel() * sizeof(float));
  im_info_guard_ =
      TargetWrapperXPU::MallocScratchPad(param.ImInfo->numel() * sizeof(float));
}

void GenerateProposalsCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto* scores = param.Scores;              // N * A * H * W
  auto* bbox_deltas = param.BboxDeltas;     // N * 4A * H * W
  auto* im_info = param.ImInfo;             // N * 3
  auto* anchors = param.Anchors;            // H * W * A * 4
  auto* variances = param.Variances;        // H * W * A * 4
  auto* rpn_rois = param.RpnRois;           // A * 4
  auto* rpn_roi_probs = param.RpnRoiProbs;  // A * 1
  int pre_nms_top_n = param.pre_nms_topN;
  int post_nms_top_n = param.post_nms_topN;
  float nms_thresh = param.nms_thresh;
  float min_size = param.min_size;
  float eta = param.eta;
  if (std::fabs(eta - 1.0f) > 1e-7) {
    LOG(FATAL) << "XPU Generate Proposals Don't Support Adaptive NMS.";
  }

  auto& scores_dim = scores->dims();
  int num = static_cast<int>(scores_dim[0]);
  int c_score = static_cast<int>(scores_dim[1]);
  int h_score = static_cast<int>(scores_dim[2]);
  int w_score = static_cast<int>(scores_dim[3]);
  auto& bbox_dim = bbox_deltas->dims();
  int c_bbox = static_cast<int>(bbox_dim[1]);
  int h_bbox = static_cast<int>(bbox_dim[2]);
  int w_bbox = static_cast<int>(bbox_dim[3]);

  rpn_rois->Resize({bbox_deltas->numel() / 4, 4});
  rpn_roi_probs->Resize({scores->numel(), 1});
  // transpose
  trans_scores_guard_->Reserve(scores->numel() * sizeof(float));
  trans_deltas_guard_->Reserve(bbox_deltas->numel() * sizeof(float));
  float* trans_scores = reinterpret_cast<float*>(trans_scores_guard_->addr_);
  float* trans_deltas = reinterpret_cast<float*>(trans_deltas_guard_->addr_);
  int r = xdnn::transpose<float>(ctx.GetRawContext(),
                                 bbox_deltas->data<float>(),
                                 trans_deltas,
                                 {num, c_bbox, h_bbox, w_bbox},
                                 {0, 2, 3, 1});
  CHECK_EQ(r, 0);
  r = xdnn::transpose<float>(ctx.GetRawContext(),
                             scores->data<float>(),
                             trans_scores,
                             {num, c_score, h_score, w_score},
                             {0, 2, 3, 1});
  CHECK_EQ(r, 0);
  LoD lod;
  lod.resize(1);
  auto& lod0 = lod[0];
  lod0.push_back(0);
  std::vector<int64_t> tmp_lod;
  std::vector<int64_t> tmp_num;
  int64_t num_proposals = 0;
  float* rpn_rois_ptr = rpn_rois->mutable_data<float>(TARGET(kXPU));
  float* rpn_roi_probs_ptr = rpn_roi_probs->mutable_data<float>(TARGET(kXPU));
  int M = c_score * h_score * w_score;
  int K = std::min(pre_nms_top_n, M);

  im_info_guard_->Reserve(im_info->numel() * sizeof(float));
  float* im_info_ptr = reinterpret_cast<float*>(im_info_guard_->addr_);
  XPU_CALL(xpu_memcpy(im_info_ptr,
                      im_info->data<float>(),
                      im_info->numel() * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  box_sel_guard_->Reserve(K * 6 * 4 * sizeof(float));
  scores_sel_guard_->Reserve(K * 2 * sizeof(float));
  index_sel_guard_->Reserve(K * 2 * sizeof(int));

  for (int64_t batch_idx = 0; batch_idx < num; batch_idx++) {
    // topK
    float* topk_scores =
        reinterpret_cast<float*>(scores_sel_guard_->addr_);  // K * 1
    int* topk_indices =
        reinterpret_cast<int*>(index_sel_guard_->addr_);  // K * 1
    float* topk_anchors =
        reinterpret_cast<float*>(box_sel_guard_->addr_);  // K * 4
    float* topk_vars = topk_anchors + K * 4;              // K * 4
    float* topk_deltas = topk_vars + K * 4;               // K * 4
    float* box_decoder_pros = topk_deltas + K * 4;
    float* box_clip_pros = box_decoder_pros;
    int* remove_small_boxes_idx = topk_indices + K;
    int* remove_small_boxes_n_keep = reinterpret_cast<int*>(num_guard_->addr_);
    float* props_after_filter = box_decoder_pros + K * 4;
    float* scores_after_filter = topk_scores + K;
    int* index_after_nms = remove_small_boxes_idx + K;

    // TODO(quwei) : Change TOPK Impl to XPU Version(k1)
    // Since XPU Topk Only Support K <= 512, Select CPU Version Right Now
    if ((K <= 512 && ctx.GetRawContext()->dev().type() == xdnn::kXPU1) ||
        (K <= 6400 && ctx.GetRawContext()->dev().type() == xdnn::kXPU2)) {
      r = xdnn::sorted_topk(ctx.GetRawContext(),
                            trans_scores + batch_idx * M,
                            topk_scores,
                            topk_indices,
                            1,
                            M,
                            K,
                            true);
    } else {
      std::vector<float> tmp_scores_cpu(M, 0);
      std::vector<int> topk_indices_cpu(K, 0);
      std::vector<float> topk_scores_cpu(K, 0);

      TargetWrapperXPU::MemcpySync(tmp_scores_cpu.data(),
                                   trans_scores + batch_idx * M,
                                   sizeof(float) * M,
                                   IoDirection::DtoH);

      xdnn::Context ctx_cpu(xdnn::kCPU);
      r = xdnn::sorted_topk(&ctx_cpu,
                            tmp_scores_cpu.data(),
                            topk_scores_cpu.data(),
                            topk_indices_cpu.data(),
                            1,
                            M,
                            K,
                            true);
      CHECK_EQ(r, 0);
      XPU_CALL(xpu_memcpy(topk_scores,
                          topk_scores_cpu.data(),
                          sizeof(float) * K,
                          XPUMemcpyKind::XPU_HOST_TO_DEVICE));
      XPU_CALL(xpu_memcpy(topk_indices,
                          topk_indices_cpu.data(),
                          sizeof(float) * K,
                          XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    }

    // gather
    r = xdnn::gather<float, int>(ctx.GetRawContext(),
                                 anchors->data<float>(),
                                 topk_indices,
                                 topk_anchors,
                                 {M, 4},
                                 K,
                                 0);
    CHECK_EQ(r, 0);
    r = xdnn::gather<float, int>(ctx.GetRawContext(),
                                 variances->data<float>(),
                                 topk_indices,
                                 topk_vars,
                                 {M, 4},
                                 K,
                                 0);
    CHECK_EQ(r, 0);
    r = xdnn::gather<float, int>(ctx.GetRawContext(),
                                 trans_deltas + batch_idx * M * 4,
                                 topk_indices,
                                 topk_deltas,
                                 {M, 4},
                                 K,
                                 0);
    CHECK_EQ(r, 0);
    // box_decoder
    r = xdnn::box_decoder<float>(ctx.GetRawContext(),
                                 topk_anchors,
                                 topk_vars,
                                 topk_deltas,
                                 box_decoder_pros,
                                 K,
                                 false);
    CHECK_EQ(r, 0);
    // box_clips
    r = xdnn::clip_box_to_image<float>(
        ctx.GetRawContext(),
        box_decoder_pros,
        box_clip_pros,
        K,
        im_info->data<float>()[batch_idx * 3],
        im_info->data<float>()[batch_idx * 3 + 1]);
    CHECK_EQ(r, 0);
    // box_remove_small
    r = xdnn::remove_small_boxes<float>(ctx.GetRawContext(),
                                        box_clip_pros,
                                        im_info_ptr + batch_idx * 3,
                                        remove_small_boxes_idx,
                                        remove_small_boxes_n_keep,
                                        K,
                                        min_size);
    CHECK_EQ(r, 0);
    // gather after remove_small_box
    int remove_small_boxes_n_keep_cpu = 0;
    TargetWrapperXPU::MemcpySync(&remove_small_boxes_n_keep_cpu,
                                 remove_small_boxes_n_keep,
                                 sizeof(int),
                                 IoDirection::DtoH);
    r = xdnn::gather<float, int>(ctx.GetRawContext(),
                                 box_clip_pros,
                                 remove_small_boxes_idx,
                                 props_after_filter,
                                 {K, 4},
                                 remove_small_boxes_n_keep_cpu,
                                 0);
    CHECK_EQ(r, 0);
    r = xdnn::gather<float, int>(ctx.GetRawContext(),
                                 topk_scores,
                                 remove_small_boxes_idx,
                                 scores_after_filter,
                                 {K, 1},
                                 remove_small_boxes_n_keep_cpu,
                                 0);
    CHECK_EQ(r, 0);
    // NMS
    int nms_n_keep_cpu = -1;
    r = xdnn::sorted_nms<float>(ctx.GetRawContext(),
                                props_after_filter,
                                index_after_nms,
                                nms_n_keep_cpu,
                                remove_small_boxes_n_keep_cpu,
                                nms_thresh);
    CHECK_EQ(r, 0);

    nms_n_keep_cpu = std::min(nms_n_keep_cpu, post_nms_top_n);
    // Gather After NMS
    r = xdnn::gather<float, int>(ctx.GetRawContext(),
                                 props_after_filter,
                                 index_after_nms,
                                 rpn_rois_ptr,
                                 {remove_small_boxes_n_keep_cpu, 4},
                                 nms_n_keep_cpu,
                                 0);
    CHECK_EQ(r, 0);
    rpn_rois_ptr = rpn_rois_ptr + nms_n_keep_cpu * 4;
    r = xdnn::gather<float, int>(ctx.GetRawContext(),
                                 scores_after_filter,
                                 index_after_nms,
                                 rpn_roi_probs_ptr,
                                 {remove_small_boxes_n_keep_cpu, 1},
                                 nms_n_keep_cpu,
                                 0);
    CHECK_EQ(r, 0);
    rpn_roi_probs_ptr = rpn_roi_probs_ptr + nms_n_keep_cpu;
    num_proposals += nms_n_keep_cpu;
    lod0.push_back(num_proposals);
    tmp_lod.push_back(num_proposals);
    tmp_num.push_back(nms_n_keep_cpu);
  }
  if (param.RpnRoisLod != nullptr) {
    param.RpnRoisLod->Resize(DDim(std::vector<DDim::value_type>({num})));
    int64_t* lod_data = param.RpnRoisLod->mutable_data<int64_t>();
    for (int i = 0; i < num; i++) {
      lod_data[i] = tmp_lod[i];
    }
  }

  if (param.RpnRoisNum != nullptr) {
    param.RpnRoisNum->Resize(DDim(std::vector<DDim::value_type>({num})));
    int64_t* num_data = param.RpnRoisNum->mutable_data<int64_t>();
    for (int i = 0; i < num; i++) {
      num_data[i] = tmp_num[i];
    }
  }
  rpn_rois->set_lod(lod);
  rpn_roi_probs->set_lod(lod);
  rpn_rois->Resize({num_proposals, 4});
  rpn_roi_probs->Resize({num_proposals, 1});
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(generate_proposals,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::GenerateProposalsCompute,
                     def)
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("BboxDeltas", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ImInfo", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Anchors", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Variances", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("RpnRois", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("RpnRoiProbs", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("RpnRoisLod",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("RpnRoisNum",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
