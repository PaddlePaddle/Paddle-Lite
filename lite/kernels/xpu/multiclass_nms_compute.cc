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
#include "lite/kernels/xpu/multiclass_nms_compute.h"
#include <algorithm>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void MulticlassNmsCompute::PrepareForRun() {
  std::vector<float> unnormalize_box_offset_cpu{0, 0, -1, -1};
  unnormalize_box_offset_guard_ =
      TargetWrapperXPU::MallocScratchPad(4 * sizeof(float));
  XPU_CALL(xpu_wait());
  XPU_CALL(xpu_memcpy(unnormalize_box_offset_guard_->addr_,
                      unnormalize_box_offset_cpu.data(),
                      sizeof(float) * 4,
                      XPU_HOST_TO_DEVICE));
  unnormalized_boxes_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);

  topk_index_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  topk_scores_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  xpu_score_thres_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  topk_scores_lower_bound_index_guard_ =
      TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  topk_boxes_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);

  nms_index_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  nms_scores_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  nms_class_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  nms_keep_box_num_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  nms_boxes_index_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);

  batch_box_index_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  merge_box_index_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  merge_index_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  merge_scores_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  merge_boxes_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  merge_class_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
  batch_offset_guard_ = TargetWrapperXPU::MallocScratchPad(1024 * 1024);
}

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

// boxes:  [n, n_boxes, 4]
// scores: [n, c, n_boxes]
void MulticlassNmsCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& op_ctx = this->ctx_->As<XPUContext>();
  xdnn::Context* ctx = op_ctx.GetRawContext();

  bool return_index = param.index ? true : false;
  auto score_dims = param.scores->dims();
  auto score_size = score_dims.size();
  auto has_roissum = param.rois_num != nullptr;
  auto return_rois_num = param.nms_rois_num != nullptr;
  auto rois_num = param.rois_num;

  int64_t batch_size = score_dims[0];
  int64_t box_dim = param.bboxes->dims()[2];
  CHECK_EQ(box_dim, 4) << "XPU Only Support Bounding Box with FOUR Coordinate.";
  int64_t out_dim = box_dim + 2;

  // attr
  float nms_thres = param.nms_threshold;
  float score_thres = param.score_threshold;
  int keep_top_k = param.keep_top_k;
  int nms_top_k = param.nms_top_k;
  int background_label = param.background_label;
  float nms_eta = param.nms_eta;
  CHECK_EQ(nms_eta, 1.0f) << "No XPU Support On Adaptive NMS.";

  int n;
  if (has_roissum) {
    n = score_size == 3 ? batch_size : rois_num->numel();
  } else {
    n = score_size == 3 ? batch_size : param.bboxes->lod().back().size() - 1;
  }

  const float* boxes = nullptr;
  const float* unnormalized_boxes = nullptr;
  const float* scores = nullptr;
  int c = score_dims[1];
  int n_boxes = -1;
  if (score_size == 3) {
    boxes = param.bboxes->data<float>();
    scores = param.scores->data<float>();
    n_boxes = score_dims[2];
  } else {
    // TODO(weihaoji): lod-like data format support
    LOG(FATAL) << "Unsupport MulticlassNMS Scores DataFormat For XPU.";
    // std::vector<uint64_t> boxes_lod;
    // if (has_roissum) {
    //   boxes_lod = GetNmsLodFromRoisNum(rois_num);
    // } else {
    //   boxes_lod = param.bboxes->lod().back();
    // }
  }

  int ret = -1;
  // box trans for normalized box to normalized box
  if (param.normalized) {
    unnormalized_boxes_guard_->Reserve(n * n_boxes * 4 * sizeof(float));
    unnormalized_boxes =
        reinterpret_cast<const float*>(unnormalized_boxes_guard_->addr_);
    ret = xdnn::broadcast_add<float>(
        ctx,
        boxes,
        reinterpret_cast<float*>(unnormalize_box_offset_guard_->addr_),
        const_cast<float*>(unnormalized_boxes),
        {n, n_boxes, 4},
        {1, 1, 4});
    CHECK_EQ(ret, 0);
  } else {
    unnormalized_boxes = boxes;
  }
  // STAGE 1. TOPK
  // topk_index:  int, [n * c, k]
  // topk_scores: float, [n * c, k]
  // xpu_score_thres: float, [n * c]
  // topk_scores_lower_bound_index: int , [n * c]
  // topk_boxes: float, [n ,c, k, 4], actual: [n, c, topk_box_num[n][c], 4] for
  // eatch class
  // topk_box_num: int, [n * c] (cpu) boxes num after topk and after score_thres
  // filt
  int k = nms_top_k < 0 ? n_boxes : std::min(n_boxes, nms_top_k);
  if (k > 512) {
    k = 512;
    // TODO(weihaoji) support XPU Topk with K more than 512
    LOG(WARNING) << "Run XPU TopK With K = 512."
                 << "Since K Value: " << k << "is not support right now";
  }
  topk_index_guard_->Reserve(n * c * k * sizeof(int));
  topk_scores_guard_->Reserve(n * c * k * sizeof(float));
  xpu_score_thres_guard_->Reserve(n * c * sizeof(float));
  topk_scores_lower_bound_index_guard_->Reserve(n * c * sizeof(int));
  topk_boxes_guard_->Reserve(n * c * k * 4 * sizeof(float));

  int* topk_index = reinterpret_cast<int*>(topk_index_guard_->addr_);
  float* topk_scores = reinterpret_cast<float*>(topk_scores_guard_->addr_);
  float* xpu_score_thres =
      reinterpret_cast<float*>(xpu_score_thres_guard_->addr_);
  int* topk_scores_lower_bound_index =
      reinterpret_cast<int*>(topk_scores_lower_bound_index_guard_->addr_);
  float* topk_boxes = reinterpret_cast<float*>(topk_boxes_guard_->addr_);
  std::vector<int> topk_box_num(n * c, 0);

  ret = xdnn::sorted_topk<float>(
      ctx, scores, topk_scores, topk_index, n * c, n_boxes, k);
  CHECK_EQ(ret, 0);
  ret = xdnn::constant<float>(ctx, xpu_score_thres, n * c, score_thres);
  CHECK_EQ(ret, 0);
  ret = xdnn::search_sorted<float, int>(ctx,
                                        topk_scores,
                                        xpu_score_thres,
                                        topk_scores_lower_bound_index,
                                        n * c,
                                        k,
                                        1,
                                        true,
                                        false);
  CHECK_EQ(ret, 0);
  XPU_CALL(xpu_wait());
  XPU_CALL(xpu_memcpy(topk_box_num.data(),
                      topk_scores_lower_bound_index,
                      sizeof(int) * n * c,
                      XPU_DEVICE_TO_HOST));
  // STAGE 2. NMS
  // nms_index [n * c, k]
  // nms_scores[n, c * k] -- xpu packed
  // nms_class[n, c * k]  -- xpu packed
  // nms_keep_box_num [n * c]
  // nms_boxes_index[n, c * k] -- xpu packed
  // nms_box_num [n * c] (cpu)
  // nms_box_num_lod[n][c + 1]
  nms_index_guard_->Reserve(n * c * k * sizeof(int));
  nms_scores_guard_->Reserve(n * c * k * sizeof(float));
  nms_class_guard_->Reserve(n * c * k * sizeof(float));
  nms_keep_box_num_guard_->Reserve(n * c * sizeof(int));
  nms_boxes_index_guard_->Reserve(n * c * k * sizeof(int));

  int* nms_index = reinterpret_cast<int*>(nms_index_guard_->addr_);
  float* nms_scores = reinterpret_cast<float*>(nms_scores_guard_->addr_);
  float* nms_class = reinterpret_cast<float*>(nms_class_guard_->addr_);
  int* nms_keep_box_num =
      reinterpret_cast<int*>(nms_keep_box_num_guard_->addr_);
  int* nms_boxes_index = reinterpret_cast<int*>(nms_boxes_index_guard_->addr_);
  std::vector<int> nms_box_num(n * c, 0);
  std::vector<std::vector<int>> nms_box_num_lod;

  ret = xdnn::constant<float>(ctx, nms_scores, n * c * k, 0.0f);
  CHECK_EQ(ret, 0);
  ret = xdnn::constant<int>(ctx, nms_keep_box_num, n * c, 0);
  CHECK_EQ(ret, 0);
  for (int batch_idx = 0; batch_idx < n; batch_idx++) {
    for (int class_idx = 0; class_idx < c; class_idx++) {
      if (topk_box_num[batch_idx * c + class_idx] <= 0 ||
          background_label == class_idx) {
        continue;
      }
      ret = xdnn::gather<float, int>(
          ctx,
          unnormalized_boxes + batch_idx * n_boxes * 4,
          topk_index + batch_idx * c * k + class_idx * k,
          topk_boxes + batch_idx * c * k * 4 + class_idx * k * 4,
          {n_boxes, 4},
          topk_box_num[batch_idx * c + class_idx],
          0);
      CHECK_EQ(ret, 0);

      ret = xdnn::sorted_nms<float>(
          ctx,
          topk_boxes + batch_idx * c * k * 4 + class_idx * k * 4,
          nms_index + batch_idx * c * k + class_idx * k,
          nms_keep_box_num + batch_idx * c + class_idx,
          topk_box_num[batch_idx * c + class_idx],
          nms_thres);
      CHECK_EQ(ret, 0);
    }
  }
  XPU_CALL(xpu_wait());
  XPU_CALL(xpu_memcpy(nms_box_num.data(),
                      nms_keep_box_num,
                      sizeof(int) * n * c,
                      XPU_DEVICE_TO_HOST));
  for (int batch_idx = 0; batch_idx < n; batch_idx++) {
    std::vector<int> cur_batch_box_num_lod(c + 1, 0);
    for (int class_idx = 0; class_idx < c; class_idx++) {
      cur_batch_box_num_lod[class_idx + 1] =
          cur_batch_box_num_lod[class_idx] +
          nms_box_num[batch_idx * c + class_idx];
    }
    nms_box_num_lod.push_back(cur_batch_box_num_lod);
  }
  for (int batch_idx = 0; batch_idx < n; batch_idx++) {
    for (int class_idx = 0; class_idx < c; class_idx++) {
      if (nms_box_num[batch_idx * c + class_idx] <= 0) {
        continue;
      }
      ret = xdnn::gather<float, int>(
          ctx,
          topk_scores + batch_idx * c * k + class_idx * k,
          nms_index + batch_idx * c * k + class_idx * k,
          nms_scores + batch_idx * c * k +
              nms_box_num_lod[batch_idx][class_idx],
          {topk_box_num[batch_idx * c + class_idx]},
          nms_box_num[batch_idx * c + class_idx],
          0);
      CHECK_EQ(ret, 0);
      ret =
          xdnn::gather<int, int>(ctx,
                                 topk_index + batch_idx * c * k + class_idx * k,
                                 nms_index + batch_idx * c * k + class_idx * k,
                                 nms_boxes_index + batch_idx * c * k +
                                     nms_box_num_lod[batch_idx][class_idx],
                                 {topk_box_num[batch_idx * c + class_idx]},
                                 nms_box_num[batch_idx * c + class_idx],
                                 0);
      CHECK_EQ(ret, 0);
      ret = xdnn::constant<float>(
          ctx,
          nms_class + batch_idx * c * k + nms_box_num_lod[batch_idx][class_idx],
          nms_box_num[batch_idx * c + class_idx],
          static_cast<float>(class_idx));
      CHECK_EQ(ret, 0);
    }
  }
  // STAGE 3. MERGE
  // merge_box_index[n * merge_k]
  // merge_scores[n * merge_k]
  // merge_index[n * merge_k]
  // merge_boxes[n * merge_k * 4]
  // merge_class[n * merge_k]
  // batch_offset[n]
  // batch_box_index[n * merge_k]
  // merge_boxes_num_lod[n + 1] (cpu)
  int merge_k = 0;
  std::vector<int> merge_boxes_num_lod(n + 1, 0);
  for (int i = 0; i < n; i++) {
    int cur_box_num = keep_top_k < 0
                          ? nms_box_num_lod[i].back()
                          : std::min(nms_box_num_lod[i].back(), keep_top_k);
    merge_boxes_num_lod[i + 1] = merge_boxes_num_lod[i] + cur_box_num;
    merge_k = std::max(merge_k, cur_box_num);
  }

  float* out = nullptr;
  if (merge_boxes_num_lod.back() > 0) {
    param.out->Resize({merge_boxes_num_lod.back(), out_dim});
    out = param.out->mutable_data<float>(TARGET(kXPU));
  }

  batch_box_index_guard_->Reserve(n * merge_k * sizeof(int));
  merge_box_index_guard_->Reserve(n * merge_k * sizeof(int));
  merge_index_guard_->Reserve(n * merge_k * sizeof(int));
  merge_scores_guard_->Reserve(n * merge_k * sizeof(float));
  merge_boxes_guard_->Reserve(n * merge_k * 4 * sizeof(float));
  merge_class_guard_->Reserve(n * merge_k * sizeof(float));
  batch_offset_guard_->Reserve(n * sizeof(int));

  int* batch_box_index = reinterpret_cast<int*>(batch_box_index_guard_->addr_);
  int* merge_box_index = reinterpret_cast<int*>(merge_box_index_guard_->addr_);
  int* merge_index = reinterpret_cast<int*>(merge_index_guard_->addr_);
  float* merge_scores = reinterpret_cast<float*>(merge_scores_guard_->addr_);
  float* merge_boxes = reinterpret_cast<float*>(merge_boxes_guard_->addr_);
  float* merge_class = reinterpret_cast<float*>(merge_class_guard_->addr_);
  int* batch_offset = reinterpret_cast<int*>(batch_offset_guard_->addr_);

  ret = xdnn::constant<float>(ctx, merge_scores, n * merge_k, 0.0f);
  CHECK_EQ(ret, 0);
  ret = xdnn::sorted_topk<float>(
      ctx, nms_scores, merge_scores, merge_index, n, c * k, merge_k);
  CHECK_EQ(ret, 0);

  for (int batch_idx = 0; batch_idx < n; batch_idx++) {
    ret = xdnn::gather<int, int>(
        ctx,
        nms_boxes_index + batch_idx * c * k,
        merge_index + batch_idx * merge_k,
        merge_box_index + batch_idx * merge_k,
        {nms_box_num_lod[batch_idx].back()},
        std::min(merge_k, nms_box_num_lod[batch_idx].back()),
        0);
    CHECK_EQ(ret, 0);
    ret = xdnn::gather<float, int>(
        ctx,
        nms_class + batch_idx * c * k,
        merge_index + batch_idx * merge_k,
        merge_class + batch_idx * merge_k,
        {nms_box_num_lod[batch_idx].back()},
        std::min(merge_k, nms_box_num_lod[batch_idx].back()),
        0);
    CHECK_EQ(ret, 0);
  }
  ret = xdnn::range<int>(ctx, batch_offset, 0, n_boxes, n);
  CHECK_EQ(ret, 0);
  ret = xdnn::broadcast_add<int>(ctx,
                                 merge_box_index,
                                 batch_offset,
                                 batch_box_index,
                                 {n, merge_k},
                                 {n, 1});
  CHECK_EQ(ret, 0);
  for (int batch_idx = 0; batch_idx < n; batch_idx++) {
    int current_box_num =
        merge_boxes_num_lod[batch_idx + 1] - merge_boxes_num_lod[batch_idx];
    if (current_box_num <= 0) {
      continue;
    }
    ret = xdnn::gather<float, int>(ctx,
                                   boxes,
                                   batch_box_index + batch_idx * merge_k,
                                   merge_boxes + batch_idx * merge_k * 4,
                                   {n_boxes, 4},
                                   current_box_num,
                                   0);
    CHECK_EQ(ret, 0);
    ret = xdnn::concat<float>(
        ctx,
        {merge_class + batch_idx * merge_k,
         merge_scores + batch_idx * merge_k,
         merge_boxes + batch_idx * merge_k * 4},
        out + merge_boxes_num_lod[batch_idx] * 6,
        {{current_box_num, 1}, {current_box_num, 1}, {current_box_num, 4}},
        1);
    CHECK_EQ(ret, 0);
  }
  if (merge_boxes_num_lod.back() > 0 && return_index) {
    param.index->Resize({merge_boxes_num_lod.back()});
    int* index_ptr = param.index->mutable_data<int>(TARGET(kXPU));
    for (int batch_idx = 0; batch_idx < n; batch_idx++) {
      int cur_box_num =
          merge_boxes_num_lod[batch_idx + 1] - merge_boxes_num_lod[batch_idx];
      if (cur_box_num <= 0) {
        continue;
      }
      ret = xdnn::copy<int>(ctx,
                            batch_box_index + batch_idx * merge_k,
                            index_ptr + merge_boxes_num_lod[batch_idx],
                            cur_box_num);
      CHECK_EQ(ret, 0);
    }
  }

  // Lod Set
  std::vector<uint64_t> batch_starts;
  if (merge_boxes_num_lod.back() <= 0) {
    if (return_index) {
      param.out->Resize({0, out_dim});
      param.index->Resize({0, 1});
    } else {
      param.out->Resize({1, 1});
      ret = xdnn::constant<float>(
          ctx, param.out->mutable_data<float>(TARGET(kXPU)), 1, -1.0f);
      CHECK_EQ(ret, 0);
      batch_starts = {0, 1};
    }
  } else {
    for (int i = 0; i < merge_boxes_num_lod.size(); i++) {
      batch_starts.push_back(merge_boxes_num_lod[i]);
    }
  }
  if (return_rois_num) {
    auto* nms_rois_num = param.nms_rois_num;
    nms_rois_num->mutable_data<int>();
    int* num_data = nms_rois_num->mutable_data<int>();
    for (int i = 1; i <= n; i++) {
      num_data[i - 1] = batch_starts[i] - batch_starts[i - 1];
    }
    nms_rois_num->Resize({n});
  }
  LoD lod;
  lod.emplace_back(batch_starts);
  if (return_index) {
    param.index->set_lod(lod);
  }
  param.out->set_lod(lod);
}
}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(multiclass_nms,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::MulticlassNmsCompute,
                     def)
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(multiclass_nms2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::MulticlassNmsCompute,
                     def)
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Index",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(multiclass_nms3,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::MulticlassNmsCompute,
                     def)
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("RoisNum",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Index",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("NmsRoisNum",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
