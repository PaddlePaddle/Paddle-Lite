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

#include "lite/kernels/xpu/multiclass_nms_compute.h"
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void MulticlassNmsCompute::PrepareForRun() {
  double n = GetDoubleFromEnv("ResetMulticlassNMSScoreThreshold", 0.0);
  if (n > 0.0) {
    custom_config_score_threshod_ = static_cast<float>(n);
    VLOG(2) << "ResetMulticlassNMSScoreThreshold set to "
            << custom_config_score_threshod_;
  }
  int k = GetIntFromEnv("ResetMulticlassNMSTopK", 0);
  if (k > 0) {
    custom_config_topk_ = k;
    VLOG(2) << "ResetMulticlassNMSTopK set to " << k;
  }
}

void MulticlassNmsCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* boxes = param.bboxes;
  auto* scores = param.scores;
  auto* outs = param.out;
  auto* out_index = param.index;
  auto score_dims = scores->dims();
  auto score_size = score_dims.size();
  bool is_lod = score_size == 2 ? true : false;
  bool return_index = param.index ? true : false;
  auto has_rois_num = param.rois_num != nullptr;
  auto return_rois_num = param.nms_rois_num != nullptr;
  auto rois_num = param.rois_num;
  int background_label = param.background_label;
  int nms_top_k = param.nms_top_k;
  if (custom_config_topk_ > 0) {
    nms_top_k = custom_config_topk_;
  }
  int keep_top_k = param.keep_top_k;
  bool normalized = param.normalized;
  float nms_threshold = static_cast<float>(param.nms_threshold);
  float nms_eta = static_cast<float>(param.nms_eta);
  float score_threshold = static_cast<float>(param.score_threshold);
  if (custom_config_score_threshod_ > 0.0) {
    score_threshold = custom_config_score_threshod_;
  }

  int n = 0;
  int b = 0;
  int class_num = scores->dims()[1];
  int out_dim = boxes->dims()[2] + 2;
  int boxes_count = 0;
  std::vector<int> rois_num_vec;
  rois_num_vec.clear();
  if (is_lod) {
    if (has_rois_num) {
      n = rois_num->numel();
      for (int i = 0; i < n; i++) {
        rois_num_vec.push_back(rois_num->data<int>()[i]);
        boxes_count += rois_num->data<int>()[i];
      }
    } else {
      auto lod = boxes->lod().back();
      boxes_count = lod[lod.size() - 1];
      n = lod.size() - 1;
      for (int i = 0; i < n; i++) {
        rois_num_vec.push_back(lod[i + 1] - lod[i]);
      }
    }
    CHECK_EQ(boxes_count, boxes->dims()[0]);
    CHECK_EQ(boxes_count, score_dims[0]);
  } else {
    n = boxes->dims()[0];
    b = boxes->dims()[1];
    boxes_count = n * b;
  }
  outs_vec_.resize(boxes_count * out_dim);
  out_index_vec_.resize(boxes_count);

  std::vector<size_t> batch_starts;

  int r = 0;
  r = xdnn::multiclass_nms<float, int>(ctx.GetRawContext(),
                                       boxes->data<float>(),
                                       scores->data<float>(),
                                       rois_num_vec,
                                       outs_vec_,
                                       out_index_vec_,
                                       batch_starts,
                                       n,
                                       b,
                                       class_num,
                                       out_dim,
                                       nms_top_k,
                                       score_threshold,
                                       keep_top_k,
                                       nms_threshold,
                                       background_label,
                                       normalized,
                                       nms_eta,
                                       return_index,
                                       is_lod);
  CHECK_EQ(r, 0);

  uint64_t num_kept = batch_starts.back();
  if (num_kept == 0) {
    if (return_index) {
      // out_dim may be zero when there is no object in picture, so add some
      // zeros to it
      // caution: results may differ between cpu and xpu due to this operation
      outs->Resize({1, out_dim});
      float* out_ptr = outs->mutable_data<float>();
      std::vector<float> temp_value(out_dim, 0.0f);
      std::memcpy(out_ptr, temp_value.data(), 1 * out_dim * sizeof(float));

      out_index->Resize({1, 1});
      int* out_index_ptr = out_index->mutable_data<int>();
      std::vector<int> temp_idx(1, 0);
      std::memcpy(out_index_ptr, temp_idx.data(), 1 * sizeof(int));
    } else {
      outs->Resize({1, 1});
      float* od = outs->mutable_data<float>(TARGET(kHost));
      od[0] = -1;
      batch_starts = {0, 1};
    }
  } else {
    outs->Resize({static_cast<int64_t>(num_kept), out_dim});
    float* out_ptr = outs->mutable_data<float>();
    std::memcpy(out_ptr, outs_vec_.data(), num_kept * out_dim * sizeof(float));
    if (return_index) {
      out_index->Resize({static_cast<int64_t>(num_kept), 1});
      int* out_index_ptr = out_index->mutable_data<int>();
      std::memcpy(
          out_index_ptr, out_index_vec_.data(), num_kept * sizeof(float));
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
  if (num_kept == 0) {
    batch_starts[batch_starts.size() - 1] = 1;
  }
  lod.emplace_back(batch_starts);
  if (return_index) {
    out_index->set_lod(lod);
  }
  outs->set_lod(lod);
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
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(multiclass_nms2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::MulticlassNmsCompute,
                     def)
    .BindInput("BBoxes", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Scores", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Index",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
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
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Index",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("NmsRoisNum",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
