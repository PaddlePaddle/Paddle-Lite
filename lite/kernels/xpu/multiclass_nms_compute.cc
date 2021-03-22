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
#include <map>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void MulticlassNmsCompute::Run() {
  auto& param = Param<operators::MulticlassNmsParam>();
  auto* boxes = param.bboxes;
  auto* scores = param.scores;
  auto* outs = param.out;
  bool return_index = param.index ? true : false;
  auto* index = param.index;
  auto score_dims = scores->dims();
  auto score_size = score_dims.size();
  int box_dim = boxes->dims()[2];
  CHECK_EQ(score_size, 3)
      << " xpu MulticlassNms only support score_dims 3 which is " << score_size;
  CHECK_EQ(box_dim, 4) << " xpu MulticlassNms only support box_dim 4 which is "
                       << box_dim;

  int background_label = param.background_label;
  int nms_top_k = param.nms_top_k;
  int keep_top_k = param.keep_top_k;
  bool normalized = param.normalized;
  float nms_threshold = param.nms_threshold;
  float nms_eta = param.nms_eta;
  float score_threshold = param.score_threshold;
  int n = score_dims[0];
  int class_num = score_dims[1];
  CHECK(class_num <= 80)
      << "xpu MulticlassNms only support class_num <= 80 which is "
      << class_num;
  int box_num = score_dims[2];
  int out_dim = box_dim + 2;
  if (nms_top_k > 100) {
    VLOG(5) << "xpu MultiClassNMS may get accuracy loss while nms_top_k is "
               "larger than 100";
  }
  auto& ctx = this->ctx_->As<XPUContext>();
  std::vector<size_t> batch_starts;
  outs->Resize({n, box_num, out_dim});
  if (return_index) {
    index->Resize({n, box_num});
  }
  int r = xdnn::yolo_nms(ctx.GetRawContext(),
                         n,
                         box_num,
                         class_num,
                         std::min(512, nms_top_k),
                         background_label,
                         std::min(keep_top_k, 100),
                         normalized,
                         nms_threshold,
                         nms_eta,
                         score_threshold,
                         boxes->data<float>(),
                         scores->data<float>(),
                         out_dim,
                         outs->mutable_data<float>(TARGET(kHost)),
                         &batch_starts);
  CHECK_EQ(r, 0);

  uint64_t num_kept = batch_starts.back();
  if (num_kept == 0) {
    if (return_index) {
      outs->Resize({0, out_dim});
      index->Resize({0, 1});
    } else {
      outs->Resize({1, 1});
      float* od = outs->mutable_data<float>(TARGET(kHost));
      od[0] = -1;
      batch_starts = {0, 1};
    }
  } else {
    outs->Resize({static_cast<int64_t>(num_kept), out_dim});
    if (return_index) {
      index->Resize({static_cast<int64_t>(num_kept), 1});
    }
  }

  LoD lod;
  lod.emplace_back(batch_starts);
  if (return_index) {
    index->set_lod(lod);
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
