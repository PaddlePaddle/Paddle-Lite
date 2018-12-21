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

#ifdef MULTICLASSNMS_OP

#include "operators/kernel/multiclass_nms_kernel.h"
#include "operators/kernel/fpga/V1/multiclass_nms_arm_func.h"

namespace paddle_mobile {
namespace operators {

// template <typename P>
// void MultiClassNMSCompute(const MultiClassNMSParam<FPGA>& param) {
//   const auto* input_bboxes = param.InputBBoxes();
//   const auto& input_bboxes_dims = input_bboxes->dims();

//   const auto* input_scores = param.InputScores();
//   const auto& input_scores_dims = input_scores->dims();

//   auto* outs = param.Out();
//   auto background_label = param.BackGroundLabel();
//   auto nms_top_k = param.NMSTopK();
//   auto keep_top_k = param.KeepTopK();
//   auto nms_threshold = param.NMSThreshold();
//   auto nms_eta = param.NMSEta();
//   auto score_threshold = param.ScoreThreshold();

//   int64_t batch_size = input_scores_dims[0];
//   int64_t class_num = input_scores_dims[1];
//   int64_t predict_dim = input_scores_dims[2];
//   int64_t box_dim = input_bboxes_dims[2];

//   std::vector<std::map<int, std::vector<int>>> all_indices;
//   std::vector<size_t> batch_starts = {0};
//   for (int64_t i = 0; i < batch_size; ++i) {
//     framework::Tensor ins_score = input_scores->Slice(i, i + 1);
//     ins_score.Resize({class_num, predict_dim});

//     framework::Tensor ins_boxes = input_bboxes->Slice(i, i + 1);
//     ins_boxes.Resize({predict_dim, box_dim});

//     std::map<int, std::vector<int>> indices;
//     int num_nmsed_out = 0;
//     MultiClassNMS<float>(ins_score, ins_boxes, &indices, &num_nmsed_out,
//                          background_label, nms_top_k, keep_top_k,
//                          nms_threshold, nms_eta, score_threshold);
//     all_indices.push_back(indices);
//     batch_starts.push_back(batch_starts.back() + num_nmsed_out);
//   }

//   int num_kept = batch_starts.back();
//   if (num_kept == 0) {
//     float* od = outs->mutable_data<float>({1});
//     od[0] = -1;
//   } else {
//     int64_t out_dim = box_dim + 2;
//     outs->mutable_data<float>({num_kept, out_dim});
//     for (int64_t i = 0; i < batch_size; ++i) {
//       framework::Tensor ins_score = input_scores->Slice(i, i + 1);
//       ins_score.Resize({class_num, predict_dim});

//       framework::Tensor ins_boxes = input_bboxes->Slice(i, i + 1);
//       ins_boxes.Resize({predict_dim, box_dim});

//       int64_t s = batch_starts[i];
//       int64_t e = batch_starts[i + 1];
//       if (e > s) {
//         framework::Tensor out = outs->Slice(s, e);
//         MultiClassOutput<float>(ins_score, ins_boxes, all_indices[i], &out);
//       }
//     }
//   }
// }

template <>
bool MultiClassNMSKernel<FPGA, float>::Init(MultiClassNMSParam<FPGA> *param) {
  return true;
}

template <>
void MultiClassNMSKernel<FPGA, float>::Compute(
    const MultiClassNMSParam<FPGA> &param) {
  MultiClassNMSCompute<float>(param);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
