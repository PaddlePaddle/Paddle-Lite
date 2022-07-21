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

#include "driver/qualcomm_qnn/converter/converter.h"
#include "operation/non_max_suppression.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace qualcomm_qnn {

int ConvertNonMaxSuppression(Converter* converter, core::Operation* operation) {
  NON_MAX_SUPPRESSION_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_LOG(INFO) << "AAAAAAAAAAAAAAAA";
  // Convert to qnn tensors and node
  auto bboxes_tensor = converter->GetMappedTensor(input_operands[0]);
  auto scores_tensor = converter->GetMappedTensor(input_operands[1]);
  auto output_boxes_tensor = converter->GetMappedTensor(output_operands[0]);
  auto output_nms_rois_num_tensor =
      converter->GetMappedTensor(output_operands[1]);
  auto output_index_tensor = converter->GetMappedTensor(output_operands[2]);
  NNADAPTER_LOG(INFO) << "AAAAAAAAAAAAAAAA";
  // attr
  auto background_label_param =
      converter->GetParam("background_label", background_label);
  auto score_threshold_param =
      converter->GetParam("score_threshold", score_threshold);
  auto nms_top_k_param = converter->GetParam("nms_top_k", nms_top_k);
  auto nms_threshold_param =
      converter->GetParam("nms_threshold", nms_threshold);
  auto nms_eta_param = converter->GetParam("nms_eta", nms_eta);
  auto keep_top_k_param = converter->GetParam("keep_top_k", keep_top_k);
  auto normalized_param = converter->GetParam("normalized", normalized);
  NNADAPTER_LOG(INFO) << "AAAAAAAAAAAAAAAA";
  converter->AddNode(
      "MulticlassNms",
      {bboxes_tensor, scores_tensor},
      {output_boxes_tensor, output_nms_rois_num_tensor, output_index_tensor},
      {background_label_param,
       score_threshold_param,
       nms_top_k_param,
       nms_threshold_param,
       nms_eta_param,
       keep_top_k_param,
       normalized_param});
  return NNADAPTER_NO_ERROR;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
