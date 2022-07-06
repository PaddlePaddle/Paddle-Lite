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
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver/intel_openvino/converter/converter.h"
#include "operation/non_max_suppression.h"
#include "utility/debug.h"
#include "utility/logging.h"
namespace nnadapter {
namespace intel_openvino {

int ConvertNonMaxSuppression(Converter* converter, core::Operation* operation) {
  NON_MAX_SUPPRESSION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto box_tensor = converter->GetMappedTensor(bboxes_operand);
  if (!box_tensor) {
    box_tensor = converter->ConvertOperand(bboxes_operand);
  }
  auto score_tensor = converter->GetMappedTensor(scores_operand);
  if (!score_tensor) {
    score_tensor = converter->ConvertOperand(scores_operand);
  }

  default_opset::MulticlassNms::Attributes attrs;
  attrs.nms_top_k = nms_top_k;
  attrs.iou_threshold = nms_threshold;
  attrs.score_threshold = score_threshold;
  attrs.sort_result_type =
      default_opset::MulticlassNms::SortResultType::CLASSID;
  attrs.keep_top_k = keep_top_k;
  attrs.background_class = background_label;
  attrs.nms_eta = nms_eta;
  attrs.normalized = normalized;
  attrs.output_type = GetElementType<int32_t>();
  attrs.sort_result_across_batch = false;

  std::shared_ptr<Operator> nms_op;
#if NNADAPTER_INTEL_OPENVINO_VERSION_GREATER_EQUAL(2022, 2, 0)
  if (rois_num_operand) {
    auto rois_num_tensor = converter->GetMappedTensor(rois_num_operand);
    if (!rois_num_tensor) {
      rois_num_tensor = converter->ConvertOperand(rois_num_operand);
    }
    // Transpose scores and boxes first.
    auto input_order_scores = converter->AddConstantTensor<int64_t>({1, 0});
    auto transposed_scores = std::make_shared<default_opset::Transpose>(
        *score_tensor, *input_order_scores);
    auto input_order_boxes = converter->AddConstantTensor<int64_t>({1, 0, 2});
    auto transposed_boxes = std::make_shared<default_opset::Transpose>(
        *box_tensor, *input_order_boxes);
    nms_op = std::make_shared<ov::opset9::MulticlassNms>(
        transposed_boxes->output(0),
        transposed_scores->output(0),
        *rois_num_tensor,
        attrs);
  } else {
    nms_op = std::make_shared<ov::opset9::MulticlassNms>(
        *box_tensor, *score_tensor, attrs);
  }
#else
  // MulticlassNms in opset8 doesn't dot support rois_num input.
  NNADAPTER_CHECK(!rois_num_operand);
  nms_op = std::make_shared<default_opset::MulticlassNms>(
      *box_tensor, *score_tensor, attrs);
#endif
  MAP_OUTPUT(output_box_operand, nms_op, 0);
  if (return_index) {
    auto squeeze_axis = converter->AddConstantTensor(std::vector<int64_t>({1}));
    auto squeeze_index = std::make_shared<default_opset::Squeeze>(
        nms_op->output(1), *squeeze_axis);
    MAP_OUTPUT(output_index_operand, squeeze_index, 0);
  }
  MAP_OUTPUT(output_nms_rois_num_operand, nms_op, 2);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
