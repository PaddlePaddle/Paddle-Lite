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

#include "operation/yolo_box_parser.h"
#include <iostream>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"
namespace nnadapter {
namespace operation {

bool ValidateYoloBoxParser(const core::Operation* operation) { return false; }

int PrepareYoloBoxParser(core::Operation* operation) {
  YOLO_BOX_PARSER_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape of boxes and scores
  auto& boxes_scores_type = boxes_scores_operand->type;
printf("四年级的v啊力所能及的sieved%d\n", (anchors_operand0->type.lifetime));
printf("四年级的v啊力所能及的sieved%d\n", (anchors_operand1->type.lifetime));
printf("四年级的v啊力所能及的sieved%d\n", (anchors_operand2->type.lifetime));
  boxes_scores_type.dimensions.count = 3;

  class_num = class_num;
  conf_thresh = conf_thresh;
  downsample_ratio0 = downsample_ratio0;
  downsample_ratio1 = downsample_ratio1;
  downsample_ratio2 = downsample_ratio2;
  clip_bbox = clip_bbox;
  scale_x_y = scale_x_y;

  boxes_scores_type.precision = x0_operand->type.precision;
  boxes_scores_type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;

  NNADAPTER_VLOG(5) << "boxes_scores: " << OperandToString(boxes_scores_operand);

  return NNADAPTER_NO_ERROR;
}

int ExecuteYoloBoxParser(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
