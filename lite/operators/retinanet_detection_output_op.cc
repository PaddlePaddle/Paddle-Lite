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

#include "lite/operators/retinanet_detection_output_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool RetinanetDetectionOutputOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.bboxes.size() > 0);
  CHECK_OR_FALSE(param_.scores.size() > 0);
  CHECK_OR_FALSE(param_.anchors.size() > 0);
  CHECK_OR_FALSE(param_.bboxes.size() == param_.scores.size());
  CHECK_OR_FALSE(param_.bboxes.size() == param_.anchors.size());
  CHECK_OR_FALSE(param_.im_info);
  CHECK_OR_FALSE(param_.out);

  DDim bbox_dims = param_.bboxes.front()->dims();
  DDim score_dims = param_.scores.front()->dims();
  DDim anchor_dims = param_.anchors.front()->dims();
  DDim im_info_dims = param_.im_info->dims();

  CHECK_OR_FALSE(bbox_dims.size() == 3);
  CHECK_OR_FALSE(score_dims.size() == 3);
  CHECK_OR_FALSE(anchor_dims.size() == 2);
  CHECK_OR_FALSE(bbox_dims[2] == 4);
  CHECK_OR_FALSE(bbox_dims[1] == score_dims[1]);
  CHECK_OR_FALSE(anchor_dims[0] == bbox_dims[1]);
  CHECK_OR_FALSE(im_info_dims.size() == 2);

  return true;
}

bool RetinanetDetectionOutputOpLite::InferShapeImpl() const {
  DDim bbox_dims = param_.bboxes.front()->dims();
  param_.out->Resize({bbox_dims[1], bbox_dims[2] + 2});
  return true;
}

bool RetinanetDetectionOutputOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                                lite::Scope *scope) {
  param_.bboxes.clear();
  for (auto arg_name : op_desc.Input("BBoxes")) {
    param_.bboxes.push_back(
        scope->FindVar(arg_name)->GetMutable<lite::Tensor>());
  }
  param_.scores.clear();
  for (auto arg_name : op_desc.Input("Scores")) {
    param_.scores.push_back(
        scope->FindVar(arg_name)->GetMutable<lite::Tensor>());
  }
  param_.anchors.clear();
  for (auto arg_name : op_desc.Input("Anchors")) {
    param_.anchors.push_back(
        scope->FindVar(arg_name)->GetMutable<lite::Tensor>());
  }
  AttachInput(op_desc, scope, "ImInfo", false, &param_.im_info);
  AttachOutput(op_desc, scope, "Out", false, &param_.out);

  param_.score_threshold = op_desc.GetAttr<float>("score_threshold");
  param_.nms_top_k = op_desc.GetAttr<int>("nms_top_k");
  param_.nms_threshold = op_desc.GetAttr<float>("nms_threshold");
  param_.nms_eta = op_desc.GetAttr<float>("nms_eta");
  param_.keep_top_k = op_desc.GetAttr<int>("keep_top_k");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(retinanet_detection_output,
                 paddle::lite::operators::RetinanetDetectionOutputOpLite);
