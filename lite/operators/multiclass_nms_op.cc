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

#include "lite/operators/multiclass_nms_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool MulticlassNmsOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.bboxes);
  CHECK_OR_FALSE(param_.scores);
  CHECK_OR_FALSE(param_.out);

  auto box_dims = param_.bboxes->dims();
  auto score_dims = param_.scores->dims();
  auto score_size = score_dims.size();

  CHECK_OR_FALSE(score_size == 2 || score_size == 3);
  CHECK_OR_FALSE(box_dims.size() == 3);
  if (score_size == 3) {
    CHECK_OR_FALSE(box_dims[2] == 4 || box_dims[2] == 8 || box_dims[2] == 16 ||
                   box_dims[2] == 24 || box_dims[2] == 32);
    CHECK_OR_FALSE(box_dims[1] == score_dims[2]);
  } else {
    CHECK_OR_FALSE(box_dims[2] == 4);
    CHECK_OR_FALSE(box_dims[1] == score_dims[1]);
  }
  return true;
}

bool MulticlassNmsOpLite::InferShape() const {
  auto box_dims = param_.bboxes->dims();
  auto score_dims = param_.scores->dims();
  auto score_size = score_dims.size();
  if (score_size == 3) {
    param_.out->Resize({box_dims[1], box_dims[2], 3});
  } else {
    param_.out->Resize({-1, box_dims[2] + 2});
  }
  return true;
}

bool MulticlassNmsOpLite::AttachImpl(const cpp::OpDesc& opdesc,
                                     lite::Scope* scope) {
  auto bboxes_name = opdesc.Input("BBoxes").front();
  auto scores_name = opdesc.Input("Scores").front();
  auto out_name = opdesc.Output("Out").front();
  param_.bboxes = GetVar<lite::Tensor>(scope, bboxes_name);
  param_.scores = GetVar<lite::Tensor>(scope, scores_name);
  param_.out = GetMutableVar<lite::Tensor>(scope, out_name);
  param_.background_label = opdesc.GetAttr<int>("background_label");
  param_.keep_top_k = opdesc.GetAttr<int>("keep_top_k");
  param_.nms_top_k = opdesc.GetAttr<int>("nms_top_k");
  param_.score_threshold = opdesc.GetAttr<float>("score_threshold");
  param_.nms_threshold = opdesc.GetAttr<float>("nms_threshold");
  param_.nms_eta = opdesc.GetAttr<float>("nms_eta");
  if (opdesc.HasAttr("normalized")) {
    param_.normalized = opdesc.GetAttr<bool>("normalized");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(multiclass_nms, paddle::lite::operators::MulticlassNmsOpLite);
