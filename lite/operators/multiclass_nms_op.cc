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
  CHECK_OR_FALSE(param_.bbox_data);
  CHECK_OR_FALSE(param_.conf_data);
  CHECK_OR_FALSE(param_.out);

  return true;
}

bool MulticlassNmsOpLite::InferShape() const {
  // param_.out->Resize(param_.loc_data->dims());
  return true;
}

bool MulticlassNmsOpLite::AttachImpl(const cpp::OpDesc& opdesc,
                                     lite::Scope* scope) {
  auto Bbox_name = opdesc.Input("BBoxes").front();
  auto Conf_name = opdesc.Input("Scores").front();
  auto Out_name = opdesc.Output("Out").front();
  param_.bbox_data = GetVar<lite::Tensor>(scope, Bbox_name);
  param_.conf_data = GetVar<lite::Tensor>(scope, Conf_name);
  param_.out = GetMutableVar<lite::Tensor>(scope, Out_name);
  param_.background_label = opdesc.GetAttr<int>("background_label");
  param_.keep_top_k = opdesc.GetAttr<int>("keep_top_k");
  param_.nms_top_k = opdesc.GetAttr<int>("nms_top_k");
  param_.score_threshold = opdesc.GetAttr<float>("score_threshold");
  param_.nms_threshold = opdesc.GetAttr<float>("nms_threshold");
  param_.nms_eta = opdesc.GetAttr<float>("nms_eta");
  if (opdesc.HasAttr("share_location")) {
    param_.share_location = opdesc.GetAttr<bool>("share_location");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(multiclass_nms, paddle::lite::operators::MulticlassNmsOpLite);
