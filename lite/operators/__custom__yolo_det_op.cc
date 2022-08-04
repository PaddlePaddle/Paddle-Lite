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

#include "lite/operators/__custom__yolo_det_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool CustomYoloDetOp::CheckShape() const {
  CHECK(param_.X0);
  CHECK(param_.X1);
  CHECK(param_.X2);
  CHECK(param_.ImgSize);
  CHECK(param_.Output);
  return true;
}

bool CustomYoloDetOp::InferShapeImpl() const {
  // InferShape is useless for multiclass_nms
  // out's dim is not sure before the end of calculation
  return true;
}

bool CustomYoloDetOp::AttachImpl(const cpp::OpDesc& op_desc,
                                 lite::Scope* scope) {
  auto X0 = op_desc.Input("X0").front();
  auto X1 = op_desc.Input("X1").front();
  auto X2 = op_desc.Input("X2").front();
  auto ImgSize = op_desc.Input("ImgSize").front();
  auto Output = op_desc.Output("Output").front();

  param_.X0 = scope->FindVar(X0)->GetMutable<lite::Tensor>();
  param_.X1 = scope->FindVar(X1)->GetMutable<lite::Tensor>();
  param_.X2 = scope->FindVar(X2)->GetMutable<lite::Tensor>();
  param_.ImgSize = scope->FindVar(ImgSize)->GetMutable<lite::Tensor>();
  param_.Output = scope->FindVar(Output)->GetMutable<lite::Tensor>();

  param_.anchors = op_desc.GetAttr<std::vector<int>>("anchors");
  param_.downsample_ratios =
      op_desc.GetAttr<std::vector<int>>("downsample_ratios");
  param_.class_num = op_desc.GetAttr<int>("class_num");
  param_.conf_thresh = op_desc.GetAttr<float>("conf_thresh");
  param_.keep_top_k = op_desc.GetAttr<int>("keep_top_k");
  param_.nms_threshold = op_desc.GetAttr<float>("nms_threshold");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__custom__yolo_det, paddle::lite::operators::CustomYoloDetOp);
