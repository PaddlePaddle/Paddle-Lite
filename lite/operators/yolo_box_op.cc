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

#include "lite/operators/yolo_box_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool YoloBoxOp::CheckShape() const {
  auto* X = param_.X;
  auto* ImgSize = param_.ImgSize;
  auto* Boxes = param_.Boxes;
  auto* Scores = param_.Scores;
  CHECK_OR_FALSE(X);
  CHECK_OR_FALSE(ImgSize);
  CHECK_OR_FALSE(Boxes);
  CHECK_OR_FALSE(Scores);

  auto dim_x = X->dims();
  auto dim_imgsize = ImgSize->dims();
  std::vector<int> anchors = param_.anchors;
  int anchor_num = anchors.size() / 2;
  auto class_num = param_.class_num;
  CHECK_OR_FALSE(dim_x.size() == 4);
  CHECK_OR_FALSE(dim_x[1] == anchor_num * (5 + class_num));
  CHECK_OR_FALSE(dim_imgsize[0] == dim_x[0]);
  CHECK_OR_FALSE(dim_imgsize[1] == 2);
  CHECK_OR_FALSE(anchors.size() > 0 && anchors.size() % 2 == 0);
  CHECK_OR_FALSE(class_num > 0);
  return true;
}

bool YoloBoxOp::InferShapeImpl() const {
  auto* X = param_.X;
  auto anchors = param_.anchors;
  int anchor_num = anchors.size() / 2;
  auto class_num = param_.class_num;
  DDim x_dim = X->dims();

  int box_num = x_dim[2] * x_dim[3] * anchor_num;
  param_.Boxes->Resize({x_dim[0], box_num, 4});
  param_.Scores->Resize({x_dim[0], box_num, class_num});
  return true;
}

bool YoloBoxOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  auto X = op_desc.Input("X").front();
  auto ImgSize = op_desc.Input("ImgSize").front();
  auto Boxes = op_desc.Output("Boxes").front();
  auto Scores = op_desc.Output("Scores").front();
  param_.X = scope->FindVar(X)->GetMutable<lite::Tensor>();
  param_.ImgSize = scope->FindVar(ImgSize)->GetMutable<lite::Tensor>();
  param_.Boxes = scope->FindVar(Boxes)->GetMutable<lite::Tensor>();
  param_.Scores = scope->FindVar(Scores)->GetMutable<lite::Tensor>();
  param_.anchors = op_desc.GetAttr<std::vector<int>>("anchors");
  param_.class_num = op_desc.GetAttr<int>("class_num");
  param_.conf_thresh = op_desc.GetAttr<float>("conf_thresh");
  param_.downsample_ratio = op_desc.GetAttr<int>("downsample_ratio");
  if (op_desc.HasAttr("clip_bbox")) {
    param_.clip_bbox = op_desc.GetAttr<bool>("clip_bbox");
  }
  if (op_desc.HasAttr("scale_x_y")) {
    param_.scale_x_y = op_desc.GetAttr<float>("scale_x_y");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(yolo_box, paddle::lite::operators::YoloBoxOp);
