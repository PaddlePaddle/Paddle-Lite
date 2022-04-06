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

#include "lite/operators/yolo_box_parser_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool YoloBoxParserOp::CheckShape() const {
  auto* image_shape = param_.image_shape;
  auto x0 = param_.x0;
  auto x1 = param_.x1;
  auto x2 = param_.x2;
  auto* image_scale = param_.image_scale;
  auto* boxes_scores = param_.boxes_scores;
  CHECK_OR_FALSE(image_shape);
  CHECK_OR_FALSE(boxes_scores);
  CHECK_OR_FALSE(image_scale);

  auto image_shape_dims = image_shape->dims();
  auto class_num = param_.class_num;
  CHECK_OR_FALSE(image_shape_dims[1] == 2);
  CHECK_OR_FALSE(class_num > 0);

  auto check = [class_num, image_shape_dims](lite::Tensor* x,
                                             std::vector<int> anchors) {
    int anchor_num = anchors.size() / 2;
    CHECK_OR_FALSE(anchor_num > 0 && anchors.size() % 2 == 0);
    auto x_dims = x->dims();
    CHECK_OR_FALSE(image_shape_dims[0] == x_dims[0]);
    CHECK_OR_FALSE(x_dims.size() == 4);
    CHECK_OR_FALSE(x_dims[1] == anchor_num * (5 + class_num));
    return true;
  };
  check(x0, param_.anchors0);
  check(x1, param_.anchors1);
  check(x2, param_.anchors2);
  return true;
}

bool YoloBoxParserOp::InferShapeImpl() const {
  // not sure about the output shape
  return true;
}

bool YoloBoxParserOp::AttachImpl(const cpp::OpDesc& op_desc,
                                 lite::Scope* scope) {
  auto x0 = op_desc.Input("x0").front();
  auto x1 = op_desc.Input("x1").front();
  auto x2 = op_desc.Input("x2").front();
  param_.x0 = scope->FindVar(x0)->GetMutable<lite::Tensor>();
  param_.x1 = scope->FindVar(x1)->GetMutable<lite::Tensor>();
  param_.x2 = scope->FindVar(x2)->GetMutable<lite::Tensor>();

  auto image_shape = op_desc.Input("image_shape").front();
  auto image_scale = op_desc.Input("image_scale").front();
  auto boxes_scores = op_desc.Output("boxes_scores").front();
  param_.image_shape = scope->FindVar(image_shape)->GetMutable<lite::Tensor>();
  param_.image_scale = scope->FindVar(image_scale)->GetMutable<lite::Tensor>();
  param_.boxes_scores =
      scope->FindVar(boxes_scores)->GetMutable<lite::Tensor>();

  param_.anchors0 = op_desc.GetAttr<std::vector<int>>("anchors0");
  param_.anchors1 = op_desc.GetAttr<std::vector<int>>("anchors1");
  param_.anchors2 = op_desc.GetAttr<std::vector<int>>("anchors2");

  param_.class_num = op_desc.GetAttr<int>("class_num");
  param_.conf_thresh = op_desc.GetAttr<float>("conf_thresh");
  param_.downsample_ratio0 = op_desc.GetAttr<int>("downsample_ratio0");
  param_.downsample_ratio1 = op_desc.GetAttr<int>("downsample_ratio1");
  param_.downsample_ratio2 = op_desc.GetAttr<int>("downsample_ratio2");
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

REGISTER_LITE_OP(yolo_box_parser, paddle::lite::operators::YoloBoxParserOp);
