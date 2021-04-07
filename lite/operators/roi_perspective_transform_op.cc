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

#include "lite/operators/roi_perspective_transform_op.h"

namespace paddle {
namespace lite {
namespace operators {

bool RoiPerspectiveTransformOp::CheckShape() const {
  CHECK(param_.x);
  CHECK(param_.rois);
  CHECK(param_.out);
  CHECK(param_.mask);
  CHECK(param_.transfor_matrix);
  CHECK(param_.out2in_idx);
  CHECK(param_.out2in_weight);

  auto x_dims = param_.x->dims();
  CHECK_EQ(x_dims.size(), 4UL)
      << "The format of input tensor must be NCHW. But received input dims is: "
      << x_dims;
  auto rois_dims = param_.rois->dims();
  CHECK_EQ(rois_dims.size(), 2UL)
      << "ROIs should be a 2-D LoDTensor of shape (num_rois, 8) given as [[x0, "
         "y0, x1, y1, x2, y2, x3, y3], ...]. But received rois dims: "
      << rois_dims;
  CHECK_EQ(rois_dims[1], 8L)
      << "ROIs should be a 2-D LoDTensor of shape (num_rois, 8) given as [[x0, "
         "y0, x1, y1, x2, y2, x3, y3], ...]. But received rois dims: "
      << rois_dims;
  return true;
}

bool RoiPerspectiveTransformOp::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  auto rois_dims = param_.rois->dims();

  DDim out_dims({rois_dims[0],
                 x_dims[1],
                 static_cast<int64_t>(param_.transformed_height),
                 static_cast<int64_t>(param_.transformed_width)});
  param_.out->Resize(out_dims);
  param_.out->set_lod(param_.rois->lod());
  DDim mask_dims(out_dims);
  mask_dims[1] = 1;
  param_.mask->Resize(mask_dims);
  param_.transfor_matrix->Resize({rois_dims[0], 9L});
  param_.out2in_idx->Resize(out_dims);
  param_.out2in_weight->Resize(out_dims);
  return true;
}

bool RoiPerspectiveTransformOp::AttachImpl(const cpp::OpDesc &op_desc,
                                           lite::Scope *scope) {
  param_.x = scope->FindTensor(op_desc.Input("X").front());
  param_.rois = scope->FindTensor(op_desc.Input("ROIs").front());
  param_.out = scope->FindMutableTensor(op_desc.Output("Out").front());
  param_.mask = scope->FindMutableTensor(op_desc.Output("Mask").front());
  param_.transfor_matrix =
      scope->FindMutableTensor(op_desc.Output("TransformMatrix").front());
  param_.out2in_idx =
      scope->FindMutableTensor(op_desc.Output("Out2InIdx").front());
  param_.out2in_weight =
      scope->FindMutableTensor(op_desc.Output("Out2InWeights").front());

  param_.spatial_scale = op_desc.GetAttr<float>("spatial_scale");
  param_.transformed_height = op_desc.GetAttr<int>("transformed_height");
  param_.transformed_width = op_desc.GetAttr<int>("transformed_width");
  CHECK_GT(param_.spatial_scale, 0.f)
      << "The spatial_scale must be greater than 0. But received: "
      << param_.spatial_scale;
  CHECK_GT(param_.transformed_height, 0)
      << "The transformed output height must be greater than 0. But received: "
      << param_.transformed_height;
  CHECK_GT(param_.transformed_width, 0)
      << "The transformed output width must be greater than 0. But received: "
      << param_.transformed_width;
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(roi_perspective_transform,
                 paddle::lite::operators::RoiPerspectiveTransformOp);
