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

#include "lite/operators/roi_align_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool RoiAlignOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.ROIs);
  CHECK_OR_FALSE(param_.Out);

  auto x_dims = param_.X->dims();
  auto rois_dims = param_.ROIs->dims();

  CHECK_OR_FALSE(x_dims.size() == 4);
  CHECK_OR_FALSE(rois_dims.size() == 2);
  CHECK_OR_FALSE(rois_dims[1] == 4);
  CHECK_OR_FALSE(param_.pooled_height > 0);
  CHECK_OR_FALSE(param_.pooled_width > 0);
  CHECK_OR_FALSE(param_.spatial_scale > 0.0f);

  return true;
}

bool RoiAlignOpLite::InferShapeImpl() const {
  auto x_dims = param_.X->dims();
  auto rois_dims = param_.ROIs->dims();

  param_.Out->Resize(
      {rois_dims[0], x_dims[1], param_.pooled_height, param_.pooled_width});
  return true;
}

bool RoiAlignOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                lite::Scope *scope) {
  param_.X =
      scope->FindVar(op_desc.Input("X").front())->GetMutable<lite::Tensor>();
  param_.ROIs =
      scope->FindVar(op_desc.Input("ROIs").front())->GetMutable<lite::Tensor>();

  if (op_desc.HasInput("RoisLod") && !op_desc.Input("RoisLod").empty()) {
    param_.RoisLod = scope->FindVar(op_desc.Input("RoisLod").front())
                         ->GetMutable<lite::Tensor>();
  }

  if (op_desc.HasInput("RoisNum") && !op_desc.Input("RoisNum").empty()) {
    auto rois_num_var_vec = op_desc.Input("RoisNum");
    if (!rois_num_var_vec.empty()) {
      param_.RoisNum =
          scope->FindVar(rois_num_var_vec.front())->GetMutable<lite::Tensor>();
    }
  }

  param_.spatial_scale = op_desc.GetAttr<float>("spatial_scale");
  param_.pooled_height = op_desc.GetAttr<int>("pooled_height");
  param_.pooled_width = op_desc.GetAttr<int>("pooled_width");
  param_.sampling_ratio = op_desc.GetAttr<int>("sampling_ratio");
  if (op_desc.HasAttr("aligned"))
    param_.align = op_desc.GetAttr<bool>("aligned");

  param_.Out =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<lite::Tensor>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(roi_align, paddle::lite::operators::RoiAlignOpLite);
