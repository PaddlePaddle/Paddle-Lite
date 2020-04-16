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

#include "lite/operators/reshape_op.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool ReshapeOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  return true;
}

bool ReshapeOp::InferShapeImpl() const {
  const auto &shape_tensor_vct = param_.shape_tensor_vct;
  auto *shape_tensor = param_.shape_tensor;
  const auto &shape_vct = param_.shape_vct;

  std::vector<int> final_shape;
  if (shape_tensor_vct.size() > 0) {
    final_shape.resize(shape_tensor_vct.size());
    for (size_t i = 0; i < shape_tensor_vct.size(); i++) {
      final_shape[i] = shape_tensor_vct[i]->data<int>()[0];
    }
  } else if (shape_tensor != nullptr && shape_tensor->data<int>() != nullptr) {
    auto *shape_tensor_data = shape_tensor->data<int>();
    final_shape = std::vector<int>(shape_tensor_data,
                                   shape_tensor_data + shape_tensor->numel());
  } else if (!shape_vct.empty()) {
    final_shape = shape_vct;
  } else {
    LOG(FATAL) << "input shape error";
  }

  const auto &x_dims = param_.x->dims();
  auto output_dims = ValidateShape(final_shape, x_dims);
  param_.output->Resize(output_dims);
  auto out_lod = param_.output->mutable_lod();
  *out_lod = param_.x->lod();
  return true;
}

bool ReshapeOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.x =
      scope->FindVar(opdesc.Input("X").front())->GetMutable<lite::Tensor>();
  param_.output =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();

  // prority: input(ShapeTensor) > input(Shape) > attr(shape)
  if (opdesc.HasInput("ShapeTensor") && !opdesc.Input("ShapeTensor").empty()) {
    auto args = opdesc.Input("ShapeTensor");
    for (auto arg : args) {
      auto *var = scope->FindVar(arg);
      if (var != nullptr) {
        param_.shape_tensor_vct.push_back(var->GetMutable<lite::Tensor>());
      }
    }
    CHECK_GT(param_.shape_tensor_vct.size(), 0u)
        << "ShapeError: When `shape` in ReshapeOp is a list or tuple "
           "which contains Tensor, the shape's size can't be zero. "
           "But received shape's size is "
        << param_.shape_tensor_vct.size();
  }
  if (opdesc.HasInput("Shape") && !opdesc.Input("Shape").empty()) {
    auto var = scope->FindVar(opdesc.Input("Shape").front());
    if (var != nullptr) {
      param_.shape_tensor = var->GetMutable<lite::Tensor>();
    }
  }
  if (opdesc.HasAttr("shape")) {
    param_.shape_vct = opdesc.GetAttr<std::vector<int>>("shape");
  }
  if (opdesc.HasAttr("inplace")) {
    param_.inplace = opdesc.GetAttr<bool>("inplace");
  }
  return true;
}

bool Reshape2Op::CheckShape() const {
  ReshapeOp::CheckShape();
  CHECK_OR_FALSE(param_.xshape);
  return true;
}

bool Reshape2Op::InferShapeImpl() const {
  ReshapeOp::InferShapeImpl();
  const auto &x_dims = param_.x->dims();
  std::vector<DDim::value_type> xshape_dims(x_dims.size() + 1);
  xshape_dims[0] = 0;
  for (size_t i = 0; i < x_dims.size(); i++) {
    xshape_dims[i + 1] = x_dims[i];
  }
  param_.xshape->Resize(xshape_dims);
  auto xshape_lod = param_.xshape->mutable_lod();
  *xshape_lod = param_.x->lod();
  return true;
}

bool Reshape2Op::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  ReshapeOp::AttachImpl(opdesc, scope);
  auto xshape_var = scope->FindVar(opdesc.Output("XShape").front());
  param_.xshape = xshape_var->GetMutable<lite::Tensor>();
  return true;
}

static bool CheckPositive(const DDim &dims) {
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] <= 0) {
      return false;
    }
  }
  return true;
}

std::vector<DDim::value_type> ValidateShape(const std::vector<int> &shape,
                                            const DDim &input_dims) {
  const DDim::value_type input_size = input_dims.production();

  // only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int unk_dim_val = -1;
  const int copy_dim_val = 0;

  std::vector<DDim::value_type> output_dims(shape.size());
  DDim::value_type capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      CHECK_EQ(unk_dim_idx, -1)
          << "Only one input dimension of Attr(shape) can be unknown.";
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      CHECK_LT(i, input_dims.size())
          << "The index of dimension to copy from input shape must be less "
             "than the size of input shape.";
    } else {
      CHECK_GT(shape[i], 0) << "Each input dimension of Attr(shape) must not "
                               "be negtive except one unknown dimension.";
    }

    DDim::value_type output_dim_i =
        shape[i] ? static_cast<DDim::value_type>(shape[i]) : input_dims[i];
    output_dims[i] = output_dim_i;
    capacity *= output_dim_i;
  }

  if (unk_dim_idx != -1) {
    if (CheckPositive(input_dims)) {
      // input_size < 0 and is un-determinate in compile time, skip the check,
      // for example, input_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, input_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_dims[unk_dim_idx] = -input_size / capacity;
      CHECK_EQ(output_dims[unk_dim_idx] * capacity, -input_size)
          << "Invalid shape is given.";
    } else {
      output_dims[unk_dim_idx] = -1;
    }
  } else {
    CHECK_EQ(capacity, input_size) << "Invalid shape is given.";
  }
  return output_dims;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(reshape, paddle::lite::operators::ReshapeOp);
REGISTER_LITE_OP(reshape2, paddle::lite::operators::Reshape2Op);
