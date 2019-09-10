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
  CHECK_OR_FALSE(!param_.shape.empty());
  return true;
}

bool ReshapeOp::InferShape() const {
  auto x_dims = param_.x->dims();
  auto output_dims = ValidateShape(param_.shape, x_dims);
  param_.output->Resize(output_dims);
  auto out_lod = param_.output->mutable_lod();
  *out_lod = param_.x->lod();
  return true;
}

bool ReshapeOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  auto x_var = scope->FindVar(opdesc.Input("X").front());
  auto output_var = scope->FindVar(opdesc.Output("Out").front());
  CHECK(x_var);
  CHECK(output_var);
  param_.x = const_cast<lite::Tensor *>(&(x_var->Get<lite::Tensor>()));
  param_.output = output_var->GetMutable<lite::Tensor>();
  std::vector<std::string> input_arg_names = opdesc.InputArgumentNames();
  if (opdesc.HasAttr("inplace")) {
    param_.inplace = opdesc.GetAttr<bool>("inplace");
  }
  CHECK(param_.x) << "Input(X) of ReshapeOp should not be null.";
  CHECK(param_.output) << "Output(Out) of ReshapeOp should not be null.";

  if (opdesc.HasInput("ShapeTensor") &&
      opdesc.Input("ShapeTensor").size() > 0) {
    auto inputs = opdesc.Input("ShapeTensor");
    for (auto var : inputs) {
      lite::Tensor *datatensor =
          scope->FindVar(var)->GetMutable<lite::Tensor>();
      param_.shape.push_back(datatensor->mutable_data<int>()[0]);
    }
    const std::vector<int> shape_vector = param_.shape;
    lite::Tensor *shape_tensor = new lite::Tensor;

    shape_tensor->Resize({static_cast<int64_t>(shape_vector.size())});
    int *data_shape = shape_tensor->mutable_data<int>();
    for (int i = 0; i < shape_vector.size(); i++) {
      data_shape[i] = shape_vector[i];
    }
    param_.actual_shape = shape_tensor;
    return true;
  } else if (opdesc.HasInput("Shape") && opdesc.Input("Shape").size() > 0) {
    auto actual_shape_var = scope->FindVar(opdesc.Input("Shape").front());
    if (actual_shape_var != nullptr) {
      param_.actual_shape =
          const_cast<lite::Tensor *>(&(actual_shape_var->Get<lite::Tensor>()));
      int length = param_.actual_shape->dims().production();
      int *shape_list = actual_shape_var->GetMutable<int>();
      param_.shape.assign(shape_list, shape_list + length);
    }
    return true;
  } else {
    param_.shape = opdesc.GetAttr<std::vector<int>>("shape");
    CHECK(!param_.shape.empty())
        << "The shape information must be set by Attr(shape).";
    const std::vector<int> shape_vector = param_.shape;
    lite::Tensor *shape_tensor = new lite::Tensor;

    shape_tensor->Resize({static_cast<int64_t>(shape_vector.size())});
    int *data_shape = shape_tensor->mutable_data<int>();
    for (int i = 0; i < shape_vector.size(); i++) {
      data_shape[i] = shape_vector[i];
    }
    param_.actual_shape = shape_tensor;
  }
  return true;
}

bool Reshape2Op::CheckShape() const {
  ReshapeOp::CheckShape();
  CHECK_OR_FALSE(param_.xshape);
  return true;
}

bool Reshape2Op::InferShape() const {
  ReshapeOp::InferShape();
  auto x_dims = param_.x->dims();
  std::vector<DDim::value_type> xshape_dims(x_dims.size() + 1, 1);
  for (size_t i = 0; i < x_dims.size(); i++) {
    xshape_dims[i + 1] = x_dims[i];
  }
  param_.xshape->Resize(xshape_dims);
  return true;
}

bool Reshape2Op::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  ReshapeOp::AttachImpl(opdesc, scope);
  auto xshape_var = scope->FindVar(opdesc.Output("XShape").front());
  CHECK(xshape_var);
  param_.xshape = xshape_var->GetMutable<lite::Tensor>();
  CHECK(param_.xshape) << "Output(XShape) of ReshapeOp should not be null.";
  return true;
}

DDim ValidateShape(const std::vector<int> &shape, const DDim &input_dims) {
  const lite::DDim::value_type input_size = input_dims.production();
  auto input_shape = input_dims.Vectorize();
  bool all_positive = std::all_of(
      input_shape.cbegin(), input_shape.cend(), [](lite::DDim::value_type i) {
        return i > 0;
      });
  // only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int unk_dim_val = -1;
  const int copy_dim_val = 0;

  std::vector<lite::DDim::value_type> output_shape(shape.size(), 0);
  lite::DDim::value_type capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      CHECK_EQ(unk_dim_idx, -1)
          << "Only one input dimension of Attr(shape) can be unknown.";
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      CHECK_LT(static_cast<int>(i), input_shape.size())
          << "The index of dimension to copy from input shape must be less "
             "than the size of input shape.";
    } else {
      CHECK_GT(shape[i], 0) << "Each input dimension of Attr(shape) must not "
                               "be negtive except one unknown dimension.";
    }

    capacity *= (shape[i] ? static_cast<lite::DDim::value_type>(shape[i])
                          : input_shape[i]);
    output_shape[i] = (shape[i] ? static_cast<lite::DDim::value_type>(shape[i])
                                : input_shape[i]);
  }

  if (unk_dim_idx != -1) {
    if (all_positive) {
      // input_size < 0 and is un-determinate in compile time, skip the check,
      // for example, input_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, input_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_shape[unk_dim_idx] = -input_size / capacity;
      CHECK_EQ(output_shape[unk_dim_idx] * capacity, -input_size)
          << "Invalid shape is given.";
    } else {
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    CHECK_EQ(capacity, input_size) << "Invalid shape is given.";
  }
  return lite::DDim(output_shape);
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(reshape, paddle::lite::operators::ReshapeOp);
REGISTER_LITE_OP(reshape2, paddle::lite::operators::Reshape2Op);
