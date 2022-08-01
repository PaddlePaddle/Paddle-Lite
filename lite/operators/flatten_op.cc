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

#include "lite/operators/flatten_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

static std::vector<int64_t> GetOutputShape(const DDim in_dims,
                                           int start_axis,
                                           int stop_axis) {
  int64_t outer = 1;
  std::vector<int64_t> out_shape;
  size_t in_dims_size = in_dims.size();
  out_shape.reserve(in_dims_size - stop_axis + start_axis);

  for (int i = 0; i < start_axis; ++i) {
    out_shape.push_back(in_dims[i]);
  }
  for (int i = start_axis; i <= stop_axis; i++) {
    outer *= in_dims[i];
  }
  out_shape.push_back(outer);
  for (size_t i = stop_axis + 1; i < in_dims_size; i++) {
    out_shape.push_back(in_dims[i]);
  }
  return out_shape;
}

bool FlattenOp::CheckShape() const {
  CHECK(param_.x);
  CHECK(param_.output);
  return true;
}

bool FlattenOp::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  int64_t outer = 1, inner = 1;
  for (size_t i = 0; i < x_dims.size(); ++i) {
    if (i < axis_) {
      outer *= x_dims[i];
    } else {
      inner *= x_dims[i];
    }
  }
  std::vector<int64_t> out_shape(2);
  out_shape[0] = outer;
  out_shape[1] = inner;

  param_.output->Resize(out_shape);
  if (x_dims[0] == out_shape[0]) {
    param_.output->set_lod(param_.x->lod());
  }
  return true;
}

bool FlattenOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.x = scope->FindTensor(opdesc.Input("X").front());
  param_.output = scope->FindMutableTensor(opdesc.Output("Out").front());
  axis_ = opdesc.GetAttr<int>("axis");
  CHECK_GE(axis_, 0) << "Flatten op axis should greater than or equal to 0.";
  if (opdesc.HasAttr("inplace")) {
    param_.inplace = opdesc.GetAttr<bool>("inplace");
  }
  return true;
}

bool Flatten2Op::CheckShape() const {
  FlattenOp::CheckShape();
  CHECK(param_.xshape);
  return true;
}

bool Flatten2Op::InferShapeImpl() const {
  FlattenOp::InferShapeImpl();
  auto x_shape = param_.x->dims().Vectorize();
  x_shape.insert(x_shape.begin(), 0);
  param_.xshape->Resize(x_shape);
  return true;
}

bool Flatten2Op::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  FlattenOp::AttachImpl(opdesc, scope);
  param_.xshape = scope->FindMutableTensor(opdesc.Output("XShape").front());
  return true;
}

bool FlattenContiguousRangeOp::AttachImpl(const cpp::OpDesc &opdesc,
                                          lite::Scope *scope) {
  param_.x = scope->FindTensor(opdesc.Input("X").front());
  param_.out = scope->FindMutableTensor(opdesc.Output("Out").front());

  if (opdesc.HasOutput("XShape")) {
    param_.xshape = scope->FindMutableTensor(opdesc.Output("XShape").front());
  }
  param_.start_axis = opdesc.GetAttr<int>("start_axis");
  param_.stop_axis = opdesc.GetAttr<int>("stop_axis");
  return true;
}

bool FlattenContiguousRangeOp::CheckShape() const {
  CHECK(param_.x);
  CHECK(param_.out);
  CHECK(param_.xshape);
  return true;
}

bool FlattenContiguousRangeOp::InferShapeImpl() const {
  int start_axis = param_.start_axis;
  int stop_axis = param_.stop_axis;
  auto x_dims = param_.x->dims();
  int x_dims_size = x_dims.size();
  if (start_axis < 0) start_axis += x_dims_size;
  if (stop_axis < 0) stop_axis += x_dims_size;
  CHECK_OR_FALSE(start_axis <= stop_axis);

  std::vector<int64_t> out_shape =
      GetOutputShape(x_dims, start_axis, stop_axis);
  param_.out->Resize(out_shape);
  if (x_dims[0] == out_shape[0]) {
    param_.out->set_lod(param_.x->lod());
  }

  auto x_shape = x_dims.Vectorize();
  x_shape.insert(x_shape.begin(), 0);
  param_.xshape->Resize(x_shape);
  param_.xshape->set_lod(param_.x->lod());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(flatten, paddle::lite::operators::FlattenOp);
REGISTER_LITE_OP(flatten2, paddle::lite::operators::Flatten2Op);
REGISTER_LITE_OP(flatten_contiguous_range,
                 paddle::lite::operators::FlattenContiguousRangeOp);
