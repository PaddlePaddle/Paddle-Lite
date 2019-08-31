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
#include "lite/operators/slice_op.h"
#include <algorithm>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SliceOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool SliceOp::InferShape() const {
  CHECK_OR_FALSE(param_.Out);
  // TODO(Superjomn) Enable data sharing.
  auto in_dims = param_.X->dims();
  auto out_dims = in_dims;
  CHECK_EQ(param_.starts.size(), param_.ends.size())
      << "for slice op starts and ends must be equal";
  int dim_value, start, end;
  auto axes = param_.axes;
  auto starts = param_.starts;
  auto ends = param_.ends;
  auto decrease_axis = param_.decrease_axis;
  for (size_t i = 0; i < axes.size(); ++i) {
    dim_value = out_dims[axes[i]];
    if (dim_value > 0) {
      start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
      end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim_value);
      out_dims[axes[i]] = end - start;
    }
  }
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      out_dims[decrease_axis[i]] = 0;
    }
    for (int i = 0; i < out_dims.size(); ++i) {
      if (out_dims[i] != 0) {
        new_out_shape.push_back(out_dims[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }
    DDim new_dims;
    new_dims.ConstructFrom(new_out_shape);
    out_dims = new_dims;
  }
  param_.Out->Resize(out_dims);
  return true;
}

bool SliceOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X =
      scope->FindVar(opdesc.Input("Input").front())->GetMutable<lite::Tensor>();
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  CHECK(param_.X);
  CHECK(param_.Out);
  param_.axes = opdesc.GetAttr<std::vector<int>>("axes");
  param_.starts = opdesc.GetAttr<std::vector<int>>("starts");
  param_.ends = opdesc.GetAttr<std::vector<int>>("ends");
  if (opdesc.HasAttr("decrease_axis")) {
    param_.decrease_axis = opdesc.GetAttr<std::vector<int>>("decrease_axis");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(slice, paddle::lite::operators::SliceOp);
