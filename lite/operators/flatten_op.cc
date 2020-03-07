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

bool FlattenOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  return true;
}

bool FlattenOp::InferShape() const {
  auto x_dims = param_.x->dims();

  auto out_lod = param_.output->mutable_lod();
  *out_lod = param_.x->lod();

  int64_t outer = 1, inner = 1;
  for (int i = 0; i < x_dims.size(); ++i) {
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

  return true;
}

bool FlattenOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  auto x_var = scope->FindVar(opdesc.Input("X").front());
  auto output_var = scope->FindVar(opdesc.Output("Out").front());
  CHECK(x_var);
  CHECK(output_var);
  param_.x = const_cast<lite::Tensor *>(&(x_var->Get<lite::Tensor>()));
  param_.output = output_var->GetMutable<lite::Tensor>();
  axis_ = opdesc.GetAttr<int>("axis");

  param_.inplace = false;

  CHECK(param_.x) << "Input(X) of FlattenOp should not be null.";
  CHECK(param_.output) << "Output(Out) of FlattenOp should not be null.";
  CHECK_GE(axis_, 0) << "Flatten op axis should >=0.";
  return true;
}

bool Flatten2Op::CheckShape() const {
  FlattenOp::CheckShape();
  CHECK_OR_FALSE(param_.xshape);
  return true;
}

bool Flatten2Op::InferShape() const {
  FlattenOp::InferShape();
  auto x_dims = param_.x->dims();
  std::vector<DDim::value_type> xshape_dims(x_dims.size() + 1, 0);
  for (size_t i = 0; i < x_dims.size(); i++) {
    xshape_dims[i + 1] = x_dims[i];
  }
  param_.xshape->Resize(DDim(xshape_dims));
  return true;
}

bool Flatten2Op::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  FlattenOp::AttachImpl(opdesc, scope);
  auto xshape_var = scope->FindVar(opdesc.Output("XShape").front());
  CHECK(xshape_var);
  param_.xshape = xshape_var->GetMutable<lite::Tensor>();
  CHECK(param_.xshape) << "Output(XShape) of FlattenOp should not be null.";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(flatten, paddle::lite::operators::FlattenOp);
REGISTER_LITE_OP(flatten2, paddle::lite::operators::Flatten2Op);
