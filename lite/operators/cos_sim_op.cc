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

#include "lite/operators/cos_sim_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool CosSimOp::CheckShape() const {
  CHECK(param_.x);
  CHECK(param_.y);
  CHECK(param_.out);
  CHECK(param_.x_norm);
  CHECK(param_.y_norm);
  return true;
}

bool CosSimOp::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  auto y_dims = param_.y->dims();
  CHECK_EQ(x_dims.size(), y_dims.size())
      << "ShapeError: Ranks of Input(X) and Input(Y) must be equal. But "
         "received x_dims: "
      << x_dims << ", y_dims: " << y_dims;
  CHECK_GE(x_dims.size(), 2UL)
      << "ShapeError: Rank of Input(X) must be greater than or equal to 2.";
  CHECK_EQ(x_dims.Slice(1, x_dims.size()), y_dims.Slice(1, y_dims.size()))
      << "All dimensions except the 1st of Input(X) and Input(Y) must be "
         "equal.";
  CHECK(x_dims[0] == y_dims[0] || y_dims[0] == 1)
      << "The 1st dimension of Input(Y) must be equal to Input(X) or just 1 "
         "(which will be broadcasted to match Input(X)). But received x_dims: "
      << x_dims << ", y_dims: " << y_dims;

  param_.out->Resize({x_dims[0], 1});
  param_.x_norm->Resize({x_dims[0], 1});
  param_.y_norm->Resize({y_dims[0], 1});
  param_.out->set_lod(param_.x->lod());
  return true;
}

bool CosSimOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  param_.x = scope->FindTensor(op_desc.Input("X").front());
  param_.y = scope->FindTensor(op_desc.Input("Y").front());
  param_.out = scope->FindMutableTensor(op_desc.Output("Out").front());
  param_.x_norm = scope->FindMutableTensor(op_desc.Output("XNorm").front());
  param_.y_norm = scope->FindMutableTensor(op_desc.Output("YNorm").front());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(cos_sim, paddle::lite::operators::CosSimOp);
