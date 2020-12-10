// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/unstack_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool UnstackOp::CheckShape() const {
  CHECK(param_.X);
  for (auto out : param_.Out) {
    CHECK(out);
  }
  return true;
}

bool UnstackOp::InferShapeImpl() const {
  auto x = param_.X;
  auto outs = param_.Out;
  int axis = param_.axis;
  if (axis < 0) {
    axis += x->dims().size();
  }
  int num = param_.num;
  auto x_shape = x->dims().Vectorize();
  CHECK_EQ(x_shape[axis], static_cast<int64_t>(num))
      << "num(attr) should be equal to x_dims[axis]. But received x_dims: "
      << x->dims() << ", axis: " << param_.axis << ", num: " << num;

  auto out_shape = x_shape;
  out_shape.erase(out_shape.begin() + axis);
  for (auto out : outs) {
    out->Resize(out_shape);
  }
  return true;
}

bool UnstackOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  param_.X = scope->FindTensor(op_desc.Input("X").front());
  auto out_names = op_desc.Output("Y");
  for (auto out_name : out_names) {
    param_.Out.emplace_back(scope->FindMutableTensor(out_name));
  }

  param_.axis = op_desc.GetAttr<int>("axis");
  param_.num = op_desc.GetAttr<int>("num");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(unstack, paddle::lite::operators::UnstackOp);
