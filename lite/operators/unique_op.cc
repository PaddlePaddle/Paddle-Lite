// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/unique_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool UniqueOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool UniqueOp::InferShapeImpl() const {
  if (param_.return_index) CHECK(param_.Indices);
  if (param_.return_inverse || !param_.is_sorted) CHECK(param_.Index);
  if (param_.return_counts) CHECK(param_.Counts);
  DDim in_dims = param_.X->dims();
  if (!param_.is_sorted) {
    CHECK_EQ(in_dims.size(), 1) << "The Input(X) should be 1-D Tensor,"
                                << "But now dims of Input(X) is "
                                << in_dims.size();
    param_.Out->Resize({1});  // need infer
    param_.Index->Resize(in_dims);
    return true;
  }
  if (param_.axis.empty()) {
    param_.Out->Resize({1});  // need infer
    if (param_.return_inverse) param_.Index->Resize(in_dims);
  } else {
    int axis_value = param_.axis[0];
    if (axis_value < 0) {
      axis_value += in_dims.size();
    }
    CHECK_LT(axis_value, in_dims.size()) << "The axis(%d) should be less than"
                                         << "the dimension size(%d) of x.";
    param_.Out->Resize({1});  // need infer
    if (param_.return_inverse) {
      param_.Index->Resize({in_dims[axis_value]});
    }
  }
  return true;
}

bool UniqueOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  CHECK(param_.X) << "Input(X) of UniqueOp should not be null.";
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());
  CHECK(param_.Out) << "Output(Out) of UniqueOp should not be null.";
  if (opdesc.HasOutput("Index")) {
    param_.Index = scope->FindMutableTensor(opdesc.Output("Index").front());
    CHECK(param_.Index) << "Output(Index) of UniqueOp should not be null.";
  }
  if (opdesc.HasOutput("Indices")) {
    param_.Indices = scope->FindMutableTensor(opdesc.Output("Indices").front());
    CHECK(param_.Indices) << "Output(Indices) of UniqueOp should not be null.";
  }
  if (opdesc.HasOutput("Counts")) {
    param_.Counts = scope->FindMutableTensor(opdesc.Output("Counts").front());
    CHECK(param_.Counts) << "Output(Counts) of UniqueOp should not be null.";
  }
  if (opdesc.HasAttr("dtype")) {
    param_.dtype = opdesc.GetAttr<int>("dtype");
  }
  if (opdesc.HasAttr("return_index")) {
    param_.return_index = opdesc.GetAttr<bool>("return_index");
  }
  if (opdesc.HasAttr("return_inverse")) {
    param_.return_inverse = opdesc.GetAttr<bool>("return_inverse");
  }
  if (opdesc.HasAttr("return_counts")) {
    param_.return_counts = opdesc.GetAttr<bool>("return_counts");
  }
  param_.axis = opdesc.GetAttr<std::vector<int>>("axis");
  if (opdesc.HasAttr("is_sorted")) {
    param_.is_sorted = opdesc.GetAttr<bool>("is_sorted");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(unique, paddle::lite::operators::UniqueOp);
