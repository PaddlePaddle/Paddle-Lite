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

#include "lite/operators/write_back_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool WriteBackOp::CheckShape() const {
  if (!param_.tensor_array_copy) {
    CHECK(param_.x);
    CHECK(param_.y);
    return true;
  } else {
    CHECK(param_.array_x);
    CHECK(param_.array_y);
    return true;
  }
}

bool WriteBackOp::InferShapeImpl() const {
  if (!param_.tensor_array_copy) {
    param_.y->Resize(param_.x->dims());
    param_.y->set_lod(param_.x->lod());
    param_.y->set_precision(param_.x->precision());
    param_.y->set_persistable(param_.x->persistable());
    return true;
  }
  return false;
}

bool WriteBackOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  if (opdesc.HasAttr("tensor_array_copy")) param_.tensor_array_copy = true;
  if (!param_.tensor_array_copy) {
    param_.x = scope->FindMutableTensor(opdesc.Input("Src_LoDTensor").front());
    param_.y = scope->FindMutableTensor(opdesc.Input("Dst_LoDTensor").front());
    return true;
  } else {
    auto src = opdesc.Input("Src_LoDTensorArray").front();
    auto dst = opdesc.Input("Dst_LoDTensorArray").front();
    param_.array_x = scope->FindVar(src)->GetMutable<std::vector<Tensor>>();
    param_.array_y = scope->FindVar(dst)->GetMutable<std::vector<Tensor>>();
    return true;
  }
  return false;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(write_back, paddle::lite::operators::WriteBackOp);
