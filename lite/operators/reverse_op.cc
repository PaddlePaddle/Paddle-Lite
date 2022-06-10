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

#include "lite/operators/reverse_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ReverseOpLite::CheckShape() const {
  CHECK(param_.X || param_.X_array);
  CHECK(param_.Out || param_.Out_array);
  if (param_.X) {
    for (auto axis : param_.Axis) {
      CHECK_LT(axis, static_cast<int>((param_.X)->dims().size()));
      CHECK_GE(axis, static_cast<int>(-(param_.X)->dims().size()));
    }
  }
  return true;
}

bool ReverseOpLite::InferShapeImpl() const {
  if (param_.X) {
    param_.Out->Resize(param_.X->dims());
  } else if (param_.X_array) {
    param_.Out_array->resize(param_.X_array->size());
    for (size_t i = 0; i < param_.X_array->size(); i++) {
      param_.Out_array->at(i).Resize(param_.X_array->at(i).dims());
    }
  } else {
    LOG(FATAL) << "x or x_array must be set.";
  }
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool ReverseOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x_name = op_desc.Input("X").front();
  auto out_name = op_desc.Output("Out").front();

  auto x_var = scope->FindVar(x_name);
  if (x_var->IsType<Tensor>()) {
    param_.X = scope->FindMutableTensor(x_name);
    param_.Out = scope->FindMutableTensor(out_name);
  } else if (x_var->IsType<std::vector<Tensor>>()) {
    param_.X_array = x_var->GetMutable<std::vector<Tensor>>();
    param_.Out_array =
        scope->FindVar(out_name)->GetMutable<std::vector<Tensor>>();
  } else {
    LOG(FATAL) << "X type for reverse op is unsupported. Expected type is "
                  "tensor or tensor_array.";
  }
  param_.Axis = op_desc.GetAttr<std::vector<int>>("axis");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(reverse, paddle::lite::operators::ReverseOpLite);
