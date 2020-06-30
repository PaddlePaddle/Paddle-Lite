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

#include "lite/operators/assign_value_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool AssignValueOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.Out);
  auto shape = param_.shape;
  auto int32_values = param_.int32_values;
  auto fp32_values = param_.fp32_values;
  auto int64_values = param_.int64_values;
  auto bool_values = param_.bool_values;
  size_t shape_num = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    shape_num *= shape[i];
  }
  CHECK_OR_FALSE(
      shape_num == int32_values.size() || shape_num == fp32_values.size() ||
      shape_num == int64_values.size() || shape_num == bool_values.size());
  return true;
}

bool AssignValueOpLite::InferShapeImpl() const {
  std::vector<int> shape = param_.shape;
  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < shape.size(); i++) out_shape.push_back(shape[i]);
  param_.Out->Resize(out_shape);
  return true;
}

bool AssignValueOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                   lite::Scope *scope) {
  param_.shape = op_desc.GetAttr<std::vector<int>>("shape");
  param_.dtype = op_desc.GetAttr<int>("dtype");
  if (op_desc.HasAttr("fp32_values")) {
    param_.fp32_values = op_desc.GetAttr<std::vector<float>>("fp32_values");
  }
  if (op_desc.HasAttr("int32_values")) {
    param_.int32_values = op_desc.GetAttr<std::vector<int>>("int32_values");
  }
  if (op_desc.HasAttr("int64_values")) {
    param_.int64_values = op_desc.GetAttr<std::vector<int64_t>>("int64_values");
  }
  if (op_desc.HasAttr("bool_values")) {
    param_.bool_values = op_desc.GetAttr<std::vector<int>>("bool_values");
  }
  auto out = op_desc.Output("Out").front();
  param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(assign_value, paddle::lite::operators::AssignValueOpLite);
