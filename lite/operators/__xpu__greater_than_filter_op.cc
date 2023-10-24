// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/__xpu__greater_than_filter_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUGreaterThanFilterOp::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);

  const auto input_dims = param_.input->dims();
  CHECK_EQ_OR_FALSE(input_dims.size(), 2UL);
  return true;
}

bool XPUGreaterThanFilterOp::InferShapeImpl() const {
  const auto& input_dims = param_.input->dims();
  param_.output->Resize(input_dims);
  // share LoD
  param_.output->set_lod(param_.input->lod());

  return true;
}

bool XPUGreaterThanFilterOp::AttachImpl(const cpp::OpDesc& op_desc,
                                        lite::Scope* scope) {
  CHECK(scope->FindVar(op_desc.Input("X").front()));
  CHECK(scope->FindVar(op_desc.Output("Out").front()));

  param_.input =
      scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  param_.output =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<Tensor>();
  param_.scale = op_desc.GetAttr<float>("scale");

  CHECK(param_.input);
  CHECK(param_.output);
  CHECK(param_.scale);

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__greater_than_filter,
                 paddle::lite::operators::XPUGreaterThanFilterOp);
