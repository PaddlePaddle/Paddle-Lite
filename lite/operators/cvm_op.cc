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

#include "lite/operators/cvm_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool CvmOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);

  const auto input_dims = param_.X->dims();
  CHECK_EQ_OR_FALSE(input_dims.size(), 2UL);

  return true;
}

bool CvmOp::InferShapeImpl() const {
  const auto& input_dims = param_.X->dims();
  const auto& use_cvm = param_.use_cvm;

  // Set output dims
  std::vector<DDim::value_type> output_dims(2);
  if (use_cvm)
    output_dims = {input_dims[0], input_dims[1]};
  else
    output_dims = {input_dims[0], input_dims[1] - 2};
  param_.Y->Resize(output_dims);

  // share LoD
  param_.Y->set_lod(param_.X->lod());

  return true;
}

bool CvmOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  CHECK(scope->FindVar(op_desc.Input("X").front()));
  CHECK(scope->FindVar(op_desc.Output("Y").front()));

  param_.X = scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  param_.Y = scope->FindVar(op_desc.Output("Y").front())->GetMutable<Tensor>();

  param_.use_cvm = op_desc.GetAttr<bool>("use_cvm");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(cvm, paddle::lite::operators::CvmOp);
