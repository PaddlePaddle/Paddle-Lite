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

#include "lite/operators/sequence_pool_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequencePoolOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  auto lod = param_.X->lod();
  CHECK_EQ_OR_FALSE(lod.size(), 1UL);
  auto dims = param_.X->dims();
  CHECK_GE_OR_FALSE(dims[0], (static_cast<int64_t>(lod[0].size()) - 1));
  return true;
}

bool SequencePoolOp::InferShapeImpl() const {
  const auto *input = param_.X;
  auto out_dims = input->dims();
  out_dims[0] = input->lod()[0].size() - 1;
  param_.Out->Resize(out_dims);
  param_.MaxIndex->Resize(out_dims);
  return true;
}

bool SequencePoolOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  param_.MaxIndex = scope->FindVar(opdesc.Output("MaxIndex").front())
                        ->GetMutable<lite::Tensor>();
  param_.pool_type = opdesc.GetAttr<std::string>("pooltype");
  CHECK(param_.X);
  CHECK(param_.Out);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_pool, paddle::lite::operators::SequencePoolOp);
