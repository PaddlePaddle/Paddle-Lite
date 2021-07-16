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

#include "lite/operators/gather_tree_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool GatherTreeOp::CheckShape() const {
  CHECK(param_.ids);
  CHECK(param_.parents);
  CHECK(param_.out);

  auto ids_dims = param_.ids->dims();
  auto parents_dims = param_.parents->dims();
  CHECK(ids_dims.Vectorize() == parents_dims.Vectorize())
      << "ids_dims: " << ids_dims << ", parents_dims: " << parents_dims;
  return true;
}

bool GatherTreeOp::InferShapeImpl() const {
  param_.out->Resize(param_.ids->dims());
  return true;
}

bool GatherTreeOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.ids = scope->FindTensor(opdesc.Input("Ids").front());
  param_.parents = scope->FindTensor(opdesc.Input("Parents").front());
  param_.out = scope->FindMutableTensor(opdesc.Output("Out").front());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(gather_tree, paddle::lite::operators::GatherTreeOp);
