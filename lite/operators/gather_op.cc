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
#include "lite/operators/gather_op.h"
#include <algorithm>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool GatherOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Index);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool GatherOp::InferShapeImpl() const {
  auto index_dims = param_.Index->dims();
  CHECK(index_dims.size() == 1 ||
        (index_dims.size() == 2 && index_dims[1] == 1))
      << "index dims unmatch";
  int batch_size = index_dims[0];
  auto out_dims = param_.X->dims();
  out_dims[0] = batch_size;
  param_.Out->Resize(out_dims);
  return true;
}

bool GatherOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  param_.Index = scope->FindTensor(opdesc.Input("Index").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());
  CHECK(param_.X) << "X is null";
  CHECK(param_.Index) << "index is null";
  CHECK(param_.Out) << "out is null";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(gather, paddle::lite::operators::GatherOp);
