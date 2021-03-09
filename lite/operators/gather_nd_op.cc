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

#include "lite/operators/gather_nd_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool GatherNdOp::CheckShape() const {
  CHECK(param_.x);
  CHECK(param_.index);
  CHECK(param_.out);
  return true;
}

bool GatherNdOp::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  auto x_dims_size = x_dims.size();
  auto index_dims = param_.index->dims();
  auto index_dims_size = index_dims.size();
  CHECK_LE(index_dims[index_dims_size - 1], static_cast<int64_t>(x_dims_size));

  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < index_dims_size - 1; i++) {
    out_shape.emplace_back(index_dims[i]);
  }
  for (size_t i = index_dims[index_dims_size - 1]; i < x_dims_size; i++) {
    out_shape.emplace_back(x_dims[i]);
  }
  param_.out->Resize(out_shape);
  param_.out->set_lod(param_.x->lod());
  return true;
}

bool GatherNdOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.x = scope->FindTensor(opdesc.Input("X").front());
  param_.index = scope->FindTensor(opdesc.Input("Index").front());
  param_.out = scope->FindMutableTensor(opdesc.Output("Out").front());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(gather_nd, paddle::lite::operators::GatherNdOp);
