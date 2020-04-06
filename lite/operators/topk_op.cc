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

#include "lite/operators/topk_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool TopkOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  CHECK_OR_FALSE(param_.Indices);
  return true;
}

bool TopkOp::InferShapeImpl() const {
  auto out_dims = param_.X->dims();
  out_dims[out_dims.size() - 1] = param_.K;
  auto out = param_.Out;
  out->Resize(out_dims);
  out->set_lod(param_.X->lod());

  auto indices = param_.Indices;
  indices->Resize(out_dims);
  indices->set_lod(param_.X->lod());

  return true;
}

bool TopkOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  param_.X = scope->FindTensor(x);

  auto output0 = op_desc.Output("Out").front();
  auto output1 = op_desc.Output("Indices").front();
  param_.Out = scope->FindMutableTensor(output0);
  param_.Indices = scope->FindMutableTensor(output1);
  param_.K = op_desc.GetAttr<int>("k");

  CHECK_GE(param_.K, 1) << "topK param is not valid";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(top_k, paddle::lite::operators::TopkOp);
