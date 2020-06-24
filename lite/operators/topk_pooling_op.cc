// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/topk_pooling_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool TopkPoolingOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool TopkPoolingOp::InferShapeImpl() const {
  auto out_dims = param_.X->dims();
  out_dims[1] *= param_.top_k;
  auto out = param_.Out;
  out->Resize(out_dims);
  out->set_lod(param_.X->lod());

  return true;
}

bool TopkPoolingOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto y = op_desc.Input("Y").front();
  param_.X = scope->FindTensor(x);
  param_.Y = scope->FindTensor(y);
  auto output = op_desc.Output("Out").front();
  param_.Out = scope->FindMutableTensor(output);
  param_.top_k = op_desc.GetAttr<int>("top_k");
  param_.feat_map_num = op_desc.GetAttr<int>("feat_map_num");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(topk_pooling, paddle::lite::operators::TopkPoolingOp);
