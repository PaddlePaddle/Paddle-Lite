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

#include "lite/operators/softmax_op.h"

#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SoftmaxOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  auto x_dims = param_.x->dims();
  auto x_rank = x_dims.size();
  CHECK_OR_FALSE(param_.axis >= -static_cast<int>(x_rank) &&
                 param_.axis < static_cast<int>(x_rank));
  return true;
}

bool SoftmaxOp::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  if (param_.eleminate_success) {
    param_.x->Resize({x_dims[0], x_dims[1]});
  }
  param_.output->Resize(param_.x->dims());
  auto out_lod = param_.output->mutable_lod();
  *out_lod = param_.x->lod();

  return true;
}

bool SoftmaxOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.x = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  param_.output =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();

  if (opdesc.HasAttr("axis")) {
    param_.axis = opdesc.GetAttr<int>("axis");
  } else {
    param_.axis = -1;
  }

  if (opdesc.HasAttr("eleminate_success")) {
    param_.eleminate_success = opdesc.GetAttr<bool>("eleminate_success");
  }

  CHECK(param_.x);
  CHECK(param_.output);
  if (opdesc.HasAttr("use_cudnn")) {
    param_.use_cudnn = opdesc.GetAttr<bool>("use_cudnn");
  }
  // TODO(wilber): use cudnn default when compile with cuda.
  param_.use_cudnn = true;
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(softmax, paddle::lite::operators::SoftmaxOp);
