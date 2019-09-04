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

#include "lite/operators/sequence_expand_as_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequenceExpandAsOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x)
  CHECK_OR_FALSE(param_.y)
  CHECK_OR_FALSE(param_.out)

  auto x_dims = param_.x->dims();
  CHECK_EQ_OR_FALSE(x_dims.size(), 2)
  auto y_lod = param_.y->lod();
  CHECK_EQ_OR_FALSE(y_lod.size(), 1)
  CHECK_EQ_OR_FALSE(static_cast<size_t>(x_dims[0]), y_lod[0].size() - 1)

  return true;
}

bool SequenceExpandAsOpLite::InferShape() const {
  auto x_dims = param_.x->dims();
  auto y_lod = param_.y->lod();
  auto out_dims = x_dims;

  int64_t out_first_dim = 0;
  if (y_lod[0].size() <= 1) {
    out_first_dim = x_dims[0];
  } else {
    for (size_t i = 1; i < y_lod[0].size(); ++i) {
      out_first_dim += (y_lod[0][i] - y_lod[0][i - 1]);
    }
  }
  out_dims[0] = out_first_dim;

  param_.out->Resize(out_dims);
  param_.out->set_lod(y_lod);

  return true;
}

bool SequenceExpandAsOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto y = op_desc.Input("Y").front();
  auto out = op_desc.Output("Out").front();

  param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.y = scope->FindVar(y)->GetMutable<lite::Tensor>();
  param_.out = scope->FindVar(out)->GetMutable<lite::Tensor>();

  return true;
}

#ifdef LITE_WITH_TRAIN
bool SequenceExpandAsGradOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.out_grad);

  return true;
}

bool SequenceExpandAsGradOpLite::InferShape() const {
  if (param_.x_grad) {
    param_.x_grad->Resize(param_.x->dims());
    param_.x_grad->set_lod(param_.x->lod());
  }
  return true;
}

bool SequenceExpandAsGradOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto y = op_desc.Input("Y").front();
  auto out_grad = op_desc.Iutput(framework::GradVarName("Out")).front();

  param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.y = scope->FindVar(y)->GetMutable<lite::Tensor>();
  param_.hidden_grad = scope->FindVar(out_grad)->GetMutable<lite::Tensor>();

  if (op_desc.Output(framework::GradVarName("Out")).size()) {
    auto out_grad = op_desc.Output(framework::GradVarName("Out")).front();
    param_.out_grad = scope->FindVar(out_grad)->GetMutable<lite::Tensor>();
  }

  return true;
}
#endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_expand_as, paddle::lite::operators::SequenceExpandAsOpLite)
#ifdef LITE_WITH_TRAIN
REGISTER_LITE_OP(sequence_expand_as_grad, paddle::lite::operators::SequenceExpandAsGradOpLite)
#endif
