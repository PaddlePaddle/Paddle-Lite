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

#include "lite/operators/index_select_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool Index_selectOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  CHECK_OR_FALSE(param_.dim >= static_cast<int>(-(param_.X)->dims().size()));
  CHECK_OR_FALSE(param_.dim < static_cast<int>((param_.X)->dims().size()));
  return true;
}

bool Index_selectOpLite::InferShapeImpl() const {
  auto x_dims = param_.X->dims();
  int x_rank = x_dims.size();

  if (param_.dim < 0) param_.dim += x_rank;
  int dim = param_.dim;

  std::vector<int64_t> out_dims;
  for (int64_t i = 0; i < dim; i++) out_dims.push_back(x_dims[i]);
  out_dims.push_back(param_.Index->dims()[0]);
  for (int64_t i = dim + 1; i < x_rank; i++) out_dims.push_back(x_dims[i]);

  // Set output dims
  param_.Out->Resize(lite::DDim(out_dims));

  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool Index_selectOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                    lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto index = op_desc.Input("Index").front();
  auto out = op_desc.Output("Out").front();

  if (op_desc.HasAttr("dim")) {
    param_.dim = op_desc.GetAttr<int32_t>("dim");
  }

  param_.X = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.Index = scope->FindVar(index)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(index_select, paddle::lite::operators::Index_selectOpLite);
