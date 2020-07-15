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

#include "lite/operators/match_matrix_tensor_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool MatchMatrixTensorOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.y);
  CHECK_OR_FALSE(param_.w);
  CHECK_OR_FALSE(param_.out);
  CHECK_OR_FALSE(param_.tmp);

  DDim x_dims = param_.x->dims();
  DDim y_dims = param_.y->dims();
  DDim w_dims = param_.w->dims();
  int dim_t = param_.dim_t;

  CHECK_OR_FALSE(x_dims.size() == 2);
  CHECK_OR_FALSE(y_dims.size() == 2);
  CHECK_OR_FALSE(w_dims.size() == 3);

  CHECK_OR_FALSE(x_dims[1] == w_dims[0] && y_dims[1] == w_dims[2] &&
                 w_dims[1] == dim_t);

  return true;
}

bool MatchMatrixTensorOpLite::InferShapeImpl() const {
  const Tensor* x = param_.x;
  const Tensor* y = param_.y;
  DDim x_dims = param_.x->dims();
  DDim y_dims = param_.y->dims();
  DDim w_dims = param_.w->dims();
  int dim_t = param_.dim_t;

  const auto& x_lod = x->lod();
  CHECK_OR_FALSE(!x_lod.empty());
  const auto& x_lod_0 = x_lod[0];
  CHECK_OR_FALSE(x_lod_0.size() >= 2);
  CHECK_OR_FALSE(x_dims[0] == x_lod_0.back());

  const auto& y_lod = y->lod();
  CHECK_OR_FALSE(!y_lod.empty());
  const auto& y_lod_0 = y_lod[0];
  CHECK_OR_FALSE(y_lod_0.size() >= 2);
  CHECK_OR_FALSE(y_dims[0] == y_lod_0.back());

  CHECK_OR_FALSE(x_lod_0.size() == y_lod_0.size());

  int out_dim_0 = 0;
  for (size_t i = 1; i < x_lod_0.size(); i++) {
    int x_len = x_lod_0[i] - x_lod_0[i - 1];
    int y_len = y_lod_0[i] - y_lod_0[i - 1];
    out_dim_0 += (x_len * y_len);
  }
  out_dim_0 *= dim_t;
  int tmp_dim_0 = x_dims[0] * dim_t * x_dims[1];

  param_.out->Resize({out_dim_0, 1});
  param_.tmp->Resize({tmp_dim_0, 1});
  return true;
}

bool MatchMatrixTensorOpLite::AttachImpl(const cpp::OpDesc& op_desc,
                                         lite::Scope* scope) {
  auto x = op_desc.Input("X").front();
  auto w = op_desc.Input("W").front();
  auto y = op_desc.Input("Y").front();
  auto out = op_desc.Output("Out").front();
  auto tmp = op_desc.Output("Tmp").front();

  param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.w = scope->FindVar(w)->GetMutable<lite::Tensor>();
  param_.y = scope->FindVar(y)->GetMutable<lite::Tensor>();
  param_.out = scope->FindVar(out)->GetMutable<lite::Tensor>();
  param_.tmp = scope->FindVar(tmp)->GetMutable<lite::Tensor>();

  param_.dim_t = op_desc.GetAttr<int32_t>("dim_t");

  if (op_desc.HasAttr("fuse_relu")) {
    param_.fuse_relu = op_desc.GetAttr<bool>("fuse_relu");
  }
#ifdef LITE_WITH_XPU
  if (op_desc.HasAttr("__xpu__float_to_fix")) {
    param_.__xpu__float_to_fix = op_desc.GetAttr<bool>("__xpu__float_to_fix");
  }
  if (op_desc.HasAttr("__xpu__w_max")) {
    param_.__xpu__w_max = op_desc.GetAttr<float>("__xpu__w_max");
  }
#endif

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(match_matrix_tensor,
                 paddle::lite::operators::MatchMatrixTensorOpLite);
