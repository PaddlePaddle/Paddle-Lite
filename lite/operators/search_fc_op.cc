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

#include "lite/operators/search_fc_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SearchFcOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.W);
  CHECK_OR_FALSE(param_.b);
  CHECK_OR_FALSE(param_.Out);

  auto x_dims = param_.X->dims();
  CHECK_EQ(x_dims.size(), 2u) << "The rank of X(Input) should be 2.";
  auto w_dims = param_.W->dims();
  CHECK_EQ(w_dims.size(), 2u) << "W should be 2-D tensor.";
  auto b_dims = param_.b->dims();
  CHECK_EQ(b_dims.size(), 1u) << "b should be 1-D tensor.";
  CHECK_EQ(w_dims[1], x_dims[1]) << "wrong shape: w_dims[1] != x_dims[1]";
  return true;
}

bool SearchFcOpLite::InferShapeImpl() const {
  auto out_size = param_.out_size;
  lite::DDim dims(std::vector<int64_t>({-1, out_size}));
  param_.Out->Resize(dims);
  return true;
}

bool SearchFcOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                lite::Scope *scope) {
  auto X = op_desc.Input("X").front();
  auto W = op_desc.Input("W").front();
  auto b = op_desc.Input("b").front();
  auto Out = op_desc.Output("Out").front();

  param_.X = scope->FindVar(X)->GetMutable<lite::Tensor>();
  param_.W = scope->FindVar(W)->GetMutable<lite::Tensor>();
  param_.b = scope->FindVar(b)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(Out)->GetMutable<lite::Tensor>();
  param_.out_size = op_desc.GetAttr<int>("out_size");

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

REGISTER_LITE_OP(search_fc, paddle::lite::operators::SearchFcOpLite);
