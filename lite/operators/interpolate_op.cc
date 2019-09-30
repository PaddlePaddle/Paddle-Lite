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

#include "lite/operators/interpolate_op.h"
#include <string>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool InterpolateOp::CheckShape() const {
  auto* X = param_.X;
  auto* OutSize = param_.OutSize;
  CHECK_OR_FALSE(X);
  if (OutSize != nullptr) {
    CHECK_OR_FALSE(OutSize);
  }
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool InterpolateOp::InferShape() const {
  auto* X = param_.X;
  auto* OutSize = param_.OutSize;

  int n = X->dims()[0];
  int c = X->dims()[1];
  int h = X->dims()[2];
  int w = X->dims()[3];
  int out_h;
  int out_w;

  if (OutSize != nullptr) {
    auto outsize_data = OutSize->data<int>();
    int h_out = outsize_data[0];  // HW
    int w_out = outsize_data[1];  // HW
    param_.Out->Resize({n, c, h_out, w_out});
  } else {
    if (0 >= param_.out_h && 0 >= param_.out_w) {
      out_h = h * param_.scale;
      out_w = w * param_.scale;
      out_h = out_h > 0 ? out_h : -1;
      out_w = out_w > 0 ? out_w : -1;
    } else {
      out_h = param_.out_h;
      out_w = param_.out_w;
    }
    param_.Out->Resize({n, c, out_h, out_w});
  }
  return true;
}

bool InterpolateOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  auto X = op_desc.Input("X").front();
  if (op_desc.HasInput("OutSize")) {
    auto out_size_var_names = op_desc.Input("OutSize");
    if (out_size_var_names.size() > 0) {
      param_.OutSize = scope->FindVar(out_size_var_names.front())
                           ->GetMutable<lite::Tensor>();
    }
  } else {
    param_.OutSize = nullptr;
  }
  auto Out = op_desc.Output("Out").front();
  param_.X = scope->FindVar(X)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(Out)->GetMutable<lite::Tensor>();
  if (op_desc.HasAttr("scale")) {
    param_.scale = op_desc.GetAttr<float>("scale");
  }
  if (op_desc.HasAttr("out_w")) {
    param_.out_w = op_desc.GetAttr<int>("out_w");
  }
  if (op_desc.HasAttr("out_h")) {
    param_.out_h = op_desc.GetAttr<int>("out_h");
  }
  param_.align_corners = op_desc.GetAttr<bool>("align_corners");
  param_.interp_method = op_desc.GetAttr<std::string>("interp_method");
  return true;
}

} /* namespace operators */
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_OP(nearest_interp, paddle::lite::operators::InterpolateOp);
REGISTER_LITE_OP(bilinear_interp, paddle::lite::operators::InterpolateOp);
