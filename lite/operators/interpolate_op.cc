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

bool InterpolateOp::InferShapeImpl() const {
  auto X = param_.X;

  int n = X->dims()[0];
  int c = X->dims()[1];
  int h = X->dims()[2];
  int w = X->dims()[3];
  int out_h;
  int out_w;

  auto SizeTensor = param_.SizeTensor;
  auto OutSize = param_.OutSize;
  auto Scale = param_.Scale;
  if (!SizeTensor.empty()) {
    CHECK_EQ(SizeTensor.size(), 2u)
        << "Input(SizeTensor)'size of Op(interpolate) must be 2. "
           "Attr(out_shape)'s length must be 2 for 4-D input tensor.";
    out_h = SizeTensor[0]->data<int>()[0];
    out_w = SizeTensor[1]->data<int>()[0];
  } else if (OutSize) {
    auto OutSize_dims = OutSize->dims();
    CHECK_EQ(OutSize_dims.size(), 1u) << "Input(OutSize)'s dims size must be 1";
    CHECK_EQ(OutSize_dims[0], 2) << "OutSize's dim[0] must be 2";
    auto OutSize_data = OutSize->data<int>();
    out_h = OutSize_data[0];
    out_w = OutSize_data[1];
  } else if (param_.out_h > 0 && param_.out_w > 0) {
    out_h = param_.out_h;
    out_w = param_.out_w;
  } else {
    float scale = -1.f;
    if (Scale) {
      auto Scale_dims = Scale->dims();
      CHECK_EQ(Scale_dims.size(), 1) << "Scale's dimension size must be 1.";
      scale = Scale->data<float>()[0];
    } else {
      scale = param_.scale;
    }
    CHECK(scale > 0) << "scale must large than 0.";
    out_h = static_cast<int>(h * scale);
    out_w = static_cast<int>(w * scale);
  }

  auto out_lod = param_.Out->mutable_lod();
  *out_lod = param_.X->lod();
  param_.Out->Resize({n, c, out_h, out_w});

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

  if (op_desc.HasInput("SizeTensor")) {
    auto size_tensor = op_desc.Input("SizeTensor");
    for (auto var : size_tensor) {
      param_.SizeTensor.push_back(
          scope->FindVar(var)->GetMutable<lite::Tensor>());
    }
  }

  if (op_desc.HasInput("Scale")) {
    auto scale_var_names = op_desc.Input("Scale");
    if (scale_var_names.size() > 0) {
      param_.Scale =
          scope->FindVar(scale_var_names.front())->GetMutable<lite::Tensor>();
    }
  } else {
    param_.Scale = nullptr;
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
  if (op_desc.HasAttr("align_mode")) {
    param_.align_mode = op_desc.GetAttr<int>("align_mode");
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
