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

#include "lite/operators/interpolate_v2_op.h"
#include <string>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool InterpolateV2Op::CheckShape() const {
  auto* X = param_.X;
  auto* OutSize = param_.OutSize;
  CHECK_OR_FALSE(X);
  if (OutSize != nullptr) {
    CHECK_OR_FALSE(OutSize);
  }
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool InterpolateV2Op::InferShapeImpl() const {
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
  } else {
    float scale_h = -1.f;
    float scale_w = -1.f;
    if (Scale) {
      auto Scale_dims = Scale->dims();
      scale_h = Scale->data<float>()[0];
      scale_w = Scale->data<float>()[1];
      out_h = static_cast<int>(h * scale_h);
      out_w = static_cast<int>(w * scale_w);
    } else {
      if (param_.scale_v.size() > 0) {
        scale_h = param_.scale_v[0];
        scale_w = param_.scale_v[1];
        CHECK_GT(scale_h, 0) << "scale_h must be greater 0.";
        CHECK_GT(scale_w, 0) << "scale_w must be greater 0.";
        out_h = static_cast<int>(h * scale_h);
        out_w = static_cast<int>(w * scale_w);
      } else {
        out_h = param_.out_h;
        out_w = param_.out_w;
      }
    }
  }

  auto out_lod = param_.Out->mutable_lod();
  *out_lod = param_.X->lod();
  param_.Out->Resize({n, c, out_h, out_w});
  return true;
}

bool InterpolateV2Op::AttachImpl(const cpp::OpDesc& op_desc,
                                 lite::Scope* scope) {
  param_.version_2 = true;
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
    param_.SizeTensor.clear();
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
    auto vs = op_desc.GetAttr<std::vector<float>>("scale");
    if (vs.size() > 0) {
      param_.scale_v = vs;
      param_.scale = vs[0];
    }
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

REGISTER_LITE_OP(bilinear_interp_v2, paddle::lite::operators::InterpolateV2Op);
REGISTER_LITE_OP(nearest_interp_v2, paddle::lite::operators::InterpolateV2Op);
