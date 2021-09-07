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

#include "lite/operators/pad3d_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool Pad3dOpLite::CheckShape() const {
  CHECK_EQ(param_.X->dims().size(), 5UL);
  CHECK_OR_FALSE(param_.Out);
  CHECK(param_.mode == "constant" || param_.mode == "reflect" ||
        param_.mode == "replicate" || param_.mode == "circular")
      << "Invilid mode.";
  CHECK_EQ(param_.paddings.size(), 6UL);
  CHECK(param_.data_format == "NCDHW" || param_.data_format == "NDHWC")
      << "Invilid data_format.";
  return true;
}

bool Pad3dOpLite::InferShapeImpl() const {
  // NCDHW
  auto x_dims = param_.X->dims();
  int out_d = x_dims[2] + param_.paddings[4] + param_.paddings[5];
  int out_h = x_dims[3] + param_.paddings[2] + param_.paddings[3];
  int out_w = x_dims[4] + param_.paddings[0] + param_.paddings[1];
  if (param_.data_format == "NDHWC") {
    out_d = x_dims[1] + param_.paddings[4] + param_.paddings[5];
    out_h = x_dims[2] + param_.paddings[2] + param_.paddings[3];
    out_w = x_dims[3] + param_.paddings[0] + param_.paddings[1];
    param_.Out->Resize(lite::DDim({x_dims[0], out_d, out_h, out_w, x_dims[4]}));
  } else {
    param_.Out->Resize(lite::DDim({x_dims[0], x_dims[1], out_d, out_h, out_w}));
  }
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool Pad3dOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  param_.X = scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  param_.Out =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<Tensor>();
  param_.mode = op_desc.GetAttr<std::string>("mode");
  param_.pad_value = op_desc.GetAttr<float>("value");
  if (op_desc.HasAttr("Paddings") && op_desc.GetAttr<bool>("Paddings")) {
    auto Paddings =
        scope->FindVar(op_desc.Input("Paddings").front())->GetMutable<Tensor>();

    if (Paddings->dims().size() != 1) {
      printf("Paddings size must be one: %d \n",
             static_cast<int>(Paddings->dims().size()));
      return false;
    }
    if (Paddings->dims()[0] != 6) {
      printf("Paddings->dims()[0] must be six: %d \n",
             static_cast<int>(Paddings->dims()[0]));
      return false;
    }
    param_.paddings = {0, 0, 0, 0, 0, 0};
  } else {
    param_.paddings = op_desc.GetAttr<std::vector<int>>("paddings");
  }
  param_.data_format = op_desc.GetAttr<std::string>("data_format");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(pad3d, paddle::lite::operators::Pad3dOpLite);
