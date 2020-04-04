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

#include "lite/operators/affine_channel_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool AffineChannelOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Scale);
  CHECK_OR_FALSE(param_.Bias);
  CHECK_OR_FALSE(param_.Out);

  const auto x_dims = param_.X->dims();
  const auto scale_dims = param_.Scale->dims();
  const auto bias_dims = param_.Bias->dims();

  CHECK_OR_FALSE(x_dims.size() == 4);
  CHECK_OR_FALSE(scale_dims.size() == 1);
  CHECK_OR_FALSE(bias_dims.size() == 1);
  CHECK_OR_FALSE(scale_dims == bias_dims);

  const std::string data_layout = param_.data_layout;
  if (data_layout == "NCHW") {
    CHECK_OR_FALSE(scale_dims[0] == x_dims[1] && bias_dims[0] == x_dims[1]);
  } else if (data_layout == "NHWC") {
    CHECK_OR_FALSE(scale_dims[0] == x_dims[3] && bias_dims[0] == x_dims[3]);
  }
  return true;
}

bool AffineChannelOpLite::InferShapeImpl() const {
  const auto x_dims = param_.X->dims();
  param_.Out->Resize(x_dims);
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool AffineChannelOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                     lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto scale = op_desc.Input("Scale").front();
  auto bias = op_desc.Input("Bias").front();
  auto output = op_desc.Output("Out").front();

  param_.X = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.Scale = scope->FindVar(scale)->GetMutable<lite::Tensor>();
  param_.Bias = scope->FindVar(bias)->GetMutable<lite::Tensor>();
  if (op_desc.HasAttr("data_layout")) {
    param_.data_layout = op_desc.GetAttr<std::string>("data_layout");
  }
  param_.Out = scope->FindVar(output)->GetMutable<lite::Tensor>();

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(affine_channel, paddle::lite::operators::AffineChannelOpLite);
