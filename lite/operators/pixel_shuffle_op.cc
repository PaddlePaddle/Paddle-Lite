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

#include "lite/operators/pixel_shuffle_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool PixelShuffleOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.upscale_factor);
  const auto x_dims = param_.x->dims();
  const auto upscale_factor = param_.upscale_factor;
  CHECK_EQ_OR_FALSE(x_dims.size(), 4);
  CHECK_EQ_OR_FALSE(x_dims[1] % (upscale_factor * upscale_factor), 0);
  return true;
}

bool PixelShuffleOpLite::InferShapeImpl() const {
  const auto x_dims = param_.x->dims();
  const auto upscale_factor = param_.upscale_factor;
  auto output_dims = x_dims;
  output_dims[0] = x_dims[0];
  output_dims[1] = x_dims[1] / (upscale_factor * upscale_factor);
  output_dims[2] = x_dims[2] * upscale_factor;
  output_dims[3] = x_dims[3] * upscale_factor;
  param_.output->Resize(output_dims);
  return true;
}

bool PixelShuffleOpLite::AttachImpl(const cpp::OpDesc& opdesc,
                                    lite::Scope* scope) {
  auto input = opdesc.Input("X").front();
  auto out = opdesc.Output("Out").front();

  param_.x = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.output = scope->FindVar(out)->GetMutable<lite::Tensor>();

  if (opdesc.HasAttr("upscale_factor")) {
    param_.upscale_factor = opdesc.GetAttr<int>("upscale_factor");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(pixel_shuffle, paddle::lite::operators::PixelShuffleOpLite);
