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

#include "lite/operators/pixel_unshuffle_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool PixelUnShuffleOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.downscale_factor > 0);

  const auto x_dims = param_.x->dims();
  const auto downscale_factor = param_.downscale_factor;

  // check input tensor dims size
  CHECK_EQ_OR_FALSE(x_dims.size(), 4);

  // check if the height and width can be divided by downscale_factor
  CHECK_EQ_OR_FALSE(x_dims[2] % downscale_factor, 0);
  CHECK_EQ_OR_FALSE(x_dims[3] % downscale_factor, 0);

  return true;
}

bool PixelUnShuffleOpLite::InferShapeImpl() const {
  const auto x_dims = param_.x->dims();
  const auto downscale_factor = param_.downscale_factor;

  // input tensor dims
  int N = x_dims[0];
  int C = x_dims[1];
  int H = x_dims[2];
  int W = x_dims[3];

  // output tensor dims
  int out_C = C * (downscale_factor * downscale_factor);
  int out_H = H / downscale_factor;
  int out_W = W / downscale_factor;

  // make sure the output height and width can be divided by downscale_factor
  if (H % downscale_factor != 0 || W % downscale_factor != 0) {
    return false;
  }

  DDim output_dims({N, out_C, out_H, out_W});
  param_.output->Resize(output_dims);
  return true;
}

bool PixelUnShuffleOpLite::AttachImpl(const cpp::OpDesc& opdesc,
                                      lite::Scope* scope) {
  auto input = opdesc.Input("X").front();
  auto out = opdesc.Output("Out").front();

  param_.x = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.output = scope->FindVar(out)->GetMutable<lite::Tensor>();

  if (opdesc.HasAttr("downscale_factor")) {
    param_.downscale_factor = opdesc.GetAttr<int>("downscale_factor");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(pixel_unshuffle,
                 paddle::lite::operators::PixelUnShuffleOpLite);
