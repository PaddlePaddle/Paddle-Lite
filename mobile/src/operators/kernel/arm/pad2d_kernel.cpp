/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PAD2D_OP

#include "operators/kernel/pad2d_kernel.h"
#include "operators/math/pad.h"

namespace paddle_mobile {
namespace operators {

template <>
bool Pad2DKernel<CPU, float>::Init(Pad2DParam<CPU> *param) {
  return true;
}

template <>
void Pad2DKernel<CPU, float>::Compute(const Pad2DParam<CPU> &param) {
  const auto *input = param.InputX();
  auto *output = param.Out();
  const auto &paddings = param.paddings_;
  //  if (param.mode_ == "constant" && param.pad_value_ == 0) {
  math::PadFunctor<CPU, float> pad;
  pad(*input, paddings[0], paddings[1], paddings[2], paddings[3], output);
  //  } else {
  //    PADDLE_MOBILE_THROW_EXCEPTION("Pad2D has not been implemented.");
  //  }
  output->set_lod(input->lod());
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // PAD2D_OP
