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

#ifdef CONV_OP

#include "operators/math/winograd/winograd.h"
#include "operators/math/winograd/winograd_transform.h"

namespace paddle_mobile {
namespace operators {
namespace math {

// F(2X2, 3X3)
void winograd_f2k3(const framework::Tensor &input,
                   const framework::Tensor &weight, framework::Tensor *output) {
}
// F(6X6, 3X3)
void winograd_f6k3(const framework::Tensor &input,
                   const framework::Tensor &weight, framework::Tensor *output) {
  framework::Tensor transformed_input;
  framework::Tensor transformed_weight;
  // transform weight
  winograd_transform_weight<8, 3>(weight, &transformed_weight);
  // tile input and transform
  winograd_transform_input<8, 3>(input, &transformed_input);
  // caculate output
  winograd_transform_output<8, 3>(transformed_input, transformed_weight,
                                  output);
}

// F(4X4, 5X5)
void winograd_f4k5(const framework::Tensor &input,
                   const framework::Tensor &weight, framework::Tensor *output) {
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif
