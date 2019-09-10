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

#include <gtest/gtest.h>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

TEST(tensor, test) {
  TensorLite tensor;
  DDimLite ddim({1, 8});
  tensor.Resize(ddim);

  for (int i = 0; i < 8; i++) {
    tensor.mutable_data<int>()[i] = i;
  }
}

#ifdef LITE_WITH_OPENCL
TEST(tensor, test_ocl_image2d) {
  using DTYPE = float;
  const size_t N = 1;
  const size_t C = 3;
  const size_t H = 5;
  const size_t W = 7;

  Tensor x;
  DDim x_dims{N, C, H, W};
  x.mutable_data<DTYPE, cl::Image2D>(x_dims);
}
#endif

}  // namespace lite
}  // namespace paddle
