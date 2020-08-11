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

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/host/one_hot_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

/* note:
One Hot Operator. This operator creates the one-hot representations for input
index values. The following example will help to explain the function of this
operator:
X is a LoDTensor:
  X.lod = [[0, 1, 4]]
  X.shape = [4, 1]
  X.data = [[1], [1], [3], [0]]
set depth = 4
Out is a LoDTensor:
  Out.lod = [[0, 1, 4]]
  Out.shape = [4, 4]
  Out.data = [[0., 1., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 0., 1.],
              [1., 0., 0., 0.]]   */
TEST(one_hot, test) {
  using T = float;

  lite::Tensor x, out;
  x.Resize({4, 1});
  out.Resize({4, 4});

  auto* x_data = x.mutable_data<T>();
  x_data[0] = 1;
  x_data[1] = 1;
  x_data[2] = 3;
  x_data[3] = 0;
  auto* out_data = out.mutable_data<T>();
  float out_ref[4][4] = {
      {0, 1, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {1, 0, 0, 0}};

  OneHotCompute one_hot;
  operators::OneHotParam param;

  param.X = &x;
  param.Out = &out;
  param.depth = 4;
  // static_cast<int>(lite::core::FluidType::FP32) = 5;
  param.dtype = 5;

  one_hot.SetParam(param);
  one_hot.PrepareForRun();

  one_hot.Run();

  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_data[i], out_ref[i], 1e-5);
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(one_hot, kHost, kAny, kAny, def);
