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

#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/x86/shape_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(shape_x86, retrive_op) {
  auto shape = KernelRegistry::Global().Create("shape");
  ASSERT_FALSE(shape.empty());
  ASSERT_TRUE(shape.front());
}

TEST(shape_x86, init) {
  ShapeCompute<float> shape;
  ASSERT_EQ(shape.precision(), PRECISION(kFloat));
  ASSERT_EQ(shape.target(), TARGET(kX86));
}

TEST(shape_x86, run_test) {
  lite::Tensor x, out;
  constexpr int batch_size = 1;
  std::vector<int64_t> x_shape{batch_size, 1, 3, 3};
  x.Resize(lite::DDim(x_shape));

  std::vector<int64_t> out_shape{4};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<int32_t>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = 1;
  }

  ShapeCompute<float> shape;
  operators::ShapeParam param;
  param.X = &x;
  param.Out = &out;

  shape.SetParam(param);
  shape.Run();

  std::vector<float> ref_results = {1, 1, 3, 3};
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_results[i], 1e-3);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(shape, kX86, kFloat, kNCHW, def);
