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

#include "lite/kernels/host/reshape_compute.h"
#include <gtest/gtest.h>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

TEST(reshape_host, init) {
  ReshapeCompute reshape;
  ASSERT_EQ(reshape.precision(), PRECISION(kAny));
  ASSERT_EQ(reshape.target(), TARGET(kHost));
}

TEST(reshape_host, compute) {
  ReshapeCompute reshape;
  operators::ReshapeParam param;

  Tensor input;
  Tensor output;
  input.Resize({1, 2, 4, 6});
  auto* input_data = input.mutable_data<float>();
  for (int i = 0; i < input.numel(); i++) {
    input_data[i] = i;
  }
  Tensor shape_tensor;
  shape_tensor.Resize({2});
  auto* shape_tensor_data = shape_tensor.mutable_data<int>();
  shape_tensor_data[0] = 6;
  shape_tensor_data[1] = 8;

  // set param and run
  param.x = &input;
  param.shape_tensor = &shape_tensor;  // use shape_tensor
  param.inplace = false;
  param.output = &output;
  reshape.SetParam(param);
  reshape.Run();

  // check output dims
  CHECK_EQ(shape_tensor.numel(), output.numel());
  for (int i = 0; i < output.dims().size(); i++) {
    CHECK_EQ(output.dims()[i], shape_tensor_data[i]);
  }

  // check output data
  auto* output_data = output.mutable_data<float>();
  CHECK_NE(output_data, input_data);
  for (int i = 0; i < output.numel(); i++) {
    EXPECT_NEAR(output_data[i], input_data[i], 1e-6);
  }

  // use shape, set param and run
  param.shape_tensor = nullptr;
  param.shape_vct = {-1, 0, 3, 2, 1};
  reshape.SetParam(param);
  reshape.Run();

  // check output dims
  CHECK_EQ(shape_tensor.numel(), output.numel());
  for (int i = 0; i < output.dims().size(); i++) {
    CHECK_EQ(output.dims()[i], shape_tensor_data[i]);
  }

  // check output data
  output_data = output.mutable_data<float>();
  CHECK_NE(output_data, input_data);
  for (int i = 0; i < output.numel(); i++) {
    EXPECT_NEAR(output_data[i], input_data[i], 1e-6);
  }

  // check output data if inplace = true;
  param.inplace = true;
  reshape.SetParam(param);
  reshape.Run();
  output_data = output.mutable_data<float>();
  CHECK_EQ(output_data, input_data);
}

TEST(reshape, retrive_op) {
  auto reshape =
      KernelRegistry::Global()
          .Create<TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny)>("reshape");
  ASSERT_FALSE(reshape.empty());
  ASSERT_TRUE(reshape.front());
}

TEST(reshape2, retrive_op) {
  auto reshape2 =
      KernelRegistry::Global()
          .Create<TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny)>("reshape2");
  ASSERT_FALSE(reshape2.empty());
  ASSERT_TRUE(reshape2.front());
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(reshape, kHost, kAny, kAny, def);
USE_LITE_KERNEL(reshape2, kHost, kAny, kAny, def);
