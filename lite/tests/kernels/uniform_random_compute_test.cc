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
// #include "lite/kernels/host/uniform_random_compute.h"
// #include <random>
#include "lite/kernels/host/uniform_random_compute.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/naive_math_impl.h"
#include "lite/tests/utils/tensor_utils.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace host {

TEST(uniformrandom, test) {
  using T = double;
  std::vector<int64_t> shape(2, 3);
  float min = -5.0f;
  float max = 10.0f;
  int seed = 0;
  int dtype = static_cast<int>(VarDescAPI::VarDataType::FP64);
  lite::Tensor shape_tensor, out;
  shape_tensor.Resize({1, 2});
  auto* shape_tensor_data = shape_tensor.mutable_data<int64_t>();
  shape_tensor_data[0] = 2;
  shape_tensor_data[1] = 2;
  out.Resize({shape_tensor_data[0], shape_tensor_data[1]});
  UniformRandomCompute uniform_random;
  paddle::lite::operators::UniformRandomParam param;
  param.shape_tensor = &shape_tensor;
  param.shape = shape;
  param.min = min;
  param.max = max;
  param.seed = seed;
  param.dtype = dtype;
  param.Out = &out;
  uniform_random.SetParam(param);
  uniform_random.Run();
  const double* outdata = out.data<double>();
  for (int i = 0; i < out.numel(); i++) {
    LOG(INFO) << "out.data: " << outdata[i];
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
