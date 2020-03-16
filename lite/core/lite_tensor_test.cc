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
#include <cstring>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

template <typename Dtype, TargetType Target>
void test_shared_memory_tensor() {
  const std::vector<Dtype> data({0, 1, 2, 3});
  const std::vector<int64_t> shape({2, 2});
  const size_t size = data.size() * sizeof(Dtype);
  TensorLite init_tensor;
  init_tensor.Assign<Dtype, DDim, Target>(data.data(),
                                          static_cast<DDim>(shape));
  Dtype* init_raw_data = init_tensor.mutable_data<Dtype>();

  TensorLite shared_tensor(
      std::make_shared<Buffer>(Buffer(init_raw_data, Target, size)));
  Buffer host_buffer;
  host_buffer.ResetLazy(TargetType::kHost, size);
  if (Target == TargetType::kHost) {
    CopySync<Target>(
        host_buffer.data(), init_raw_data, size, IoDirection::HtoH);
  } else {
    CopySync<Target>(
        host_buffer.data(), init_raw_data, size, IoDirection::DtoH);
  }
  EXPECT_EQ(std::memcmp(host_buffer.data(), data.data(), size), 0);

  shared_tensor.Resize({1, 5});
  ASSERT_DEATH(shared_tensor.mutable_data<Dtype>(), "");
}

TEST(tensor, shared_memory) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  test_shared_memory_tensor<float, TargetType::kHost>();
  test_shared_memory_tensor<int64_t, TargetType::kHost>();
  test_shared_memory_tensor<int8_t, TargetType::kHost>();
#ifdef LITE_WITH_CUDA
  test_shared_memory_tensor<float, TargetType::kCUDA>();
  test_shared_memory_tensor<int64_t, TargetType::kCUDA>();
  test_shared_memory_tensor<int8_t, TargetType::kCUDA>();
#endif
}

}  // namespace lite
}  // namespace paddle
