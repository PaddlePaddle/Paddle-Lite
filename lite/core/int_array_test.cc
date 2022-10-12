// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/int_array.h"
#include <gtest/gtest.h>
#include <vector>

namespace paddle {
namespace lite {
namespace operators {

template <typename Dtype, TargetType Target>
static void InitTensor(lite::Tensor* tensor) {
  const std::vector<Dtype> data({0, 1});
  const std::vector<int64_t> shape({2});
  tensor->Assign<Dtype, DDim, Target>(data.data(), static_cast<DDim>(shape));
}

template <typename Dtype, TargetType Target>
static void InitSingleEleTensor(lite::Tensor* tensor) {
  const std::vector<Dtype> data({1});
  const std::vector<int64_t> shape({1});
  tensor->Assign<Dtype, DDim, Target>(data.data(), static_cast<DDim>(shape));
}

void CheckEqual(const IntArray& array, const std::vector<int64_t>& data) {
  auto& arr_data = array.GetData();
  ASSERT_EQ(arr_data.size(), data.size());
  for (size_t i = 0; i < array.size(); ++i) {
    ASSERT_EQ(arr_data[i], data[i]);
  }
}

void CheckEqual(const IntArray& array, const std::vector<int32_t>& data) {
  std::vector<int64_t> copy_data(data.begin(), data.end());
  CheckEqual(array, copy_data);
}

TEST(IntArray, Constructor) {
  std::vector<int32_t> data1{0, 2};
  IntArray arr1(data1);
  CheckEqual(arr1, data1);
  IntArray arr5(data1.data(), 2);
  CheckEqual(arr5, data1);

  std::vector<int64_t> data2{3, 4};
  IntArray arr2(data2);
  CheckEqual(arr2, data2);
  IntArray arr4(data2.data(), 2);
  CheckEqual(arr4, data2);

  IntArray arr3({3, 4});
  CheckEqual(arr3, data2);
  ASSERT_EQ(arr3.FromTensor(), false);
}

TEST(IntArray, ConstructFromTensor) {
  std::vector<int64_t> res{0, 1};
  lite::Tensor t1;
  InitTensor<int64_t, TargetType::kHost>(&t1);
  IntArray arr1(&t1);
  CheckEqual(arr1, res);
  ASSERT_EQ(arr1.FromTensor(), true);

  IntArray arr1_copy(arr1);
  CheckEqual(arr1_copy, res);

  std::vector<int64_t> res2{1, 1};
  lite::Tensor t2, t3;
  InitSingleEleTensor<int64_t, TargetType::kHost>(&t2);
  InitSingleEleTensor<int64_t, TargetType::kHost>(&t3);
  std::vector<const lite::Tensor*> ts{&t2, &t2};
  IntArray arr3(ts);
  CheckEqual(arr3, res2);
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle
