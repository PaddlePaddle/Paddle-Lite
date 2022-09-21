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

#include "lite/core/scalar.h"
#include <gtest/gtest.h>
#include <vector>

namespace paddle {
namespace lite {
namespace operators {
TEST(Scalar, Constrcutor) {
  bool bool_val = true;
  Scalar bool_s(bool_val);
  ASSERT_EQ(bool_val, bool_s.to<bool>());
  ASSERT_EQ(bool_s.FromTensor(), false);

  int32_t int32_val = 10;
  Scalar int32_s(int32_val);
  ASSERT_EQ(int32_val, int32_s.to<int32_t>());

  int64_t int64_val = 100;
  Scalar int64_s(int64_val);
  ASSERT_EQ(int64_val, int64_s.to<int64_t>());

  float fp32_val = 1.3;
  Scalar fp32_s(fp32_val);
  ASSERT_NEAR(fp32_val, fp32_s.to<float>(), 1e-6);

  double fp64_val = 5.6;
  Scalar fp64_s(fp64_val);
  ASSERT_NEAR(fp64_val, fp64_s.to<double>(), 1e-6);

  Scalar fp64_copy = fp64_s;
  ASSERT_NEAR(fp64_val, fp64_copy.to<double>(), 1e-6);
}

template <typename Dtype, TargetType Target>
static void InitTensor(lite::Tensor* tensor) {
  const std::vector<Dtype> data({2});
  const std::vector<int64_t> shape({1});
  tensor->Assign<Dtype, DDim, Target>(data.data(), static_cast<DDim>(shape));
}

TEST(Scalar, ConstructFromTensor) {
  lite::Tensor int32_tensor;
  InitTensor<int32_t, TargetType::kHost>(&int32_tensor);
  Scalar int32_s(&int32_tensor);
  ASSERT_EQ(int32_s.to<int32_t>(), 2);

  lite::Tensor int64_tensor;
  InitTensor<int64_t, TargetType::kHost>(&int64_tensor);
  Scalar int64_s(&int64_tensor);
  ASSERT_EQ(int64_s.to<int64_t>(), 2);

  lite::Tensor fp32_tensor;
  InitTensor<float, TargetType::kHost>(&fp32_tensor);
  Scalar fp32_s(&fp32_tensor);
  ASSERT_NEAR(fp32_s.to<float>(), 2.0, 1e-6);

  lite::Tensor fp64_tensor;
  InitTensor<double, TargetType::kHost>(&fp64_tensor);
  Scalar fp64_s(&fp64_tensor);
  ASSERT_NEAR(fp64_s.to<double>(), 2.0, 1e-6);

  int64_s.SetTensor(&fp64_tensor);
  ASSERT_EQ(int64_s.dtype(), PrecisionType::kFP64);
  ASSERT_NEAR(int64_s.to<double>(), 2.0, 1e-6);

  Scalar int32_copy(int32_s);
  ASSERT_EQ(int32_copy.to<int32_t>(), 2);
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle
