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

#include "lite/operators/one_hot_op.h"
#include <gtest/gtest.h>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

TEST(one_hot_op_lite, TestHost) {
  // prepare variables
  Scope scope;
  auto* x = scope.Var("X")->GetMutable<Tensor>();
  auto* depth_tensor = scope.Var("depth_tensor")->GetMutable<Tensor>();
  auto* output = scope.Var("Out")->GetMutable<Tensor>();
  depth_tensor->dims();
  output->dims();

  // set data
  x->Resize(DDim(std::vector<int64_t>({4, 1})));
  auto* x_data = x->mutable_data<int64_t>();
  x_data[0] = 1;
  x_data[1] = 1;
  x_data[2] = 3;
  x_data[3] = 0;

  // prepare op desc
  cpp::OpDesc desc;
  desc.SetType("one_hot");
  desc.SetInput("X", {"X"});
  desc.SetOutput("Out", {"Out"});
  desc.SetAttr("depth", static_cast<int>(4));
  desc.SetAttr("dtype", static_cast<int>(1));
  desc.SetAttr("allow_out_of_range", static_cast<bool>(0));
  OneHotOp one_hot("one_hot");
  one_hot.SetValidPlaces({Place{TARGET(kHost), PRECISION(kAny)}});
  one_hot.Attach(desc, &scope);
  auto kernels = one_hot.CreateKernels({Place{TARGET(kHost), PRECISION(kAny)}});
  ASSERT_FALSE(kernels.empty());
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(one_hot, kHost, kAny, kAny, def);
