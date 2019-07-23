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

#include "lite/kernels/arm/sroi_align_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

TEST(sroi_align_arm, retrive_op) {
  auto sroi_align =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
          "sroi_align");
  ASSERT_FALSE(sroi_align.empty());
  ASSERT_TRUE(sroi_align.front());
}

TEST(sroi_align_arm, init) {
  SroiAlignCompute sroi_align;
  ASSERT_EQ(sroi_align.precision(), PRECISION(kFloat));
  ASSERT_EQ(sroi_align.target(), TARGET(kARM));
}

TEST(sroi_align_arm, compute) {}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(sroi_align, kARM, kFloat, kNCHW, def);
