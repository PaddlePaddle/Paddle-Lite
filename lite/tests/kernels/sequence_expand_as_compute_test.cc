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

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#ifdef LITE_WITH_ARM
#include "lite/kernels/arm/sequence_expand_as_compute.h"
namespace paddle {
namespace lite {
TEST(sequence_expand_as, retrive_op) {
  auto sequence_expand_as =
      KernelRegistry::Global().Create("sequence_expand_as");
  ASSERT_FALSE(sequence_expand_as.empty());
  ASSERT_TRUE(sequence_expand_as.front());
}

TEST(sequence_expand_as, init) {
  paddle::lite::kernels::arm::SequenceExpandAsCompute sequence_expand_as;
  ASSERT_EQ(sequence_expand_as.precision(), PRECISION(kFloat));
  ASSERT_EQ(sequence_expand_as.target(), TARGET(kARM));
}

TEST(sequence_expand_as, run_test) {
  lite::Tensor x, y, out;
  std::vector<int64_t> x_shape{4, 1};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> y_shape{1, 5};
  y.Resize(lite::DDim(y_shape));
  std::vector<int64_t> out_shape{8, 1};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto y_data = y.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < y.dims().production(); i++) {
    y_data[i] = static_cast<float>(i);
  }

  std::vector<std::vector<uint64_t>> lod{{0, 3, 6, 7, 8}};
  y.set_lod(lod);
  paddle::lite::kernels::arm::SequenceExpandAsCompute sequence_expand_as;

  operators::SequenceExpandAsParam param;

  param.x = &x;
  param.y = &y;
  param.out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<ARMContext>();

  sequence_expand_as.SetContext(std::move(ctx));
  sequence_expand_as.SetParam(param);
  sequence_expand_as.Run();
  auto out_data = out.mutable_data<float>();

  int index = 1;
  auto out_lod = param.out->lod()[0];
  int lod_sum = out_lod[index] - out_lod[index - 1];
  LOG(INFO) << "output: ";
  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
    if (i >= lod_sum) {
      index++;
      lod_sum = out_lod[index];
    }
    ASSERT_EQ(out_data[i], x_data[index - 1]);
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(sequence_expand_as, kARM, kFloat, kNCHW, def);
#endif
