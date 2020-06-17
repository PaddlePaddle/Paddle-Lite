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

#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/arm/merge_lod_tensor_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

TEST(merge_lod_tensor_arm, retrive_op) {
  auto kernel = KernelRegistry::Global().Create("merge_lod_tensor");
  ASSERT_FALSE(kernel.empty());
  ASSERT_TRUE(kernel.front());
}

TEST(merge_lod_tensor_arm, init) {
  MergeLodTensorCompute cpt;
  ASSERT_EQ(cpt.precision(), PRECISION(kFloat));
  ASSERT_EQ(cpt.target(), TARGET(kARM));
}

TEST(merge_lod_tensor_arm_0, compute) {
  DeviceInfo::Init();
  Tensor x;
  Tensor mask;
  Tensor in_true;
  Tensor in_false;
  Tensor out;
  int level = 0;

  // set dims and lod
  mask.Resize({3, 1});

  in_true.Resize({1, 1});
  LoD in_true_lod;
  std::vector<uint64_t> in_true_lod0 = {0, 1};
  in_true_lod.push_back(in_true_lod0);
  in_true.set_lod(in_true_lod);

  in_false.Resize({4, 1});
  LoD in_false_lod;
  std::vector<uint64_t> in_false_lod0 = {0, 2, 4};
  in_false_lod.push_back(in_false_lod0);
  in_false.set_lod(in_false_lod);

  // initialize data
  auto* in_true_data = in_true.mutable_data<float>();
  for (size_t i = 0; i < in_true.numel(); i++) {
    in_true_data[i] = static_cast<float>(i);
  }
  auto* in_false_data = in_false.mutable_data<float>();
  for (size_t i = 0; i < in_false.numel(); i++) {
    in_false_data[i] = static_cast<float>(i + 1);
  }
  auto* mask_data = mask.mutable_data<bool>();
  for (size_t i = 0; i < mask.numel(); i++) {
    mask_data[i] = static_cast<bool>(i % 2);
  }

  // prepare kernel params and run to obtain output_data
  MergeLodTensorCompute op;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<ARMContext>();
  op.SetContext(std::move(ctx));

  operators::MergeLodTensorParam param;
  param.x = &x;
  param.mask = &mask;
  param.in_true = &in_true;
  param.in_false = &in_false;
  param.out = &out;
  param.level = level;
  op.SetParam(param);
  op.Launch();

  auto* out_data = out.data<float>();
  std::vector<float> out_ref = {1, 2, 0, 3, 4};
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_data[i], out_ref[i], 1e-5);
  }
}
TEST(merge_lod_tensor_arm_1, compute) {
  DeviceInfo::Init();
  Tensor x;
  Tensor mask;
  Tensor in_true;
  Tensor in_false;
  Tensor out;
  int level = 0;

  // set dims and lod
  mask.Resize({3, 1});

  in_true.Resize({3, 3});
  LoD in_true_lod = {{0, 1}, {0, 3}};
  in_true.set_lod(in_true_lod);

  in_false.Resize({6, 3});
  LoD in_false_lod = {{0, 2, 4}, {0, 1, 3, 5, 6}};
  in_false.set_lod(in_false_lod);

  // initialize data
  auto* in_true_data = in_true.mutable_data<float>();
  for (size_t i = 0; i < in_true.numel(); i++) {
    in_true_data[i] = static_cast<float>(i);
  }
  auto* in_false_data = in_false.mutable_data<float>();
  for (size_t i = 0; i < in_false.numel(); i++) {
    in_false_data[i] = static_cast<float>(i + 1);
  }
  auto* mask_data = mask.mutable_data<bool>();
  for (size_t i = 0; i < mask.numel(); i++) {
    mask_data[i] = static_cast<bool>(i % 2);
  }

  // prepare kernel params and run to obtain output_data
  MergeLodTensorCompute op;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<ARMContext>();
  op.SetContext(std::move(ctx));

  operators::MergeLodTensorParam param;
  param.x = &x;
  param.mask = &mask;
  param.in_true = &in_true;
  param.in_false = &in_false;
  param.out = &out;
  param.level = level;
  op.SetParam(param);
  op.Launch();

  auto* out_data = out.data<float>();
  std::vector<float> out_ref = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                                0,  1,  2,  3,  4,  5,  6,  7,  8,
                                10, 11, 12, 13, 14, 15, 16, 17, 18};
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_data[i], out_ref[i], 1e-5);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(merge_lod_tensor, kARM, kFloat, kNCHW, def);
