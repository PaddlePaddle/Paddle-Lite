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
#include "lite/kernels/arm/split_lod_tensor_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

TEST(split_lod_tensor_arm, retrive_op) {
  auto kernel = KernelRegistry::Global().Create("split_lod_tensor");
  ASSERT_FALSE(kernel.empty());
  ASSERT_TRUE(kernel.front());
}

TEST(split_lod_tensor_arm, init) {
  SplitLodTensorCompute cpt;
  ASSERT_EQ(cpt.precision(), PRECISION(kFloat));
  ASSERT_EQ(cpt.target(), TARGET(kARM));
}

TEST(split_lod_tensor_arm_0, compute) {
  DeviceInfo::Init();
  Tensor x;
  Tensor mask;
  Tensor out_true;
  Tensor out_false;
  int level = 0;

  // set dims and lod
  VLOG(5) << "set dims and lod";
  x.Resize({5, 1});
  LoD x_lod;
  std::vector<uint64_t> x_lod0 = {0, 2, 3, 5};
  x_lod.push_back(x_lod0);
  x.set_lod(x_lod);
  mask.Resize({3, 1});
  out_true.Resize({5, 1});
  out_false.Resize({5, 1});

  // initialize data
  VLOG(5) << "initialize data";
  auto* x_data = x.mutable_data<float>();
  for (size_t i = 0; i < x.numel(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  auto* mask_data = mask.mutable_data<bool>();
  for (size_t i = 0; i < mask.numel(); i++) {
    mask_data[i] = static_cast<bool>(i % 2);
  }

  // prepare kernel params and run to obtain output_data
  VLOG(5) << "prepare kernel params";
  SplitLodTensorCompute op;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<ARMContext>();
  op.SetContext(std::move(ctx));

  VLOG(5) << "run kernel";
  operators::SplitLodTensorParam param;
  param.x = &x;
  param.mask = &mask;
  param.out_true = &out_true;
  param.out_false = &out_false;
  param.level = level;
  op.SetParam(param);
  op.Launch();

  VLOG(5) << "obtain results";
  auto* out_true_data = out_true.data<float>();
  std::vector<float> out_true_ref = {2};
  for (int i = 0; i < out_true.numel(); i++) {
    LOG(INFO) << out_true_data[i];
    EXPECT_NEAR(out_true_data[i], out_true_ref[i], 1e-5);
  }
  auto* out_false_data = out_false.data<float>();
  std::vector<float> out_false_ref = {0, 1, 3, 4};
  for (int i = 0; i < out_false.numel(); i++) {
    LOG(INFO) << out_false_data[i];
    EXPECT_NEAR(out_false_data[i], out_false_ref[i], 1e-5);
  }
}
TEST(split_lod_tensor_arm_1, compute) {
  DeviceInfo::Init();
  Tensor x;
  Tensor mask;
  Tensor out_true;
  Tensor out_false;
  int level = 0;

  // set dims and lod
  x.Resize({9, 3});
  LoD x_lod;
  std::vector<uint64_t> x_lod0 = {0, 2, 3, 5};
  std::vector<uint64_t> x_lod1 = {0, 1, 3, 6, 8, 9};
  x_lod.push_back(x_lod0);
  x_lod.push_back(x_lod1);
  x.set_lod(x_lod);
  mask.Resize({3, 1});
  out_true.Resize({9, 2});
  out_false.Resize({9, 2});

  // initialize data
  auto* x_data = x.mutable_data<float>();
  for (size_t i = 0; i < x.numel(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  auto* mask_data = mask.mutable_data<bool>();
  for (size_t i = 0; i < mask.numel(); i++) {
    mask_data[i] = static_cast<bool>(i % 2);
  }

  // prepare kernel params and run to obtain output_data
  SplitLodTensorCompute op;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<ARMContext>();
  op.SetContext(std::move(ctx));

  operators::SplitLodTensorParam param;
  param.x = &x;
  param.mask = &mask;
  param.out_true = &out_true;
  param.out_false = &out_false;
  param.level = level;
  op.SetParam(param);
  op.Launch();

  auto* out_true_data = out_true.data<float>();
  std::vector<float> out_true_ref = {9, 10, 11, 12, 13, 14, 15, 16, 17};
  for (int i = 0; i < out_true.numel(); i++) {
    EXPECT_NEAR(out_true_data[i], out_true_ref[i], 1e-5);
  }
  auto* out_false_data = out_false.data<float>();
  std::vector<float> out_false_ref = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26};
  for (int i = 0; i < out_false.numel(); i++) {
    EXPECT_NEAR(out_false_data[i], out_false_ref[i], 1e-5);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(split_lod_tensor, kARM, kFloat, kNCHW, def);
