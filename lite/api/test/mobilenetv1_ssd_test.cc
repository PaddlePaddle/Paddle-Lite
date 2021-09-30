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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/api/cxx_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {

#ifdef LITE_WITH_ARM
void TestModel(const std::vector<Place>& valid_places) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_NO_BIND, FLAGS_threads);
  lite::Predictor predictor;

  predictor.Build(FLAGS_model_dir, "", "", valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 300, 300})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor.Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor.Run();
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  std::vector<std::vector<float>> results;
  // i = 1
  results.emplace_back(std::vector<float>(
      {3, 0.042103, 0.00439525, 0.0234783, 1.01127, 0.990756}));
  results.emplace_back(std::vector<float>(
      {5, 0.0145793, 0.00860882, 0.0344975, 1.01375, 1.00129}));
  results.emplace_back(std::vector<float>(
      {8, 0.560059, 0.00439525, 0.0234783, 1.01127, 0.990756}));
  results.emplace_back(std::vector<float>(
      {9, 0.0165109, -0.0020006, 0.0013622, 0.999179, 0.991846}));
  results.emplace_back(std::vector<float>(
      {12, 0.0263337, -0.0020006, 0.0013622, 0.999179, 0.991846}));
  results.emplace_back(std::vector<float>(
      {15, 0.0116742, 0.00580454, 0.0321349, 1.00545, 0.98476}));
  results.emplace_back(std::vector<float>(
      {17, 0.0405541, 0.00860882, 0.0344975, 1.01375, 1.00129}));
  results.emplace_back(std::vector<float>(
      {18, 0.0231487, -0.00245976, 0.00771075, 1.01654, 1.00395}));
  results.emplace_back(std::vector<float>(
      {19, 0.0133921, 0.00860882, 0.0344975, 1.01375, 1.00129}));
  results.emplace_back(std::vector<float>(
      {20, 0.039664, 0.00860882, 0.0344975, 1.01375, 1.00129}));

  auto* out = predictor.GetOutput(0);
  ASSERT_EQ(out->dims().size(), 2);
  ASSERT_EQ(out->dims()[0], 10);
  ASSERT_EQ(out->dims()[1], 6);
  ASSERT_EQ(out->lod().size(), 1);
  ASSERT_EQ(out->lod()[0].size(), 2);
  ASSERT_EQ(out->lod()[0][0], 0);
  ASSERT_EQ(out->lod()[0][1], 10);

  for (int i = 0; i < results.size(); ++i) {
    for (int j = 0; j < results[i].size(); ++j) {
      EXPECT_NEAR(
          out->data<float>()[j + (out->dims()[1] * i)], results[i][j], 5e-6);
    }
  }
}

TEST(MobileNetV1_SSD, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  TestModel(valid_places, Place({TARGET(kARM), PRECISION(kFloat)}));
}

#endif  // LITE_WITH_ARM

}  // namespace lite
}  // namespace paddle
