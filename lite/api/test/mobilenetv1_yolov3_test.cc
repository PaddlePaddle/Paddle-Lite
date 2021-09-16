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
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 608, 608})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 50;
  }

  auto* img_size = predictor.GetInput(1);
  img_size->Resize(DDim(std::vector<DDim::value_type>({1, 2})));
  auto* size_data = img_size->mutable_data<float>();
  size_data[0] = 608;
  size_data[1] = 608;

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
      {0., 0.7803235, 577.7447, 592.5643, 582.15314, 597.3399}));
  results.emplace_back(std::vector<float>(
      {0., 0.7643098, 473.50653, 592.58966, 478.26117, 597.2353}));
  results.emplace_back(std::vector<float>(
      {0., 0.7614112, 593.06946, 591.99646, 598.64087, 597.553}));
  results.emplace_back(std::vector<float>(
      {0., 0.7579255, 161.40321, 592.61694, 166.33885, 597.28406}));
  results.emplace_back(std::vector<float>(
      {0., 0.7569634, 193.39563, 592.62164, 198.35269, 597.2968}));
  results.emplace_back(std::vector<float>(
      {0., 0.7568337, 297.3981, 592.62024, 302.35202, 597.2969}));
  results.emplace_back(std::vector<float>(
      {0., 0.7568283, 265.39816, 592.6203, 270.35214, 597.29694}));
  results.emplace_back(std::vector<float>(
      {0., 0.74383223, 33.430492, 592.7017, 38.453976, 597.4267}));
  results.emplace_back(std::vector<float>(
      {0., 0.66492873, 9.396143, 576.7084, 15.35708, 581.8059}));
  results.emplace_back(std::vector<float>(
      {0., 0.6568178, 9.970305, 145.12535, 15.043035, 149.76646}));

  auto* out = predictor.GetOutput(0);
  ASSERT_EQ(out->dims().size(), 2);
  ASSERT_EQ(out->dims()[0], 100);
  ASSERT_EQ(out->dims()[1], 6);
  ASSERT_EQ(out->lod().size(), 1);
  ASSERT_EQ(out->lod()[0].size(), 2);
  ASSERT_EQ(out->lod()[0][0], 0);
  ASSERT_EQ(out->lod()[0][1], 100);

  int skip = 10;
  for (int i = 0; i < results.size(); i += skip) {
    for (int j = 0; j < results[i].size(); ++j) {
      EXPECT_NEAR(
          out->data<float>()[j + (out->dims()[1] * i)], results[i][j], 3e-6);
    }
  }
}

TEST(MobileNetV1_YoloV3, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  TestModel(valid_places);
}

#endif  // LITE_WITH_ARM

}  // namespace lite
}  // namespace paddle
