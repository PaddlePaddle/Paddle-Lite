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
TEST(unet, test) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_HIGH, FLAGS_threads);
  lite::Predictor predictor;
  std::vector<Place> valid_places({Place{TARGET(kARM), PRECISION(kFloat)}});

  predictor.Build(FLAGS_model_dir, "", "", valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 512, 512})));
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

  // std::vector<float> results({0.00078033, 0.00083865, 0.00060029, 0.00057083,
  //                            0.00070094, 0.00080584, 0.00044525, 0.00074907,
  //                            0.00059774, 0.00063654});
  //
  std::vector<std::vector<float>> results;
  // i = 1
  results.emplace_back(std::vector<float>(
      {0.9134332,  0.9652493,  0.959906,   0.96601194, 0.9704161,  0.973321,
       0.9763035,  0.9788776,  0.98090196, 0.9823532,  0.9830632,  0.98336476,
       0.9837605,  0.98430413, 0.9848935,  0.9854547,  0.9858877,  0.9862335,
       0.9865361,  0.9867324,  0.98686767, 0.9870094,  0.98710895, 0.98710257,
       0.98703253, 0.98695105, 0.98681927, 0.98661137, 0.98637575, 0.98613656,
       0.9858899,  0.98564225, 0.9853931,  0.9851323,  0.98487836, 0.9846578,
       0.9844529,  0.9842441,  0.98405427, 0.9839205,  0.98382735, 0.98373055,
       0.9836299,  0.9835474,  0.9834818,  0.9834427,  0.98343164, 0.9834163,
       0.9833809,  0.9833255,  0.9832343,  0.9831207,  0.98302484, 0.9829579,
       0.9829039,  0.98283756, 0.9827444,  0.98264474, 0.9825466,  0.98243505,
       0.982312,   0.98218083, 0.98203814, 0.981895,   0.9817609,  0.9816264,
       0.9814932,  0.9813706,  0.98124915, 0.9811211,  0.98099536, 0.9808748,
       0.98075336, 0.9806301,  0.98050594, 0.98038554, 0.980272,   0.9801562,
       0.9800356,  0.9799207,  0.9798147,  0.97971845, 0.97963905, 0.9795745,
       0.9795107,  0.97943753, 0.9793595,  0.97928876, 0.97922987, 0.9791764,
       0.97912955, 0.9790941,  0.9790663,  0.9790414,  0.9790204,  0.9790055,
       0.97899526, 0.9789867,  0.9789797,  0.9789748}));
  auto* out = predictor.GetOutput(0);
  ASSERT_EQ(out->dims().size(), 4);
  ASSERT_EQ(out->dims()[0], 1);
  ASSERT_EQ(out->dims()[1], 21);

  int step = 1;
  for (int i = 0; i < results.size(); ++i) {
    for (int j = 0; j < results[i].size(); ++j) {
      EXPECT_NEAR(out->data<float>()[j * step + (out->dims()[1] * i)],
                  results[i][j],
                  1e-6);
    }
  }
}
#endif

}  // namespace lite
}  // namespace paddle
