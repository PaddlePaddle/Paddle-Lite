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
#include "lite/api/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {

void TestModel(const std::vector<Place>& valid_places,
               const Place& preferred_place) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(LITE_POWER_HIGH, FLAGS_threads);
  lite::Predictor predictor;

  predictor.Build(FLAGS_model_dir, preferred_place, valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 224, 224})));
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
      {0.00078033, 0.00083864, 0.00060029, 0.00057083, 0.00070094, 0.00080585,
       0.00044525, 0.00074907, 0.00059774, 0.00063654, 0.00061012, 0.00047342,
       0.00056656, 0.00048569, 0.00058536, 0.00055791, 0.00060953, 0.00050026,
       0.00053206, 0.0005766,  0.00049303, 0.00098111, 0.00098136, 0.00079492,
       0.00063321, 0.00116216, 0.00200527, 0.00125035, 0.0013654,  0.00128721,
       0.00094687, 0.00113662, 0.00107747, 0.00046942, 0.00065095, 0.00054033,
       0.00074176, 0.00052958, 0.00153806, 0.00064412, 0.00074696, 0.00074853,
       0.00069978, 0.00083423, 0.00104772, 0.00062703, 0.0008324,  0.00084235,
       0.00050299, 0.0008887,  0.00077417, 0.0008537,  0.001429,   0.00110349,
       0.00090407, 0.00068112, 0.00083266, 0.00076748, 0.00094665, 0.00238264,
       0.00164756, 0.00092233, 0.00103418, 0.0012954,  0.00108995, 0.00068761,
       0.0010689,  0.00051305, 0.00097589, 0.00071219, 0.00064634, 0.0011234,
       0.00065846, 0.00146709, 0.00054817, 0.00078511, 0.0007248,  0.00080298,
       0.00117994, 0.00160993, 0.00087876, 0.00062687, 0.00050534, 0.0004736,
       0.00057464, 0.00051153, 0.0006622,  0.00072636, 0.00061334, 0.00074368,
       0.00051138, 0.00060684, 0.000731,   0.00070752, 0.00069557, 0.00054018,
       0.00074147, 0.00041445, 0.00066844, 0.00043833}));
  auto* out = predictor.GetOutput(0);
  ASSERT_EQ(out->dims().size(), 2);
  ASSERT_EQ(out->dims()[0], 1);
  ASSERT_EQ(out->dims()[1], 1000);

  int step = 1;
  for (int i = 0; i < results.size(); ++i) {
    for (int j = 0; j < results[i].size(); ++j) {
      EXPECT_NEAR(out->data<float>()[j * step + (out->dims()[1] * i)],
                  results[i][j],
                  1e-6);
    }
  }
}

TEST(InceptionV4, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
      // Place{TARGET(kOpenCL), PRECISION(kFloat)},
  });

  TestModel(valid_places, Place({TARGET(kARM), PRECISION(kFloat)}));
}

TEST(InceptionV4, test_opencl) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
      Place{TARGET(kOpenCL), PRECISION(kFloat)},
  });

  TestModel(valid_places, Place({TARGET(kOpenCL), PRECISION(kFloat)}));
}

}  // namespace lite
}  // namespace paddle
