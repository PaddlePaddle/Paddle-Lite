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
      {-6.74661934e-01,
       7.40523696e-01 - 6.21755242e-01 - 7.45860696e-01 - 1.00494301e+00,
       1.18149504e-01,
       -1.24185324e+00,
       -3.02645117e-01,
       -4.00785387e-01,
       -2.07692608e-01,
       6.47164583e-01,
       -1.14950381e-01,
       -2.93140471e-01,
       -3.80539566e-01,
       -5.52867711e-01,
       -3.81976813e-02,
       5.34957230e-01,
       -4.04274255e-01,
       -3.16598490e-02,
       -2.68218815e-01,
       -1.50845259e-01,
       1.34118640e+00,
       8.56662571e-01,
       7.28657842e-01,
       1.45721704e-01,
       5.73931694e-01,
       7.31215298e-01,
       7.36926973e-01,
       -5.81491947e-01,
       -3.20370972e-01,
       -8.41052294e-01,
       4.05473530e-01,
       -2.71624058e-01,
       -5.14828444e-01,
       -9.44813132e-01,
       -6.91899836e-01,
       -2.31078506e-01,
       -6.81701183e-01,
       1.13040745e+00,
       -5.92844725e-01,
       2.10539132e-01,
       -1.82752237e-02,
       6.47306144e-01,
       3.22634727e-01,
       2.93605298e-01,
       1.03763886e-01,
       -5.32437146e-01,
       9.66050148e-01,
       -9.57293630e-01,
       -1.08309519e+00,
       -7.11931109e-01,
       -3.75701845e-01,
       5.14552355e-01,
       6.90255404e-01,
       -1.49508595e-01,
       -4.49519217e-01,
       -1.89805046e-01,
       -4.15019155e-01,
       -2.94157803e-01,
       9.30079818e-01,
       -2.17601657e-04,
       -3.56084049e-01,
       -2.27572903e-01,
       7.56741047e-01,
       1.75599754e-01,
       -5.24350762e-01,
       6.97201729e-01,
       -1.11027181e+00,
       6.47180319e-01,
       -7.17998505e-01,
       -1.60027474e-01,
       3.37199241e-01,
       -2.04888716e-01,
       1.09218264e+00,
       -4.20800418e-01,
       -4.77505326e-02,
       -1.15316200e+00,
       -2.64905952e-02,
       1.40972161e+00,
       5.07942140e-01,
       5.61829031e-01,
       2.36277089e-01,
       -3.36595476e-01,
       -1.28554717e-01,
       -1.44331670e+00,
       2.06333444e-01,
       1.02405399e-01,
       1.22171128e+00,
       -8.27865116e-03,
       -2.20889851e-01,
       -1.72840074e-01,
       3.28697786e-02,
       9.46103573e-01,
       -5.31968176e-01,
       -2.68633127e-01,
       -2.25545272e-01,
       -3.85793716e-01,
       -6.70128465e-01,
       3.53779793e-01,
       -2.61181474e-01}));
  auto* out = predictor.GetOutput(0);
  ASSERT_EQ(out->dims().size(), 2);
  ASSERT_EQ(out->dims()[0], 1);
  ASSERT_EQ(out->dims()[1], 1000);

  int step = 50;
  for (int i = 0; i < results.size(); ++i) {
    for (int j = 0; j < results[i].size(); ++j) {
      EXPECT_NEAR(out->data<float>()[j * step + (out->dims()[1] * i)],
                  results[i][j],
                  1e-6);
    }
  }
}

TEST(EfficientNetB0, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
      // Place{TARGET(kOpenCL), PRECISION(kFloat)},
  });

  TestModel(valid_places, Place({TARGET(kARM), PRECISION(kFloat)}));
}

TEST(EfficientNetB0, test_opencl) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
      Place{TARGET(kOpenCL), PRECISION(kFloat)},
  });

  TestModel(valid_places, Place({TARGET(kOpenCL), PRECISION(kFloat)}));
}

}  // namespace lite
}  // namespace paddle
