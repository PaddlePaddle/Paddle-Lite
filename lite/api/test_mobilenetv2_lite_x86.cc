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
#include "lite/api/lite_api_test_helper.h"
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/utils/cp_logging.h"
// for googlenet

namespace paddle {
namespace lite {

TEST(Mobilenet_v2, test_mobilenetv2_lite_x86) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  auto predictor = lite_api::CreatePaddlePredictor(config);

  auto input_tensor = predictor->GetInput(0);
  std::vector<int64_t> input_shape{1, 3, 224, 224};
  input_tensor->Resize(input_shape);
  auto* data = input_tensor->mutable_data<float>();
  int input_num = 1;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    input_num *= input_shape[i];
  }
  for (int i = 0; i < input_num; i++) {
    data[i] = 1;
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";
  std::vector<std::vector<float>> results;
  // i = 1
  results.emplace_back(std::vector<float>(
      {0.00017082224, 5.699624e-05,  0.000260885,   0.00016412718,
       0.00034818667, 0.00015230637, 0.00032959113, 0.0014772735,
       0.0009059976,  9.5378724e-05, 5.386537e-05,  0.0006427285,
       0.0070957416,  0.0016094646,  0.0018807327,  0.00010506048,
       6.823785e-05,  0.00012269315, 0.0007806194,  0.00022354358}));
  auto out = predictor->GetOutput(0);
  ASSERT_EQ(out->shape().size(), 2u);
  ASSERT_EQ(out->shape()[0], 1);
  ASSERT_EQ(out->shape()[1], 1000);

  int step = 50;
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = 0; j < results[i].size(); ++j) {
      EXPECT_NEAR(out->data<float>()[j * step + (out->shape()[1] * i)],
                  results[i][j],
                  1e-6);
    }
  }
}

}  // namespace lite
}  // namespace paddle
