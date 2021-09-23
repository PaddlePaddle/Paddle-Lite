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
#include <cmath>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/test/lite_api_test_helper.h"
#include "lite/api/test/test_helper.h"
#include "lite/utils/log/cp_logging.h"

DEFINE_string(data_dir, "", "input image path");

namespace paddle {
namespace lite {

TEST(OCR_LSTM_INT8_MODEL, test_ocr_lstm_int8_arm) {
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  // Use the full api with CxxConfig to generate the optimized model
  lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(FLAGS_model_dir);
  cxx_config.set_valid_places(
      {lite_api::Place{TARGET(kARM), PRECISION(kFloat)},
       lite_api::Place{TARGET(kARM), PRECISION(kInt32)},
       lite_api::Place{TARGET(kARM), PRECISION(kInt64)}});
  predictor = lite_api::CreatePaddlePredictor(cxx_config);
  predictor->SaveOptimizedModel(FLAGS_model_dir,
                                paddle::lite_api::LiteModelType::kNaiveBuffer);

  // Use the light api with MobileConfig to load and run the optimized model
  paddle::lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(FLAGS_model_dir + ".nb");
  mobile_config.set_threads(FLAGS_threads);
  mobile_config.set_power_mode(
      static_cast<lite_api::PowerMode>(FLAGS_power_mode));
  predictor = paddle::lite_api::CreatePaddlePredictor(mobile_config);

  const int batch_size = 1;
  const int channel = 3;
  const int height = 32;
  const int width = 320;
  const int input_size = batch_size * channel * height * width;

  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize({batch_size, channel, height, width});
  auto* input_data = input_tensor->mutable_data<float>();
  ReadTxtFile(FLAGS_data_dir, input_data, input_size);

  predictor->Run();

  auto out_0 = predictor->GetOutput(0);
  int64_t out_size_0 = ShapeProduction(out_0->shape());
  auto* out_data_0 = out_0->data<int64_t>();
  ASSERT_EQ(out_size_0, 4);
  ASSERT_EQ(out_data_0[0], 389);
  ASSERT_EQ(out_data_0[1], 519);
  ASSERT_EQ(out_data_0[2], 472);
  ASSERT_EQ(out_data_0[3], 519);

  auto out_1 = predictor->GetOutput(1);
  int64_t out_size_1 = ShapeProduction(out_1->shape());
  auto* out_data_1 = out_1->data<float>();
  const float eps = 1e-3;
  int num = 0;
  for (int i = 0; i < out_size_1 && num < 10; i++) {
    if (std::abs(out_data_1[i]) > eps) {
      LOG(INFO) << "out_data_1[" << i << "]:" << out_data_1[i];
      ++num;
    }
  }
  ASSERT_EQ(out_size_1, 530000);
  ASSERT_LT(std::abs(out_data_1[6624] - 0.999878), eps);
  ASSERT_LT(std::abs(out_data_1[13249] - 0.999965), eps);
  ASSERT_LT(std::abs(out_data_1[13639] - 0.113657), 1e-2);
  ASSERT_LT(std::abs(out_data_1[20264] - 0.999932), eps);
}

}  // namespace lite
}  // namespace paddle
