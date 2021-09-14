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
#include "lite/api/paddle_api.h"
#include "lite/api/test/lite_api_test_helper.h"
#include "lite/api/test/test_helper.h"
#include "lite/utils/log/cp_logging.h"

DEFINE_string(data_dir, "", "data dir");

namespace paddle {
namespace lite {

TEST(NLP_LSTM_INT8_MODEL, test_nlp_lstm_int8_arm) {
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  // Use the full api with CxxConfig to generate the optimized model
  lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(FLAGS_model_dir);
  cxx_config.set_valid_places(
      {lite_api::Place{TARGET(kARM), PRECISION(kFloat)},
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

  // prepare
  std::ifstream fs(FLAGS_data_dir);
  ASSERT_TRUE(fs.is_open()) << "open input file " << FLAGS_data_dir
                            << " error.";
  int img_nums = 0;
  fs >> img_nums;

  // loop
  double all_time = 0;
  int cls_right_num = 0;
  int ctc_right_num = 0;
  for (int i = 0; i < img_nums; i++) {
    if (i % 10 == 0) {
      LOG(INFO) << "Iter:" << i;
    }

    // set input
    int64_t input_h, input_w;
    fs >> input_h >> input_w;
    auto input_tensor = predictor->GetInput(0);
    input_tensor->Resize({input_h, input_w});
    auto input_data = input_tensor->mutable_data<float>();
    for (int i = 0; i < input_h * input_w; i++) {
      fs >> input_data[i];
    }

    uint64_t lod_data;
    fs >> lod_data;
    std::vector<std::vector<uint64_t>> lod;
    lod.push_back({0, lod_data});
    input_tensor->SetLoD(lod);

    // run
    for (int i = 0; i < FLAGS_warmup; i++) predictor->Run();

    auto start = GetCurrentUS();
    for (int i = 0; i < FLAGS_repeats; i++) predictor->Run();
    auto end = GetCurrentUS();
    all_time += (end - start);

    // get output
    int gt_label;
    fs >> gt_label;
    {
      int idx = 0;
      auto out = predictor->GetOutput(idx);
      std::vector<int64_t> out_shape = out->shape();
      int64_t out_num = ShapeProduction(out_shape);
      auto* out_data = out->data<float>();
      auto max_iter = std::max_element(out_data, out_data + out_num);
      int max_idx = max_iter - out_data;
      if (max_idx == gt_label) {
        cls_right_num++;
      }
    }

    {
      int idx = 1;
      auto out = predictor->GetOutput(idx);
      auto* out_data = out->data<int64_t>();
      if (out_data[0] == gt_label) {
        ctc_right_num++;
      }
    }
  }
  fs.close();

  LOG(INFO) << "================== Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", avg tims: " << all_time / FLAGS_repeats / img_nums / 1000.0
            << " ms.";

  float cls_acc = static_cast<float>(cls_right_num) / img_nums;
  float ctc_acc = static_cast<float>(ctc_right_num) / img_nums;
  LOG(INFO) << "cls accuracy:" << cls_acc;
  LOG(INFO) << "ctc accuracy:" << ctc_acc;

  ASSERT_GE(cls_acc, 0.9566);
  ASSERT_GE(ctc_acc, 0.9666);
}

}  // namespace lite
}  // namespace paddle
