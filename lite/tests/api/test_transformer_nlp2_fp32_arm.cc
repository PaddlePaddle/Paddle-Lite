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
#include <cstring>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/test/lite_api_test_helper.h"
#include "lite/api/test/test_helper.h"
#include "lite/utils/log/cp_logging.h"

DEFINE_string(data_dir, "", "data dir");

namespace paddle {
namespace lite {

template <class T>
void ShowVector(const std::string& info, const std::vector<T>& vct) {
  LOG(INFO) << info;

  std::stringstream ss;
  for (auto x : vct) {
    ss << x << " ";
  }
  LOG(INFO) << ss.str();
}

// Read the nums in a line for txt file
template <typename T>
std::vector<T> ReadLineNums(std::istream& is) {
  std::string line;
  std::getline(is, line);
  std::stringstream ss(line);

  std::vector<T> res;
  T x;
  while (ss >> x) {
    res.push_back(x);
  }
  return res;
}

TEST(TRANSFORMER_FP32_MODEL, test_transformer_nlp2_fp32_arm) {
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  // Use the full api with CxxConfig to generate the optimized model
  lite_api::CxxConfig cxx_config;
  cxx_config.set_model_file(FLAGS_model_dir + "/model");
  cxx_config.set_param_file(FLAGS_model_dir + "/params");
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

  // prepare
  std::ifstream fs(FLAGS_data_dir);
  ASSERT_TRUE(fs.is_open()) << "open input file " << FLAGS_data_dir
                            << " error.";
  int sample_nums = 0;
  fs >> sample_nums;
  fs.get();
  LOG(INFO) << "sample nums:" << sample_nums;

  // loop
  double all_time = 0;
  int right_num = 0;
  for (int i = 0; i < sample_nums; i++) {
    LOG(INFO) << "iter:" << i;

    // set input
    std::vector<int64_t> input_shape = ReadLineNums<int64_t>(fs);
    std::vector<int64_t> input_data = ReadLineNums<int64_t>(fs);
    ShowVector("input_shape:", input_shape);
    ShowVector("input_data:", input_data);

    auto input_tensor = predictor->GetInput(0);
    input_tensor->Resize(input_shape);
    auto input_data_ptr = input_tensor->mutable_data<int64_t>();
    std::memcpy(
        input_data_ptr, input_data.data(), sizeof(int64_t) * input_data.size());

    // run
    for (int i = 0; i < FLAGS_warmup; i++) predictor->Run();

    auto start = GetCurrentUS();
    for (int i = 0; i < FLAGS_repeats; i++) predictor->Run();
    auto end = GetCurrentUS();
    all_time += (end - start);

    // check
    std::vector<int64_t> out_shape_gt = ReadLineNums<int64_t>(fs);
    std::vector<int64_t> out_data_gt = ReadLineNums<int64_t>(fs);
    ShowVector("out_shape_gt:", out_shape_gt);
    ShowVector("out_data_gt:", out_data_gt);

    auto out = predictor->GetOutput(0);
    std::vector<int64_t> out_shape_pd = out->shape();
    auto* out_data = out->data<int64_t>();
    std::vector<int64_t> out_data_pd(out_data,
                                     out_data + ShapeProduction(out_shape_pd));
    ShowVector("out_shape_pd:", out_shape_pd);
    ShowVector("out_data_pd:", out_data_pd);

    if (out_shape_pd == out_shape_gt && out_data_pd == out_data_gt) {
      ++right_num;
    }
  }
  fs.close();

  LOG(INFO) << "================== Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", avg tims: " << all_time / FLAGS_repeats / sample_nums / 1000.0
            << " ms.";

  float acc = static_cast<float>(right_num) / sample_nums;
  LOG(INFO) << "accuracy:" << acc;

  ASSERT_GE(acc, 1.0);
}

}  // namespace lite
}  // namespace paddle
