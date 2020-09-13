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
#include <iostream>
#include <vector>
#include "lite/api/lite_api_test_helper.h"
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/io.h"

namespace paddle {
namespace lite {

TEST(Resnet50, test_resnet50_fp32_xpu) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  config.set_xpu_workspace_l3_size_per_thread();
  auto predictor = lite_api::CreatePaddlePredictor(config);

  auto input_tensor = predictor->GetInput(0);
  std::vector<int64_t> input_shape{1, 3, 224, 224};
  input_tensor->Resize(input_shape);
  auto* data = input_tensor->mutable_data<float>();
  int input_num = 1;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    input_num *= input_shape[i];
  }
  std::string data_path = FLAGS_data_dir + std::string("/tabby_cat.data");
  std::ifstream fin(data_path, std::ios::in | std::ios::binary);
  CHECK(fin.is_open()) << "failed to open file " << data_path;
  fin.seekg(0, std::ios::end);
  auto file_size = fin.tellg();
  fin.seekg(0, std::ios::beg);
  fin.read(reinterpret_cast<char*>(data), file_size);
  fin.close();

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  auto out = predictor->GetOutput(0);
  auto out_shape = out->shape();
  auto out_data = out->data<float>();
  ASSERT_EQ(out_shape.size(), 2UL);
  ASSERT_EQ(out_shape[0], 1);
  ASSERT_EQ(out_shape[1], 1000);

  std::string label_path = FLAGS_data_dir + std::string("/synset_words.txt");
  std::vector<std::string> labels = ReadLines(label_path);
  for (size_t i = 0; i < labels.size(); i++) {
    if (labels[i].empty()) {
      labels.erase(labels.begin() + i);
      i--;
      continue;
    }
    labels[i].erase(labels[i].begin(), labels[i].begin() + 10);
  }
  CHECK_EQ(labels.size(), out_shape[1]);
  std::vector<std::tuple<float, std::string>> results;
  for (size_t i = 0; i < labels.size(); i++) {
    results.push_back(std::make_tuple(out_data[i], labels[i]));
  }
  std::sort(
      results.begin(),
      results.end(),
      [](std::tuple<float, std::string> a, std::tuple<float, std::string> b) {
        return std::get<0>(a) > std::get<0>(b);
      });

  for (int i = 0; i < 3; i++) {
    LOG(INFO) << "top" << i << ": " << std::get<0>(results[i]) << ", "
              << std::get<1>(results[i]);
  }
  ASSERT_EQ(std::get<1>(results[0]), std::string("tabby, tabby cat"));
  ASSERT_GT(std::get<0>(results[0]), 0.68f);
}

}  // namespace lite
}  // namespace paddle
