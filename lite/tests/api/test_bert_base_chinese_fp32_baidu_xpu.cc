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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test/lite_api_test_helper.h"
#include "lite/api/test/test_helper.h"
#include "lite/utils/log/cp_logging.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(warmup_num, 1, "warmup rounds");

namespace paddle {
namespace lite {

void ReadInputData(const std::string& input_file,
                   std::vector<std::vector<int64_t>>* input0,
                   std::vector<std::vector<int64_t>>* input1,
                   std::vector<std::vector<int64_t>>* input_shapes) {
  auto lines = ReadLines(input_file);
  for (auto line : lines) {
    std::vector<std::string> shape_and_data = Split(line, ":");
    std::vector<int64_t> input_shape = Split<int64_t>(shape_and_data[0], " ");
    input_shapes->emplace_back(input_shape);
    std::vector<int64_t> input0_data = Split<int64_t>(shape_and_data[1], " ");
    input0->emplace_back(input0_data);
    input1->emplace_back(std::vector<int64_t>(input0_data.size(), 0));
  }
}

float CompareDiff(const std::vector<std::vector<float>>& outs,
                  const std::string& ref_out_file,
                  float diff = 0.01) {
  auto lines = ReadLines(ref_out_file);
  std::vector<std::vector<float>> ref_outs;
  for (auto line : lines) {
    ref_outs.push_back(Split<float>(Split(line, ":")[1], " "));
  }

  size_t all_num = 0;
  size_t right_num = 1;
  for (size_t i = 0; i < outs.size(); i++) {
    CHECK_EQ(outs[i].size(), ref_outs[i].size());
    all_num += outs[i].size();
    for (size_t j = 0; j < outs[i].size(); j++) {
      if (std::fabs(outs[i][j] - ref_outs[i][j]) < 0.01f) {
        right_num++;
      }
    }
  }
  return static_cast<float>(right_num) / static_cast<float>(all_num);
}

TEST(bert_base_chinese, test_bert_base_chinese_fp32_baidu_xpu) {
  lite_api::CxxConfig config;
  config.set_model_file(FLAGS_model_dir + "/model.pdmodel");
  config.set_param_file(FLAGS_model_dir + "/model.pdiparams");
  config.set_valid_places({lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  config.set_xpu_workspace_l3_size_per_thread();
  auto predictor = lite_api::CreatePaddlePredictor(config);

  std::string input_data_file = FLAGS_data_dir + std::string("/input0.txt");
  std::vector<std::vector<int64_t>> input0;
  std::vector<std::vector<int64_t>> input1;
  std::vector<std::vector<int64_t>> input_shapes;
  ReadInputData(input_data_file, &input0, &input1, &input_shapes);

  for (int i = 0; i < FLAGS_warmup_num; ++i) {
    FillTensor(predictor, 0, input_shapes[i], input0[i]);
    FillTensor(predictor, 1, input_shapes[i], input1[i]);
    predictor->Run();
  }

  std::vector<std::vector<float>> out_rets(input0.size());
  double cost_time = 0;
  for (size_t i = 0; i < input0.size(); ++i) {
    FillTensor(predictor, 0, input_shapes[i], input0[i]);
    FillTensor(predictor, 1, input_shapes[i], input1[i]);

    double start = GetCurrentUS();
    predictor->Run();
    cost_time += GetCurrentUS() - start;

    auto output_tensor = predictor->GetOutput(0);
    auto output_shape = output_tensor->shape();
    auto output_data = output_tensor->data<float>();
    ASSERT_EQ(output_shape.size(), 3UL);
    ASSERT_EQ(output_shape[0], input_shapes[i][0]);
    ASSERT_EQ(output_shape[1], input_shapes[i][1]);
    ASSERT_EQ(output_shape[2], 7);

    auto output_size = ShapeProduction(output_shape);
    out_rets[i].resize(output_size);
    memcpy(&(out_rets[i].at(0)), output_data, sizeof(float) * output_size);
  }

  std::string ref_out_file = FLAGS_data_dir + std::string("/out_gpu.txt");
  float diff = 0.01;
  float acc = CompareDiff(out_rets, ref_out_file, diff);
  ASSERT_GT(acc, 0.99);
  float latency = cost_time / input0.size() / 1000.0;
  ASSERT_LT(latency, 31.f);

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup_num
            << ", batch: " << input_shapes[0][0] << ", sequence num: "
            << input_shapes[0][0] * (input_shapes.size() - 1) +
                   input_shapes[input_shapes.size() - 1][0]
            << ", latency: " << latency << " ms in average, acc: " << acc
            << " if diff with gpu is less than " << diff;
}

}  // namespace lite
}  // namespace paddle
