// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <queue>
#include <unordered_map>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test/test_helper.h"
#include "lite/utils/io.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/string.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(batch, 16, "batch of image");

namespace paddle {
namespace lite {

float CompareDiffWithCpu(
    const std::unordered_map<std::string, std::vector<float>>& outs,
    const std::unordered_map<std::string, std::vector<float>>& cpu_outs,
    const float threshold = 0.5,
    const float score_diff = 0.01,
    const int box_diff = 1) {
  // compare out_xpu with out_cpu.
  // For boxes which socres > threshold, if abs(score_xpu - score_cpu) <
  // score_diff and
  // abs(box_xpu - box_cpu) < box_diff and label is right, it's correct.

  int right_num = 0;
  for (auto cpu_out : cpu_outs) {
    std::string img_name = cpu_out.first;
    if (outs.count(img_name) < 1) continue;
    std::vector<float> ref_box = cpu_out.second;
    std::vector<float> box(outs.at(img_name));

    bool is_right = true;
    for (size_t i = 1; i < ref_box.size(); i += 6) {
      if (ref_box[i] < threshold) continue;
      if (std::fabs(ref_box[i - 1] - box[i - 1]) > 1e-6 ||
          std::fabs(ref_box[i] - box[i]) > 0.01 ||
          std::abs(static_cast<int>(ref_box[i + 1]) -
                   static_cast<int>(box[i + 1])) > box_diff ||
          std::abs(static_cast<int>(ref_box[i + 2]) -
                   static_cast<int>(box[i + 2])) > box_diff ||
          std::abs(static_cast<int>(ref_box[i + 3]) -
                   static_cast<int>(box[i + 3])) > box_diff ||
          std::abs(static_cast<int>(ref_box[i + 4]) -
                   static_cast<int>(box[i + 4])) > box_diff) {
        is_right = false;
      }
    }
    right_num += is_right;
  }

  return static_cast<float>(right_num) / static_cast<float>(outs.size());
}

std::unordered_map<std::string, std::vector<float>> ReadCpuOut(
    const std::string& cpu_out_dir) {
  auto lines = ReadLines(cpu_out_dir);
  std::unordered_map<std::string, std::vector<float>> cpu_out;
  for (auto line : lines) {
    std::string image = Split(line, ":")[0];
    std::vector<float> out = Split<float>(Split(line, ":")[1], " ");
    cpu_out[image] = out;
  }
  return cpu_out;
}

void ReadInputData(
    const std::string& input_data_dir,
    std::unordered_map<std::string, std::vector<float>>* input0,
    std::unordered_map<std::string, std::vector<int>>* input1,
    const std::unordered_map<std::string, std::vector<float>>& cpu_out) {
  for (auto it : cpu_out) {
    std::string input_dir = input_data_dir + "/" + it.first;
    if (!IsFileExists(input_dir)) continue;
    std::vector<float> input0_data;
    CHECK(ReadFile(input_dir, &input0_data));
    (*input0)[it.first] = input0_data;
  }

  auto lines = ReadLines(input_data_dir + "/img_shape.txt");
  for (auto line : lines) {
    std::string input_name = Split(line, ":")[0];
    if (input0->count(input_name) == 0) continue;
    std::vector<int> input1_data = Split<int>(Split(line, ":")[1], " ");
    (*input1)[input_name] = input1_data;
  }
  return;
}

TEST(yolov3_darknet53, test_yolov3_darknet53_fp32_baidu_xpu) {
  lite_api::CxxConfig config;
  config.set_model_file(FLAGS_model_dir + "/__model__");
  config.set_param_file(FLAGS_model_dir + "/__params__");
  config.set_valid_places({lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  config.set_xpu_l3_cache_method(16773120, false);

  auto predictor = lite_api::CreatePaddlePredictor(config);

  std::string cpu_out_dir =
      FLAGS_data_dir + std::string("/output_data/out_cpu.txt");
  auto cpu_out = ReadCpuOut(cpu_out_dir);
  std::string input_data_dir = FLAGS_data_dir + std::string("/input_data");
  std::unordered_map<std::string, std::vector<float>> input0;
  std::unordered_map<std::string, std::vector<int>> input1;
  ReadInputData(input_data_dir, &input0, &input1, cpu_out);

  const std::vector<int> input_shape{FLAGS_batch, 3, 608, 608};
  // warmup.
  for (int i = 0; i < 1; ++i) {
    std::vector<int64_t> shape(input_shape.begin(), input_shape.end());
    FillTensor(
        predictor, 0, shape, std::vector<float>(ShapeProduction(shape), 0.f));
    FillTensor(
        predictor, 1, {shape[0], 2L}, std::vector<int>(shape[0] * 2, 400));
    predictor->Run();
  }

  const int image_size = ShapeProduction(input_shape) / FLAGS_batch;
  std::unordered_map<std::string, std::vector<float>> out_rets;
  double cost_time = 0;
  std::vector<std::string> input_names;
  std::vector<float> input0_data(ShapeProduction(input_shape));
  std::vector<int> input1_data(FLAGS_batch * 2);
  for (auto in0 : input0) {
    input_names.push_back(in0.first);
    int batch_i = input_names.size() - 1;
    memcpy(&(input0_data.at(batch_i * image_size)),
           in0.second.data(),
           sizeof(float) * image_size);
    memcpy(&(input1_data.at(batch_i * 2)),
           input1.at(in0.first).data(),
           sizeof(int) * 2);
    if (batch_i + 1 < FLAGS_batch) continue;

    std::vector<int64_t> shape(input_shape.begin(), input_shape.end());
    FillTensor(predictor, 0, shape, input0_data);
    FillTensor(predictor, 1, {shape[0], 2L}, input1_data);

    double start = GetCurrentUS();
    predictor->Run();
    cost_time += GetCurrentUS() - start;

    auto output_tensor = predictor->GetOutput(0);
    auto output_shape = output_tensor->shape();
    auto output_data = output_tensor->data<float>();
    auto output_lod = output_tensor->lod();
    ASSERT_EQ(output_shape.size(), 2UL);
    ASSERT_EQ(output_shape[1], 6L);
    ASSERT_EQ(output_lod.size(), 1UL);
    ASSERT_EQ(output_lod[0].size(), static_cast<size_t>(FLAGS_batch + 1));

    for (size_t i = 1; i < output_lod[0].size(); i++) {
      int out_size = (output_lod[0][i] - output_lod[0][i - 1]) * 6;
      std::vector<float> out(out_size);
      memcpy(&(out.at(0)), output_data, sizeof(float) * out_size);
      out_rets[input_names[i - 1]] = out;
      output_data += out_size;
    }
    input_names.clear();
  }

  float score = CompareDiffWithCpu(out_rets, cpu_out);
  ASSERT_GT(score, 0.98f);
  float speed = cost_time / (input0.size() / FLAGS_batch) / 1000.0;

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", batch: " << FLAGS_batch
            << ", iteration: " << input0.size() << ", spend " << speed
            << " ms in average.";
}

}  // namespace lite
}  // namespace paddle
