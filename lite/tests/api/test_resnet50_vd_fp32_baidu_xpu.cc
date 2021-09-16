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
DEFINE_int32(channel, 3, "image channel");

namespace paddle {
namespace lite {

float CalOutAccuracy(
    const std::unordered_map<std::string, std::vector<float>>& out_rets,
    const std::unordered_map<std::string, int>& val_label,
    const int top_n = 1) {
  int right_num = 0;
  for (auto& out : out_rets) {
    int label = val_label.at(out.first);

    auto cmp = [](const std::pair<int, float> a,
                  const std::pair<int, float> b) {
      return a.second < b.second;
    };
    std::priority_queue<std::pair<int, float>,
                        std::vector<std::pair<int, float>>,
                        decltype(cmp)>
        out_queue(cmp);
    for (size_t j = 0; j < out.second.size(); j++) {
      out_queue.push(std::make_pair(static_cast<int>(j), out.second[j]));
    }
    for (int j = 0; j < top_n; j++) {
      auto tmp = out_queue.top();
      out_queue.pop();
      if (tmp.first == label) {
        right_num++;
        break;
      }
    }
  }
  return static_cast<float>(right_num) / static_cast<float>(out_rets.size());
}

std::unordered_map<std::string, int> ReadLabels(const std::string& label_dir) {
  auto lines = ReadLines(label_dir);
  std::unordered_map<std::string, int> labels;
  for (auto line : lines) {
    std::string image = Split(line, " ")[0];
    int label = std::stoi(Split(line, " ")[1]);
    labels[image] = label;
  }
  return labels;
}

std::unordered_map<std::string, std::vector<float>> ReadInputData(
    const std::string& input_data_dir,
    const std::unordered_map<std::string, int>& labels) {
  std::unordered_map<std::string, std::vector<float>> input_data;
  for (auto label : labels) {
    std::string image_name = Split(label.first, "/")[1];
    std::string image_dir = input_data_dir + "/" + image_name;
    std::vector<float> image_data;
    CHECK(ReadFile(image_dir, &image_data));
    input_data[label.first] = image_data;
  }
  return input_data;
}

TEST(Resnet50_vd, test_resnet50_vd_fp32_baidu_xpu) {
  lite_api::CxxConfig config;
  config.set_model_file(FLAGS_model_dir + "/inference.pdmodel");
  config.set_param_file(FLAGS_model_dir + "/inference.pdiparams");
  config.set_valid_places({lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  config.set_xpu_l3_cache_method(16773120, false);
  auto predictor = lite_api::CreatePaddlePredictor(config);

  std::string labels_dir = FLAGS_data_dir + std::string("/val_list.txt");
  auto labels = ReadLabels(labels_dir);
  std::string input_data_dir = FLAGS_data_dir + std::string("/input_data");
  auto input_data = ReadInputData(input_data_dir, labels);

  const std::vector<int> input_shape{
      FLAGS_batch, FLAGS_channel, FLAGS_im_width, FLAGS_im_height};
  // warmup
  for (int i = 0; i < 1; ++i) {
    std::vector<int64_t> shape(input_shape.begin(), input_shape.end());
    FillTensor(
        predictor, 0, shape, std::vector<float>(ShapeProduction(shape), 0.f));
    predictor->Run();
  }

  const int image_size = FLAGS_channel * FLAGS_im_width * FLAGS_im_height;
  std::unordered_map<std::string, std::vector<float>> out_rets;
  double cost_time = 0;
  std::vector<std::string> input_images;
  for (auto image_in : input_data) {
    input_images.push_back(image_in.first);
    if (input_images.size() < static_cast<size_t>(FLAGS_batch)) {
      continue;
    }
    auto input_tensor = predictor->GetInput(0);
    input_tensor->Resize(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()));
    auto* data = input_tensor->mutable_data<float>();
    for (auto image : input_images) {
      memcpy(data, input_data[image].data(), sizeof(float) * image_size);
      data += image_size;
    }

    double start = GetCurrentUS();
    predictor->Run();
    cost_time += GetCurrentUS() - start;

    auto output_tensor = predictor->GetOutput(0);
    auto output_shape = output_tensor->shape();
    auto output_data = output_tensor->data<float>();
    ASSERT_EQ(output_shape.size(), 2UL);
    ASSERT_EQ(output_shape[0], static_cast<int64_t>(FLAGS_batch));
    ASSERT_EQ(output_shape[1], 102);

    for (auto image : input_images) {
      std::vector<float> out(output_shape[1]);
      memcpy(&(out.at(0)), output_data, sizeof(float) * output_shape[1]);
      out_rets[image] = out;
      output_data += output_shape[1];
    }
    input_images.clear();
  }

  float top1_acc = CalOutAccuracy(out_rets, labels, 1);
  float top5_acc = CalOutAccuracy(out_rets, labels, 5);
  ASSERT_GT(top1_acc, 0.94f);
  ASSERT_GT(top5_acc, 0.98f);

  float speed = cost_time / (input_data.size() / FLAGS_batch) / 1000.0;

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", batch: " << FLAGS_batch
            << ", iteration: " << input_data.size() << ", spend " << speed
            << " ms in average.";
}

}  // namespace lite
}  // namespace paddle
