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
#include <stdlib.h>
#include <sys/wait.h>
#include <thread>  // NOLINT
#include <vector>
#include "lite/api/lite_api_test_helper.h"
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/tests/api/ILSVRC2012_utility.h"
#include "lite/utils/cp_logging.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(iteration, 100, "iteration times to run");
DEFINE_int32(batch, 1, "batch of image");
DEFINE_int32(channel, 3, "image channel");

namespace paddle {
namespace lite {

void TestRunModel(std::shared_ptr<lite_api::PaddlePredictor> predictor,
                  const std::vector<std::vector<float>> &raw_data,
                  const std::vector<int> &input_shape) {
  int input_size = 1;
  for (auto i : input_shape) {
    input_size *= i;
  }

  std::vector<std::vector<float>> out_rets;
  out_rets.resize(FLAGS_iteration);
  double cost_time = 0;
  for (size_t i = 0; i < raw_data.size(); ++i) {
    auto input_tensor = predictor->GetInput(0);
    input_tensor->Resize(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()));
    auto *data = input_tensor->mutable_data<float>();
    memcpy(data, raw_data[i].data(), sizeof(float) * input_size);

    double start = GetCurrentUS();
    predictor->Run();
    cost_time += GetCurrentUS() - start;

    auto output_tensor = predictor->GetOutput(0);
    auto output_shape = output_tensor->shape();
    auto output_data = output_tensor->data<float>();
    ASSERT_EQ(output_shape.size(), 2UL);
    ASSERT_EQ(output_shape[0], 1);
    ASSERT_EQ(output_shape[1], 1000);

    int output_size = output_shape[0] * output_shape[1];
    out_rets[i].resize(output_size);
    memcpy(&(out_rets[i].at(0)), output_data, sizeof(float) * output_size);
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", batch: " << FLAGS_batch
            << ", iteration: " << FLAGS_iteration << ", spend "
            << cost_time / FLAGS_iteration / 1000.0 << " ms in average.";

  std::string labels_dir = FLAGS_data_dir + std::string("/labels.txt");
  float out_accuracy = CalOutAccuracy(out_rets, labels_dir);
  ASSERT_GT(out_accuracy, 0.6f);
}

TEST(multi_process_resnet50, test_multi_process_baidu_xpu) {
  setenv("XPU_LOCK_REQUIRED", "1", 1);
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  config.set_xpu_workspace_l3_size_per_thread();
  auto predictor0 = lite_api::CreatePaddlePredictor(config);
  auto predictor1 = lite_api::CreatePaddlePredictor(config);

  std::string raw_data_dir = FLAGS_data_dir + std::string("/raw_data");
  std::vector<int> input_shape{
      FLAGS_batch, FLAGS_channel, FLAGS_im_width, FLAGS_im_height};
  auto raw_data = ReadRawData(raw_data_dir, input_shape, FLAGS_iteration);

  pid_t fpid = fork();
  CHECK_GE(fpid, 0) << "fork failed";
  if (fpid == 0) {
    TestRunModel(predictor0, raw_data, input_shape);
    exit(1);
  } else {
    TestRunModel(predictor1, raw_data, input_shape);
  }

  unsetenv("XPU_LOCK_REQUIRED");
}

}  // namespace lite
}  // namespace paddle
