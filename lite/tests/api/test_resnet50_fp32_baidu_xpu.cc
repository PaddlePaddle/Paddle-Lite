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
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test/lite_api_test_helper.h"
#include "lite/api/test/test_helper.h"
#include "lite/tests/api/ILSVRC2012_utility.h"
#include "lite/utils/log/cp_logging.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(images_num,
             480,
             "images_num should be an integral multiple of batch");
DEFINE_int32(batch, 16, "batch of image");
DEFINE_int32(warmup_num, 1, "warmup rounds");

namespace paddle {
namespace lite {

TEST(resnet50, test_resnet50_fp32_baidu_xpu) {
  setenv("XPU_CONV_AUTOTUNE", "5", 1);
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  config.set_xpu_l3_cache_method(16773120, false);
  auto predictor = lite_api::CreatePaddlePredictor(config);

  std::string raw_data_dir = FLAGS_data_dir + std::string("/raw_data");
  std::vector<int> input_shape{FLAGS_batch, 3, 224, 224};
  auto raw_data =
      ReadRawData(raw_data_dir, input_shape, FLAGS_images_num / FLAGS_batch);

  std::vector<int64_t> shape(input_shape.begin(), input_shape.end());
  for (int i = 0; i < FLAGS_warmup_num; ++i) {
    FillTensor(predictor, 0, shape, raw_data[i]);
    predictor->Run();
  }

  std::vector<std::vector<float>> out_rets;
  out_rets.resize(FLAGS_images_num);
  double cost_time = 0;
  for (size_t i = 0; i < raw_data.size(); ++i) {
    FillTensor(predictor, 0, shape, raw_data[i]);

    double start = GetCurrentUS();
    predictor->Run();
    cost_time += GetCurrentUS() - start;

    auto output_tensor = predictor->GetOutput(0);
    auto output_shape = output_tensor->shape();
    auto output_data = output_tensor->data<float>();
    ASSERT_EQ(output_shape.size(), 2UL);
    ASSERT_EQ(output_shape[0], FLAGS_batch);
    ASSERT_EQ(output_shape[1], 1000);

    for (int j = 0; j < FLAGS_batch; j++) {
      out_rets[i * FLAGS_batch + j].resize(output_shape[1]);
      memcpy(&(out_rets[i * FLAGS_batch + j].at(0)),
             output_data,
             sizeof(float) * output_shape[1]);
      output_data += output_shape[1];
    }
  }

  std::string labels_dir = FLAGS_data_dir + std::string("/labels.txt");
  float out_accuracy = CalOutAccuracy(out_rets, labels_dir);
  ASSERT_GT(out_accuracy, 0.74f);
  float latency = cost_time / (FLAGS_images_num / FLAGS_batch) / 1000.0;
  ASSERT_LT(latency, 20.f);

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup_num << ", batch: " << FLAGS_batch
            << ", images_num: " << FLAGS_images_num << ", latency: " << latency
            << " ms in average, top_1 acc: " << out_accuracy;
}

}  // namespace lite
}  // namespace paddle
