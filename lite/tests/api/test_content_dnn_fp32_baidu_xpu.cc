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
#include "lite/api/lite_api_test_helper.h"
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/tests/api/bert_utility.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(iteration, 100, "iteration times to run");
DEFINE_int32(batch, 10, "batch_size to run");

namespace paddle {
namespace lite {

TEST(CONTENT_DNN, test_content_dnn_fp32_baidu_xpu) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kXPU), PRECISION(kInt64)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kX86), PRECISION(kInt64)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  config.set_xpu_workspace_l3_size_per_thread();
  auto predictor = lite_api::CreatePaddlePredictor(config);

  std::string input_data_file =
      FLAGS_data_dir + std::string("/test.expand.small");
  std::vector<std::vector<int64_t>> data;
  std::vector<std::vector<uint64_t>> lod;
  ReadMmdnnRawData(input_data_file, &data, &lod);

  for (size_t i = 0; i < data.size(); i++) {
    int64_t start_pos = lod[i][0];
    int64_t end_pos = lod[i][FLAGS_batch];
    std::vector<int64_t> tensor_shape{end_pos - start_pos, 1};
    std::vector<int64_t> tensor_value(data[i].begin() + start_pos,
                                      data[i].begin() + end_pos);
    std::vector<std::vector<uint64_t>> tensor_lod{{0}};
    for (int k = 0; k < FLAGS_batch; k++) {
      tensor_lod[0].push_back(lod[i][k + 1] - start_pos);
    }
    FillTensor(predictor, i, tensor_shape, tensor_value, tensor_lod);
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  std::vector<float> out_rets;
  double cost_time = 0;
  for (int iter = 0; iter < FLAGS_iteration; iter++) {
    for (size_t i = 0; i < data.size(); i++) {
      int64_t start_pos = lod[i][iter * FLAGS_batch];
      int64_t end_pos = lod[i][(iter + 1) * FLAGS_batch];
      std::vector<int64_t> tensor_shape{end_pos - start_pos, 1};
      std::vector<int64_t> tensor_value(data[i].begin() + start_pos,
                                        data[i].begin() + end_pos);
      std::vector<std::vector<uint64_t>> tensor_lod{{0}};
      for (int k = 0; k < FLAGS_batch; k++) {
        tensor_lod[0].push_back(lod[i][iter * FLAGS_batch + k + 1] - start_pos);
      }
      FillTensor(predictor, i, tensor_shape, tensor_value, tensor_lod);
    }

    double start = GetCurrentUS();
    predictor->Run();
    cost_time += GetCurrentUS() - start;

    auto output_tensor = predictor->GetOutput(0);
    auto output_shape = output_tensor->shape();
    auto output_data = output_tensor->data<float>();
    ASSERT_EQ(output_shape.size(), 2UL);
    ASSERT_EQ(output_shape[0], FLAGS_batch);
    ASSERT_EQ(output_shape[1], 1);

    for (int i = 0; i < FLAGS_batch; i++) {
      out_rets.push_back(output_data[i]);
    }
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", batch: " << FLAGS_batch
            << ", iteration: " << FLAGS_iteration << ", spend "
            << cost_time / FLAGS_iteration / 1000.0 << " ms in average.";

  std::string ref_out_file = FLAGS_data_dir + std::string("/ref_out.txt");
  float out_accuracy = CalMmdnnOutAccuracy(out_rets, ref_out_file);
  ASSERT_GT(out_accuracy, 0.99f);
}

}  // namespace lite
}  // namespace paddle
