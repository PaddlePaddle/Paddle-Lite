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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test/lite_api_test_helper.h"
#include "lite/api/test/test_helper.h"
#include "lite/tests/api/bert_utility.h"
#include "lite/utils/log/cp_logging.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(iteration, 9, "iteration times to run");

namespace paddle {
namespace lite {

template <typename T>
lite::Tensor GetTensorWithShape(std::vector<int64_t> shape) {
  lite::Tensor ret;
  ret.Resize(shape);
  T* ptr = ret.mutable_data<T>();
  for (int i = 0; i < ret.numel(); ++i) {
    ptr[i] = (T)1;
  }
  return ret;
}

TEST(Ernie, test_ernie_fp32_baidu_xpu) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  config.set_xpu_l3_cache_method(16773120, false);
  // test warmup
  config.set_preferred_inputs_for_warmup<int64_t>(0, 0, {1, 64, 1});
  config.set_preferred_inputs_for_warmup<int64_t>(0, 1, {1, 64, 1});
  config.set_preferred_inputs_for_warmup<int64_t>(0, 2, {1, 64, 1});
  config.set_preferred_inputs_for_warmup<float>(0, 3, {1, 64, 1});
  config.set_preferred_inputs_for_warmup<int64_t>(2, 0, {1, 128, 1});
  config.set_preferred_inputs_for_warmup<int64_t>(2, 1, {1, 128, 1});
  config.set_preferred_inputs_for_warmup<int64_t>(2, 2, {1, 128, 1});
  config.set_preferred_inputs_for_warmup<float>(2, 3, {1, 128, 1});
  auto predictor = lite_api::CreatePaddlePredictor(config);

  std::string input_data_file = FLAGS_data_dir + std::string("/bert_in.txt");
  std::vector<std::vector<int64_t>> input0;
  std::vector<std::vector<int64_t>> input1;
  std::vector<std::vector<int64_t>> input2;
  std::vector<std::vector<int64_t>> input3;
  std::vector<std::vector<int64_t>> input_shapes;
  ReadRawData(
      input_data_file, &input0, &input1, &input2, &input3, &input_shapes);

  for (int i = 0; i < FLAGS_warmup; ++i) {
    std::vector<int64_t> shape = {1, 64, 1};
    std::vector<int64_t> fill_value(64, 0);
    for (int j = 0; j < 4; j++) {
      FillTensor(predictor, j, shape, fill_value);
    }
    predictor->Run();
  }

  std::vector<std::vector<float>> out_rets;
  out_rets.resize(FLAGS_iteration);
  double cost_time = 0;
  for (int i = 0; i < FLAGS_iteration; ++i) {
    FillTensor(predictor, 0, input_shapes[i], input0[i]);
    FillTensor(predictor, 1, input_shapes[i], input1[i]);
    FillTensor(predictor, 2, input_shapes[i], input2[i]);
    FillTensor(predictor, 3, input_shapes[i], input3[i]);

    double start = GetCurrentUS();
    predictor->Run();
    cost_time += GetCurrentUS() - start;

    auto output_tensor = predictor->GetOutput(0);
    auto output_shape = output_tensor->shape();
    auto output_data = output_tensor->data<float>();
    ASSERT_EQ(output_shape.size(), 2UL);
    ASSERT_EQ(output_shape[0], 1);
    ASSERT_EQ(output_shape[1], 1);

    int output_size = output_shape[0] * output_shape[1];
    out_rets[i].resize(output_size);
    memcpy(&(out_rets[i].at(0)), output_data, sizeof(float) * output_size);
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup
            << ", iteration: " << FLAGS_iteration << ", spend "
            << cost_time / FLAGS_iteration / 1000.0 << " ms in average.";

  std::string ref_out_file = FLAGS_data_dir + std::string("/ernie_out.txt");
  float out_accuracy = CalErnieOutAccuracy(out_rets, ref_out_file);
  ASSERT_GT(out_accuracy, 0.95f);
}

}  // namespace lite
}  // namespace paddle
