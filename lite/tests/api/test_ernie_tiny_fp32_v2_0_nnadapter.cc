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
#include "lite/tests/api/utility.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(iteration, 5, "iteration times to run");

namespace paddle {
namespace lite {

TEST(ernie_tiny, test_ernie_tiny_fp32_v2_0_nnadapter) {
  FLAGS_warmup = 1;
  std::vector<std::string> nnadapter_device_names;
  std::string nnadapter_context_properties;
  std::string nnadapter_subgraph_partition_config_path;
  std::string nnadapter_subgraph_partition_config_buffer;
  std::vector<paddle::lite_api::Place> valid_places;
  valid_places.push_back(
      lite_api::Place{TARGET(kNNAdapter), PRECISION(kFloat)});
#if defined(LITE_WITH_ARM)
  valid_places.push_back(lite_api::Place{TARGET(kARM), PRECISION(kFloat)});
#elif defined(LITE_WITH_X86)
  valid_places.push_back(lite_api::Place{TARGET(kX86), PRECISION(kFloat)});
#else
  LOG(INFO) << "Unsupported host arch!";
  return;
#endif
  valid_places.push_back(lite_api::Place{TARGET(kHost), PRECISION(kFloat)});
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  nnadapter_device_names.emplace_back("huawei_ascend_npu");
  nnadapter_context_properties = "HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0";
  nnadapter_subgraph_partition_config_path =
      FLAGS_model_dir +
      "/huawei_ascend_npu_subgraph_custom_partition_config_file.txt";
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  nnadapter_device_names.emplace_back("intel_openvino");
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  nnadapter_device_names.emplace_back("qualcomm_qnn");
  nnadapter_subgraph_partition_config_buffer =
      "equal:input_ids,tmp_0:tmp_1\n"
      "cast:tmp_1:tmp_2\n"
      "scale:tmp_2:tmp_3\n"
      "unsqueeze2:tmp_3:unsqueeze2_0.tmp_0,unsqueeze2_0.tmp_1\n"
      "fill_any_like:input_ids:full_like_0.tmp_0\n"
      "cumsum:full_like_0.tmp_0:cumsum_0.tmp_0\n"
      "elementwise_sub:cumsum_0.tmp_0,full_like_0.tmp_0:tmp_4\n"
      "lookup_table_v2:input_ids,embedding_0.w_0:embedding_3.tmp_0\n"
      "lookup_table_v2:tmp_4,embedding_1.w_0:embedding_4.tmp_0\n"
      "lookup_table_v2:token_type_ids,embedding_2.w_0:embedding_5.tmp_0";
  FLAGS_warmup = 0;
  FLAGS_iteration = 1;
#else
  LOG(INFO) << "Unsupported NNAdapter device!";
  return;
#endif
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  // Use the full api with CxxConfig to generate the optimized model
  lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(FLAGS_model_dir);
  cxx_config.set_valid_places(valid_places);
  cxx_config.set_nnadapter_device_names(nnadapter_device_names);
  cxx_config.set_nnadapter_context_properties(nnadapter_context_properties);
  cxx_config.set_nnadapter_subgraph_partition_config_path(
      nnadapter_subgraph_partition_config_path);
  cxx_config.set_nnadapter_subgraph_partition_config_buffer(
      nnadapter_subgraph_partition_config_buffer);
  predictor = lite_api::CreatePaddlePredictor(cxx_config);
  predictor->SaveOptimizedModel(FLAGS_model_dir,
                                paddle::lite_api::LiteModelType::kNaiveBuffer);
  // Use the light api with MobileConfig to load and run the optimized model
  paddle::lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(FLAGS_model_dir + ".nb");
  mobile_config.set_threads(FLAGS_threads);
  mobile_config.set_power_mode(
      static_cast<lite_api::PowerMode>(FLAGS_power_mode));
  mobile_config.set_nnadapter_device_names(nnadapter_device_names);
  mobile_config.set_nnadapter_context_properties(nnadapter_context_properties);
  predictor = paddle::lite_api::CreatePaddlePredictor(mobile_config);

  // Load input_data
  auto input_lines = ReadLines(FLAGS_data_dir + "/input.txt");
  std::vector<std::vector<int64_t>> input0_data;
  std::vector<std::vector<int64_t>> input1_data;
  std::vector<std::vector<int64_t>> input0_shapes;
  std::vector<std::vector<int64_t>> input1_shapes;
  for (auto line : input_lines) {
    input0_data.push_back(
        Split<int64_t>(Split(Split(line, ";")[0], ":")[1], " "));
    input0_shapes.push_back(
        Split<int64_t>(Split(Split(line, ";")[0], ":")[0], " "));
    input1_data.push_back(
        Split<int64_t>(Split(Split(line, ";")[1], ":")[1], " "));
    input1_shapes.push_back(
        Split<int64_t>(Split(Split(line, ";")[1], ":")[0], " "));
  }

  // Load output_data
  auto output_lines = ReadLines(FLAGS_data_dir + "/output.txt");
  std::vector<std::vector<float>> output0_data;
  for (auto line : output_lines) {
    output0_data.push_back(Split<float>(Split(line, ":")[1], " "));
  }

  for (int i = 0; i < FLAGS_warmup; i++) {
    int data_idx = i % static_cast<int>(input0_data.size());
    fill_tensor(
        predictor, 0, input0_data[data_idx].data(), input0_shapes[data_idx]);
    fill_tensor(
        predictor, 1, input1_data[data_idx].data(), input1_shapes[data_idx]);
    predictor->Run();
  }

  std::vector<std::vector<float>> results;
  double cost_time = 0;
  for (size_t i = 0; i < FLAGS_iteration; ++i) {
    int data_idx = i % static_cast<int>(input0_data.size());
    fill_tensor(
        predictor, 0, input0_data[data_idx].data(), input0_shapes[data_idx]);
    fill_tensor(
        predictor, 1, input1_data[data_idx].data(), input1_shapes[data_idx]);

    double start = GetCurrentUS();
    predictor->Run();
    cost_time += (GetCurrentUS() - start);

    auto output_tensor = predictor->GetOutput(0);
    auto output_shape = output_tensor->shape();
    auto output_data = output_tensor->data<float>();
    ASSERT_EQ(output_shape.size(), 2UL);
    int64_t out_size = 1;
    for (auto dim : output_shape) {
      out_size *= dim;
    }
    std::vector<float> ret(out_size);
    memcpy(ret.data(), output_data, sizeof(float) * out_size);
    results.push_back(ret);
  }

  for (float abs_error : {1e-1, 1e-2}) {
    float acc = CalOutAccuracy(results, output0_data, abs_error);
    LOG(INFO) << "acc: " << acc << ", if abs_error < " << abs_error;
    ASSERT_GE(CalOutAccuracy(results, output0_data, abs_error), 0.99);
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup
            << ", iteration: " << FLAGS_iteration << ", spend "
            << cost_time / FLAGS_iteration / 1000.0 << " ms in average.";
}

}  // namespace lite
}  // namespace paddle
