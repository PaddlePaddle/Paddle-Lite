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
#include "lite/utils/string.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(iteration, 1, "iteration times to run");

namespace paddle {
namespace lite {

TEST(bisenet, test_bisenet_fp32_v2_3_nnadapter) {
  std::vector<std::string> nnadapter_device_names;
  std::string nnadapter_context_properties;
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
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  nnadapter_device_names.emplace_back("huawei_ascend_npu");
  nnadapter_context_properties = "HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0";
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  nnadapter_device_names.emplace_back("intel_openvino");
// TODO(hong19860320) Fix core dump
// 1. Model (split to relu_8.tmp_0) core dump (android htp fp16)
// 2. Error is "A single op (1e7f00000017) requires 0x704800 bytes of TCM, which
// is greater than the TCM size of 0x400000!". It seems like shape is too large?
// #elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
//   nnadapter_device_names.emplace_back("qualcomm_qnn");
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

  std::string input_data_dir =
      FLAGS_data_dir + std::string("/bisenet_input.txt");
  std::string output_data_dir =
      FLAGS_data_dir + std::string("/bisenet_output.txt");
  std::vector<std::vector<std::vector<uint8_t>>> input_data_set;
  std::vector<std::vector<std::vector<int64_t>>> input_data_set_shapes;
  LoadSpecificData(input_data_dir,
                   input_data_set,
                   input_data_set_shapes,
                   predictor,
                   "input");
  std::vector<std::vector<std::vector<uint8_t>>> output_data_set;
  std::vector<std::vector<std::vector<int64_t>>> output_data_set_shapes;
  LoadSpecificData(output_data_dir,
                   output_data_set,
                   output_data_set_shapes,
                   predictor,
                   "output");

  FLAGS_warmup = 1;
  for (int i = 0; i < FLAGS_warmup; i++) {
    FillModelInput(input_data_set[i], input_data_set_shapes[i], predictor);
    predictor->Run();
  }

  double cost_time = 0;
  std::vector<std::vector<float>> results;
  for (int i = 0; i < FLAGS_iteration; i++) {
    FillModelInput(input_data_set[i], input_data_set_shapes[i], predictor);

    double start = GetCurrentUS();
    predictor->Run();
    cost_time += (GetCurrentUS() - start);

    std::vector<float> abs_error;
    GetModelOutputAndAbsError(
        predictor, output_data_set[i], output_data_set_shapes[i], abs_error);
    results.push_back(abs_error);
  }

  for (float abs_error : {1e-0, 1e-1, 1e-2}) {
    float acc = CalOutAccuracy(results, abs_error);
    LOG(INFO) << "acc: " << acc << ", if abs_error < " << abs_error;
    ASSERT_GE(acc, 0.99);
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup
            << ", iteration: " << FLAGS_iteration << ", spend "
            << cost_time / FLAGS_iteration / 1000.0 << " ms in average.";
}

}  // namespace lite
}  // namespace paddle
