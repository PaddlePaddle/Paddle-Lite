// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/tests/api/COCO2017_utility.h"
#include "lite/tests/api/utility.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(iteration, 10, "iteration times to run");
DEFINE_int32(batch, 1, "batch of image");
DEFINE_int32(channel, 3, "image channel");
DEFINE_int32(height, 640, "image height");
DEFINE_int32(width, 640, "image width");

namespace paddle {
namespace lite {

TEST(ppyolo2_r50vd_dcn, test_ppyolo2_r50vd_dcn_coco_fp32_v2_3_nnadapter) {
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
// TODO(hong19860320) Fix timeout
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

  std::string raw_data_dir = FLAGS_data_dir + std::string("/raw_data");
  std::vector<int> input_shape{
      FLAGS_batch, FLAGS_channel, FLAGS_height, FLAGS_width};
  auto raw_data = ReadRawData(raw_data_dir, input_shape, FLAGS_iteration);

  int input_size = 1;
  for (auto i : input_shape) {
    input_size *= i;
  }

  FLAGS_warmup = 1;
  for (int i = 0; i < FLAGS_warmup; ++i) {
    SetDetectionInput(predictor, input_shape, std::vector<float>(), input_size);
    predictor->Run();
  }

  std::vector<std::vector<float>> out_rets;
  out_rets.resize(FLAGS_iteration);
  double cost_time = 0;
  for (size_t i = 0; i < raw_data.size(); ++i) {
    SetDetectionInput(predictor, input_shape, raw_data[i], input_size);

    double start = GetCurrentUS();
    predictor->Run();
    cost_time += (GetCurrentUS() - start);

    auto output_tensor = predictor->GetOutput(0);
    CHECK(output_tensor);
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", batch: " << FLAGS_batch
            << ", iteration: " << FLAGS_iteration << ", spend "
            << cost_time / FLAGS_iteration / 1000.0 << " ms in average.";
}

}  // namespace lite
}  // namespace paddle
