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
#include "lite/tests/api/ILSVRC2012_utility.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(iteration, 100, "iteration times to run");
DEFINE_int32(batch, 1, "batch of image");
DEFINE_int32(channel, 3, "image channel");
static const int32_t inceptionv3_model_input_width = 299;
static const int32_t inceptionv3_model_input_height = 299;

namespace paddle {
namespace lite {

TEST(InceptionV3, test_inception_v3_fp32_v2_0_nnadapter) {
  std::vector<std::string> nnadapter_device_names;
  std::string nnadapter_context_properties;
  std::vector<paddle::lite_api::Place> valid_places;
  float out_accuracy_threshold = 1.0f;
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
  out_accuracy_threshold = 0.74f;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  nnadapter_device_names.emplace_back("intel_openvino");
  out_accuracy_threshold = 0.73f;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  nnadapter_device_names.emplace_back("huawei_kirin_npu");
  out_accuracy_threshold = 0.73f;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  nnadapter_device_names.emplace_back("qualcomm_qnn");
  FLAGS_iteration = 3;
  out_accuracy_threshold = 0.3f;
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

  std::string raw_data_dir = FLAGS_data_dir + std::string("/raw_data_299_299");
  std::vector<int> input_shape{FLAGS_batch,
                               FLAGS_channel,
                               inceptionv3_model_input_width,
                               inceptionv3_model_input_height};
  auto raw_data = ReadRawData(raw_data_dir, input_shape, FLAGS_iteration);

  int input_size = 1;
  for (auto i : input_shape) {
    input_size *= i;
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    auto input_tensor = predictor->GetInput(0);
    input_tensor->Resize(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()));
    auto* data = input_tensor->mutable_data<float>();
    for (int j = 0; j < input_size; j++) {
      data[j] = 0.f;
    }
    predictor->Run();
  }

  std::vector<std::vector<float>> out_rets;
  out_rets.resize(FLAGS_iteration);
  double cost_time = 0;
  for (size_t i = 0; i < raw_data.size(); ++i) {
    auto input_tensor = predictor->GetInput(0);
    input_tensor->Resize(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()));
    auto* data = input_tensor->mutable_data<float>();
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
  ASSERT_GE(out_accuracy, out_accuracy_threshold);
}

}  // namespace lite
}  // namespace paddle
