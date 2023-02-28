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
#include "lite/tests/api/ocr_data_utility.h"
#include "lite/tests/api/utility.h"
#include "lite/utils/string.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(iteration, 5, "iteration times to run");

namespace paddle {
namespace lite {

TEST(ch_ppocr_server_v2_0_rec,
     test_ch_ppocr_server_v2_0_rec_fp32_v2_3_nnadapter) {
  FLAGS_warmup = 1;
  bool prepare_before_timing = true;
  std::string nnadapter_subgraph_partition_config_buffer;
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
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  nnadapter_device_names.emplace_back("qualcomm_qnn");
  // 1. Not support dynamic shape
  // 2. Reduce execute time
  FLAGS_iteration = 1;
  FLAGS_warmup = 0;
  prepare_before_timing = false;
  // TODO(zhupengyang): Last matmul is not supported on htp+fp16.
  nnadapter_subgraph_partition_config_buffer =
      "transpose2:lstm_0.tmp_0:transpose_2.tmp_0,transpose_2.tmp_1\n"
      "matmul:transpose_2.tmp_0,ctc_fc_w_attr:ctc_fc.tmp_0\n"
      "elementwise_add:ctc_fc.tmp_0,ctc_fc_b_attr:ctc_fc.tmp_1\n"
      "softmax:ctc_fc.tmp_1:save_infer_model/scale_0.tmp_1";
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

  std::string raw_data_dir = FLAGS_data_dir + std::string("/raw_data");
  std::string out_data_dir =
      FLAGS_data_dir + std::string("/ch_ppocr_mobile_v2_0_out_data");
  std::string images_shape_path =
      FLAGS_data_dir + std::string("/images_shape.txt");

  auto input_lines = ReadLines(images_shape_path);
  std::vector<std::string> input_names;
  std::vector<std::vector<int64_t>> input_shapes;
  for (auto line : input_lines) {
    input_names.push_back(Split(line, ":")[0]);
    input_shapes.push_back(Split<int64_t>(Split(line, ":")[1], " "));
  }

  std::vector<std::vector<float>> raw_data;
  std::vector<std::vector<float>> gt_data;
  for (size_t i = 0; i < FLAGS_iteration; i++) {
    raw_data.push_back(
        ReadRawData(raw_data_dir, input_names[i], input_shapes[i]));
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    fill_tensor(predictor, 0, raw_data[i].data(), input_shapes[i]);
    predictor->Run();
  }

  double cost_time = 0;
  std::vector<std::vector<float>> results;
  for (size_t i = 0; i < raw_data.size(); ++i) {
    fill_tensor(predictor, 0, raw_data[i].data(), input_shapes[i]);
    if (prepare_before_timing) predictor->Run();

    double start = GetCurrentUS();
    predictor->Run();
    cost_time += (GetCurrentUS() - start);

    auto output_tensor = predictor->GetOutput(0);
    auto output_shape = output_tensor->shape();
    auto output_data = output_tensor->data<float>();
    ASSERT_EQ(output_shape.size(), 3UL);

    int64_t output_size = 1;
    for (auto dim : output_shape) {
      output_size *= dim;
    }
    std::vector<float> ret(output_size);
    memcpy(ret.data(), output_data, sizeof(float) * output_size);
    results.push_back(ret);
    gt_data.push_back(ReadRawData(out_data_dir, input_names[i], output_shape));
  }

  for (float abs_error : {1e-1, 1e-2, 1e-3, 1e-4}) {
    float acc = CalOutAccuracy(results, gt_data, abs_error);
    LOG(INFO) << "acc: " << acc << ", if abs_error < " << abs_error;
    ASSERT_GE(CalOutAccuracy(results, gt_data, abs_error), 0.99);
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup
            << ", iteration: " << FLAGS_iteration << ", spend "
            << cost_time / FLAGS_iteration / 1000.0 << " ms in average.";
}

}  // namespace lite
}  // namespace paddle
