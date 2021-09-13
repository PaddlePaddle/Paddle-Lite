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
#include "lite/api/test/test_helper.h"
#include "lite/core/op_registry.h"

DEFINE_int32(N, 1, "input_batch");
DEFINE_int32(C, 3, "input_channel");
DEFINE_int32(H, 224, "input_height");
DEFINE_int32(W, 224, "input_width");

namespace paddle {
namespace lite {

void TestModel(const std::string& model_dir,
               const std::vector<float>& ref,
               float eps,
               lite_api::QuantType quant_type) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_NO_BIND, FLAGS_threads);

  LOG(INFO) << "Load fp32 model from " << model_dir;
  lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(model_dir);
  std::vector<Place> vaild_places = {
      Place{TARGET(kARM), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kInt32)},
      Place{TARGET(kARM), PRECISION(kInt64)},
  };
  cxx_config.set_valid_places(vaild_places);
  cxx_config.set_quant_model(true);
  cxx_config.set_quant_type(quant_type);
  auto cxx_predictor = lite_api::CreatePaddlePredictor(cxx_config);

  LOG(INFO) << "Save quantized model";
  std::string opt_model_path;
  if (quant_type == lite_api::QuantType::QUANT_INT8) {
    opt_model_path = model_dir + "/mobilenetv1_opt_quant_int8";
  } else if (quant_type == lite_api::QuantType::QUANT_INT16) {
    opt_model_path = model_dir + "/mobilenetv1_opt_quant_int16";
  }
  cxx_predictor->SaveOptimizedModel(opt_model_path,
                                    lite_api::LiteModelType::kNaiveBuffer);

  LOG(INFO) << "Load optimized model";
  lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(opt_model_path + ".nb");
  auto mobile_predictor = lite_api::CreatePaddlePredictor(mobile_config);

  auto input_tensor = mobile_predictor->GetInput(0);
  input_tensor->Resize({FLAGS_N, FLAGS_C, FLAGS_H, FLAGS_W});
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = FLAGS_N * FLAGS_C * FLAGS_H * FLAGS_W;
  for (int i = 0; i < item_size; i++) {
    data[i] = 1.;
  }

  LOG(INFO) << "Predictor run.";
  mobile_predictor->Run();

  auto out = mobile_predictor->GetOutput(0);
  const auto* pdata = out->data<float>();

  for (int i = 0; i < ref.size(); ++i) {
    LOG(INFO) << "predict:" << pdata[i];
    LOG(INFO) << "gt:" << ref[i];
    EXPECT_NEAR(pdata[i], ref[i], eps);
  }
}

TEST(bobileetv1_opt_quant_int16, test_arm) {
  std::vector<float> ref = {
      0.000191383, 0.000592063, 0.000112282, 6.27426e-05, 0.000127522};
  float eps = 1e-5;
  TestModel(
      FLAGS_model_dir, ref, eps, paddle::lite_api::QuantType::QUANT_INT16);
}

TEST(mobilenetv1_opt_quant_int8, test_arm) {
  std::vector<float> ref = {
      0.0002320, 0.0006248689, 0.000112282, 6.27426e-05, 0.0001111296};
  float eps = 3e-5;
  TestModel(FLAGS_model_dir, ref, eps, paddle::lite_api::QuantType::QUANT_INT8);
}

}  // namespace lite
}  // namespace paddle
