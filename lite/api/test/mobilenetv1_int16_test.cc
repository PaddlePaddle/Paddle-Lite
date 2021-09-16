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
#include "lite/api/cxx_api.h"
#include "lite/api/light_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test/test_helper.h"
#include "lite/core/op_registry.h"

DEFINE_string(optimized_model,
              "/data/local/tmp/int16_model",
              "optimized_model");
DEFINE_int32(N, 1, "input_batch");
DEFINE_int32(C, 3, "input_channel");
DEFINE_int32(H, 224, "input_height");
DEFINE_int32(W, 224, "input_width");

namespace paddle {
namespace lite {

void TestModel(const std::vector<Place>& valid_places,
               const std::string& model_dir) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_NO_BIND, FLAGS_threads);

  LOG(INFO) << "Optimize model.";
  lite::Predictor cxx_predictor;
  cxx_predictor.Build(model_dir, "", "", valid_places);
  cxx_predictor.SaveModel(FLAGS_optimized_model,
                          paddle::lite_api::LiteModelType::kNaiveBuffer);

  LOG(INFO) << "Load optimized model.";
  lite::LightPredictor predictor(FLAGS_optimized_model + ".nb", false);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(
      std::vector<DDim::value_type>({FLAGS_N, FLAGS_C, FLAGS_H, FLAGS_W})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = FLAGS_N * FLAGS_C * FLAGS_H * FLAGS_W;
  for (int i = 0; i < item_size; i++) {
    data[i] = 1.;
  }

  LOG(INFO) << "Predictor run.";
  predictor.Run();

  auto* out = predictor.GetOutput(0);
  const auto* pdata = out->data<float>();

  std::vector<float> ref = {
      0.000191383, 0.000592063, 0.000112282, 6.27426e-05, 0.000127522};
  double eps = 1e-5;
  for (int i = 0; i < ref.size(); ++i) {
    EXPECT_NEAR(pdata[i], ref[i], eps);
  }
}

TEST(MobileNetV1_Int16, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kARM), PRECISION(kFloat)},
  });
  std::string model_dir = FLAGS_model_dir;
  TestModel(valid_places, model_dir);
}

}  // namespace lite
}  // namespace paddle
