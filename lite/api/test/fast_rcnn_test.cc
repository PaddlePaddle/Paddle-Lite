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

namespace paddle {
namespace lite {

void TestModel(const std::string& model_dir) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_NO_BIND, FLAGS_threads);

  LOG(INFO) << "Load fp32 model from " << model_dir;
  lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(model_dir);
  cxx_config.set_model_file(model_dir + "/__model__");
  cxx_config.set_param_file(model_dir + "/__params__");
  std::vector<Place> vaild_places = {
      Place{TARGET(kARM), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kInt32)},
      Place{TARGET(kARM), PRECISION(kInt64)},
  };
  cxx_config.set_valid_places(vaild_places);
  auto cxx_predictor = lite_api::CreatePaddlePredictor(cxx_config);

  std::string opt_model_path = model_dir + "/fast_rcnn_opt";
  LOG(INFO) << "Save quantized model to " << opt_model_path;
  cxx_predictor->SaveOptimizedModel(opt_model_path,
                                    lite_api::LiteModelType::kNaiveBuffer);

  LOG(INFO) << "Load optimized model";
  lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(opt_model_path + ".nb");
  auto mobile_predictor = lite_api::CreatePaddlePredictor(mobile_config);

  // set input
  int n = 1;
  int c = 3;
  int h = 768;
  int w = 1312;
  auto img = mobile_predictor->GetInput(0);
  img->Resize({n, c, h, w});
  auto img_data = img->mutable_data<float>();
  for (int i = 0; i < n * c * h * w; i++) {
    img_data[i] = 20.0;
  }

  std::vector<float> src_data = {768, 1312, 0.6833333373069763};
  auto im_info = mobile_predictor->GetInput(1);
  im_info->Resize({1, 3});
  auto* im_info_data = im_info->mutable_data<float>();
  memcpy(im_info_data, src_data.data(), src_data.size() * sizeof(float));

  auto im_shape = mobile_predictor->GetInput(2);
  im_shape->Resize({1, 3});
  auto im_shape_data = im_shape->mutable_data<float>();
  memcpy(im_shape_data, src_data.data(), src_data.size() * sizeof(float));

  // run
  mobile_predictor->Run();

  // check
  auto out = mobile_predictor->GetOutput(0);

  std::vector<float> results = {1.6000000e+01,
                                5.2030690e-02,
                                1.8142526e+03,
                                9.6864917e+02,
                                1.9105895e+03};
  auto out_data = out->data<float>();
  for (int i = 0; i < results.size(); ++i) {
    EXPECT_NEAR(out_data[i], results[i], 1);
    LOG(INFO) << "predict:" << out_data[i] << ", gt:" << results[i];
  }
}

TEST(test_fast_rcnn, test_arm) { TestModel(FLAGS_model_dir); }

}  // namespace lite
}  // namespace paddle
