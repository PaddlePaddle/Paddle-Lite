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
#include "lite/api/lite_api_test_helper.h"
#include "lite/api/paddle_api.h"
#include "lite/api/test_helper.h"
#include "lite/utils/cp_logging.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(test_img_num, 10000, "0 < test_img_num <= 10000");

namespace paddle {
namespace lite {

TEST(LAC_MODEL, test_lac_crf_fp32_arm) {
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  // Use the full api with CxxConfig to generate the optimized model
  lite_api::CxxConfig cxx_config;
  cxx_config.set_model_dir(FLAGS_model_dir);
  cxx_config.set_valid_places(
      {lite_api::Place{TARGET(kARM), PRECISION(kFloat)}});
  predictor = lite_api::CreatePaddlePredictor(cxx_config);
  predictor->SaveOptimizedModel(FLAGS_model_dir,
                                paddle::lite_api::LiteModelType::kNaiveBuffer);

  // Use the light api with MobileConfig to load and run the optimized model
  paddle::lite_api::MobileConfig mobile_config;
  mobile_config.set_model_from_file(FLAGS_model_dir + ".nb");
  mobile_config.set_threads(FLAGS_threads);
  mobile_config.set_power_mode(
      static_cast<lite_api::PowerMode>(FLAGS_power_mode));
  predictor = paddle::lite_api::CreatePaddlePredictor(mobile_config);

  // prepare
  std::ifstream fs(FLAGS_data_dir);
  ASSERT_TRUE(fs.is_open()) << "open input file " << FLAGS_data_dir
                            << " error.";
  ASSERT_TRUE(FLAGS_test_img_num > 0 && FLAGS_test_img_num <= 10000);

  for (int i = 0; i < FLAGS_test_img_num; i++) {
    if (i % 100 == 0) {
      LOG(INFO) << "Iter:" << i;
    }

    // set input
    int64_t input_h, input_w;
    fs >> input_h >> input_w;

    uint64_t lod_data_0, lod_data_1;
    fs >> lod_data_0 >> lod_data_1;
    std::vector<std::vector<uint64_t>> lod;
    lod.push_back({lod_data_0, lod_data_1});

    auto input_tensor = predictor->GetInput(0);
    input_tensor->Resize({input_h, input_w});
    input_tensor->SetLoD(lod);

    auto input_data = input_tensor->mutable_data<int64_t>();
    for (int i = 0; i < input_h * input_w; i++) {
      fs >> input_data[i];
    }

    // run
    predictor->Run();

    // get output
    auto out = predictor->GetOutput(0);
    std::vector<int64_t> out_shape = out->shape();
    int64_t out_num = ShapeProduction(out_shape);
    auto* out_data = out->data<int64_t>();

    int64_t gt_h, gt_w;
    fs >> gt_h >> gt_w;
    int gt_num = gt_h * gt_w;
    std::vector<int64_t> gt_data(gt_h * gt_w, 0);
    for (int i = 0; i < gt_num; i++) {
      fs >> gt_data[i];
    }

    // check
    ASSERT_EQ(out_shape.size(), 2);
    ASSERT_EQ(out_shape[0], gt_h);
    ASSERT_EQ(out_shape[1], gt_w);
    for (int i = 0; i < gt_num; i++) {
      ASSERT_EQ(out_data[i], gt_data[i]);
    }
  }
  fs.close();
}

}  // namespace lite
}  // namespace paddle
