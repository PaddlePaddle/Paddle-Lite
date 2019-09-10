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
#include <fstream>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/core/op_registry.h"

DEFINE_bool(is_run_model_optimize,
            false,
            "apply model_optimize_tool to model, use optimized model to test");

namespace paddle {
namespace lite_api {

void OutputOptModel(const std::string& load_model_dir,
                    const std::string& save_optimized_model_dir) {
  lite_api::CxxConfig config;
  config.set_model_dir(load_model_dir);
  config.set_preferred_place(Place{TARGET(kX86), PRECISION(kFloat)});
  config.set_valid_places({
      Place{TARGET(kX86), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });
  auto predictor = lite_api::CreatePaddlePredictor(config);

  int ret = system(
      paddle::lite::string_format("rm -rf %s", save_optimized_model_dir.c_str())
          .c_str());
  if (ret == 0) {
    LOG(INFO) << "delete old optimized model " << save_optimized_model_dir;
  }
  predictor->SaveOptimizedModel(save_optimized_model_dir,
                                LiteModelType::kNaiveBuffer);
  LOG(INFO) << "Load model from " << load_model_dir;
  LOG(INFO) << "Save optimized model to " << save_optimized_model_dir;
}

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
void Run(const std::string& model_dir,
         const int repeat,
         const int warmup_times,
         const int thread_num) {
  // set config and create predictor
  lite_api::MobileConfig config;
  config.set_model_dir(model_dir);
  config.set_threads(thread_num);
  if (thread_num == 1) {
    config.set_power_mode(LITE_POWER_HIGH);
  } else {
    config.set_power_mode(LITE_POWER_NO_BIND);
  }

  auto predictor = lite_api::CreatePaddlePredictor(config);

  // set input
  auto input_image = predictor->GetInput(0);
  input_image->Resize({1, 3, 300, 300});
  auto input_image_data = input_image->mutable_data<float>();
  std::ifstream read_file("/data/local/tmp/pjc/ssd_img.txt");
  if (!read_file.is_open()) {
    LOG(INFO) << "read image file fail";
    return;
  }
  auto input_shape = input_image->shape();
  int64_t input_image_size = 1;
  for (auto t : input_shape) {
    input_image_size *= t;
  }
  for (int i = 0; i < input_image_size; i++) {
    read_file >> input_image_data[i];
  }

  // warmup and run
  for (int i = 0; i < warmup_times; ++i) {
    predictor->Run();
  }

  auto start = lite::GetCurrentUS();
  for (int i = 0; i < repeat; ++i) {
    predictor->Run();
  }

  // show result
  auto end = lite::GetCurrentUS();
  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (end - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  auto out = predictor->GetOutput(0);
  auto out_data = out->data<float>();
  LOG(INFO) << "output shape:";
  auto out_shape = out->shape();
  for (auto t : out_shape) {
    LOG(INFO) << t;
  }
  LOG(INFO) << "output data:";
  int output_len = 20;
  for (int i = 0; i < output_len; i++) {
    LOG(INFO) << out_data[i];
  }
}
#endif

}  // namespace lite_api
}  // namespace paddle

TEST(Faster_RCNN, test_arm) {
  std::string save_optimized_model_dir;
  if (FLAGS_is_run_model_optimize) {
    save_optimized_model_dir = FLAGS_model_dir + "opt";
    paddle::lite_api::OutputOptModel(FLAGS_model_dir, save_optimized_model_dir);
  }
  std::string run_model_dir =
      FLAGS_is_run_model_optimize ? save_optimized_model_dir : FLAGS_model_dir;
  paddle::lite_api::Run(
      run_model_dir, FLAGS_repeats, FLAGS_threads, FLAGS_warmup);
}
