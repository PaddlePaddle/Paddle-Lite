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
#include "lite/api/cxx_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {

#ifdef LITE_WITH_ARM
void TestModel(const std::vector<Place>& valid_places,
               const Place& preferred_place) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_HIGH, FLAGS_threads);
  lite::Predictor predictor;

  predictor.Build(FLAGS_model_dir, "", "", preferred_place, valid_places);

  auto* input_image = predictor.GetInput(0);
  input_image->Resize({1, 3, 1333, 800});
  auto* input_image_data = input_image->mutable_data<float>();
  std::ifstream read_file("/data/local/tmp/pjc/faster_rcnn_img.txt");
  for (int i = 0; i < input_image->numel(); i++) {
    read_file >> input_image_data[i];
  }
  read_file.close();
  LOG(INFO) << "image data:" << input_image_data[0] << " "
            << input_image_data[input_image->numel() - 1];

  auto* im_info = predictor.GetInput(1);
  im_info->Resize({1, 3});
  auto* im_info_data = im_info->mutable_data<float>();
  im_info_data[0] = 1333;
  im_info_data[1] = 800;
  im_info_data[2] = 1;

  auto* im_shape = predictor.GetInput(2);
  im_shape->Resize({1, 3});
  auto* im_shape_data = im_shape->mutable_data<float>();
  im_shape_data[0] = 1333;
  im_shape_data[1] = 800;
  im_shape_data[2] = 1;

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor.Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor.Run();
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  auto* out = predictor.GetOutput(0);
  auto* out_data = out->data<float>();
  LOG(INFO) << "==========output data===============";
  LOG(INFO) << out->dims();
  for (int i = 0; i < out->numel(); i++) {
    LOG(INFO) << out_data[i];
  }
}

TEST(Faster_RCNN, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  TestModel(valid_places, Place({TARGET(kARM), PRECISION(kFloat)}));
}

#endif  // LITE_WITH_ARM

}  // namespace lite
}  // namespace paddle
