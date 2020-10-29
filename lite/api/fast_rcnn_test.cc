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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {

#ifdef LITE_WITH_ARM
void TestModel(const std::vector<Place>& valid_places) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_HIGH, FLAGS_threads);
  lite::Predictor predictor;

  predictor.Build(FLAGS_model_dir,
                  FLAGS_model_dir + "/__model__",
                  FLAGS_model_dir + "/__params__",
                  valid_places);

  // set input
  auto* img = predictor.GetInput(0);
  img->Resize({1, 3, 768, 1312});
  auto* img_data = img->mutable_data<float>();
  for (int i = 0; i < img->numel(); i++) {
    img_data[i] = 20.0;
  }

  std::vector<float> src_data = {768, 1312, 0.6833333373069763};
  auto* im_info = predictor.GetInput(1);
  im_info->Resize({1, 3});
  auto* im_info_data = im_info->mutable_data<float>();
  memcpy(im_info_data, src_data.data(), src_data.size() * sizeof(float));

  auto* im_shape = predictor.GetInput(2);
  im_shape->Resize({1, 3});
  auto* im_shape_data = im_shape->mutable_data<float>();
  memcpy(im_shape_data, src_data.data(), src_data.size() * sizeof(float));

  // run
  predictor.Run();

  // check
  auto* out = predictor.GetOutput(0);
  ASSERT_EQ(out->dims().size(), 2);

  std::vector<float> results = {1.6000000e+01,
                                5.2030690e-02,
                                1.8142526e+03,
                                9.6864917e+02,
                                1.9105895e+03};
  auto* out_data = out->data<float>();
  for (int i = 0; i < results.size(); ++i) {
    EXPECT_NEAR(out_data[i], results[i], 1);
  }
}

TEST(Fast_RCNN, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  TestModel(valid_places);
}

#endif  // LITE_WITH_ARM

}  // namespace lite
}  // namespace paddle
