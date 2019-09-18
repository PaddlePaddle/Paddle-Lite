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

DEFINE_string(optimized_model, "", "optimized_model");

namespace paddle {
namespace lite {

void TestModel(const std::vector<Place>& valid_places,
               const Place& preferred_place,
               const std::string& model_dir = FLAGS_model_dir,
               bool save_model = false) {
  LOG(INFO) << "debug herer";
  DeviceInfo::Init();
  LOG(INFO) << "debug herer";
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_HIGH, FLAGS_threads);
  LOG(INFO) << "debug herer";
  lite::Predictor predictor;

  LOG(INFO) << "debug herer";
  predictor.Build(model_dir, "", "", preferred_place, valid_places);
  LOG(INFO) << "debug herer";
  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 1, 1, 100})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = i;
  }
  LOG(INFO) << "debug herer";

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor.Run();
  }
  LOG(INFO) << "debug herer";

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor.Run();
  }

  if (save_model) {
    LOG(INFO) << "Save optimized model to " << FLAGS_optimized_model;
    predictor.SaveModel(FLAGS_optimized_model);
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  std::vector<std::vector<float>> ref;
  ref.emplace_back(std::vector<float>(
      {0.00019130898, 9.467885e-05,  0.00015971427, 0.0003650665,
       0.00026431272, 0.00060884043, 0.0002107942,  0.0015819625,
       0.0010323516,  0.00010079765, 0.00011006987, 0.0017364529,
       0.0048292773,  0.0013995157,  0.0018453331,  0.0002428986,
       0.00020211363, 0.00013668182, 0.0005855956,  0.00025901722}));
  auto* out = predictor.GetOutput(0);
  const auto* pdata = out->data<float>();
  int step = 50;
#ifdef LITE_WITH_NPU
  ASSERT_EQ(out->dims().production(), 1000);
  double eps = 0.1;
  for (int i = 0; i < ref.size(); ++i) {
    for (int j = 0; j < ref[i].size(); ++j) {
      auto result = pdata[j * step + (out->dims()[1] * i)];
      auto diff = std::fabs((result - ref[i][j]) / ref[i][j]);
      VLOG(3) << diff;
      EXPECT_LT(diff, eps);
    }
  }
#else
  ASSERT_EQ(out->dims().size(), 2);
  ASSERT_EQ(out->dims()[0], 1);
  ASSERT_EQ(out->dims()[1], 500);
  double eps = 1e-6;
#if 0
  for (int i = 0; i < ref.size(); ++i) {
    for (int j = 0; j < ref[i].size(); ++j) {
      auto result = pdata[j * step + (out->dims()[1] * i)];
      EXPECT_NEAR(result, ref[i][j], eps);
    }
  }
#endif
#endif
}

#ifdef LITE_WITH_NPU
TEST(MobileNetV1, test_npu) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
      Place{TARGET(kNPU), PRECISION(kFloat)},
  });

  TestModel(valid_places,
            Place({TARGET(kARM), PRECISION(kFloat)}),
            FLAGS_model_dir,
            true /* save_model*/);

  TestModel(valid_places,
            Place({TARGET(kARM), PRECISION(kFloat)}),
            FLAGS_optimized_model,
            false /* save model */);
}
#endif  // LITE_WITH_NPU

#if 1
TEST(MobileNetV1, test_arm) {
  LOG(INFO) << "2019年09月18日09:31:53";
  LOG(INFO) << "debug herer";
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kAny)},

      Place{TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNHWC)},
      Place{TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kNHWC)},
      Place{TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kAny)},
  });

  LOG(INFO) << "debug herer";
  TestModel(valid_places,
            Place({TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kNCHW)}));
  // TestModel(valid_places, Place({TARGET(kARM), PRECISION(kFloat),
  // DATALAYOUT(kNHWC)}));
}
#endif

#ifdef LITE_WITH_OPENCL
TEST(MobileNetV1, test_opencl) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNHWC)},
      Place{TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kNHWC)},
      Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNHWC)},
  });

  TestModel(valid_places,
            Place({TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNHWC)}));
}
#endif  // LITE_WITH_OPENCL

}  // namespace lite
}  // namespace paddle
