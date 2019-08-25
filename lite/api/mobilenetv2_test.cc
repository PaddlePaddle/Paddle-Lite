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

#ifdef LITE_WITH_ARM
void TestModel(const std::vector<Place>& valid_places,
               const Place& preferred_place,
               const std::string& model_dir = FLAGS_model_dir,
               bool gen_npu = false,
               bool save_model = false) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_HIGH, FLAGS_threads);
  lite::Predictor predictor;

  predictor.Build(model_dir, preferred_place, valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 224, 224})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }

  if (gen_npu) {
    predictor.GenNPURuntimeProgram();
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor.Run();
  }

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
  // i = 1
  ref.emplace_back(std::vector<float>(
      {0.00017082224, 5.699624e-05,  0.000260885,   0.00016412718,
       0.00034818667, 0.00015230637, 0.00032959113, 0.0014772735,
       0.0009059976,  9.5378724e-05, 5.386537e-05,  0.0006427285,
       0.0070957416,  0.0016094646,  0.0018807327,  0.00010506048,
       6.823785e-05,  0.00012269315, 0.0007806194,  0.00022354358}));
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
  ASSERT_EQ(out->dims()[1], 1000);
  for (int i = 0; i < ref.size(); ++i) {
    for (int j = 0; j < ref[i].size(); ++j) {
      EXPECT_NEAR(pdata[j * step + (out->dims()[1] * i)], ref[i][j], 1e-6);
    }
  }
#endif
}

#ifdef LITE_WITH_NPU
TEST(MobileNetV2, test_npu) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
      Place{TARGET(kNPU), PRECISION(kFloat)},
  });

  TestModel(valid_places,
            Place({TARGET(kARM), PRECISION(kFloat)}),
            FLAGS_model_dir,
            true /* gen_npu */,
            true /* save_model*/);

  TestModel(valid_places,
            Place({TARGET(kARM), PRECISION(kFloat)}),
            FLAGS_optimized_model,
            false /* gen_npu */,
            false /* save model */);
}
#endif  // LITE_WITH_NPU

TEST(MobileNetV2, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  TestModel(valid_places, Place({TARGET(kARM), PRECISION(kFloat)}));
}

#ifdef LITE_WITH_OPENCL
TEST(MobileNetV2, test_opencl) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
      Place{TARGET(kOpenCL), PRECISION(kFloat)},
  });

  TestModel(valid_places, Place({TARGET(kOpenCL), PRECISION(kFloat)}));
}
#endif  // LITE_WITH_OPENCL

#endif  // LITE_WITH_ARM

}  // namespace lite
}  // namespace paddle
