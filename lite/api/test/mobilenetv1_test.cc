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
#include "lite/api/test/test_helper.h"
#include "lite/core/op_registry.h"

DEFINE_string(optimized_model, "", "optimized_model");
DEFINE_int32(N, 1, "input_batch");
DEFINE_int32(C, 3, "input_channel");
DEFINE_int32(H, 224, "input_height");
DEFINE_int32(W, 224, "input_width");

namespace paddle {
namespace lite {

void TestModel(const std::vector<Place>& valid_places,
               const std::string& model_dir = FLAGS_model_dir,
               bool save_model = false) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_NO_BIND, FLAGS_threads);
  lite::Predictor predictor;

  predictor.Build(model_dir, "", "", valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(
      std::vector<DDim::value_type>({FLAGS_N, FLAGS_C, FLAGS_H, FLAGS_W})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor.Run();
  }

  double sum_duration = 0.0;  // millisecond;
  for (int i = 0; i < FLAGS_repeats; ++i) {
    auto start = GetCurrentUS();
    predictor.Run();
    auto duration = (GetCurrentUS() - start) / 1000.0;
    sum_duration += duration;
    VLOG(1) << "run_idx:" << i << " " << duration << " ms";
  }

  if (save_model) {
    LOG(INFO) << "Save optimized model to " << FLAGS_optimized_model;
    predictor.SaveModel(FLAGS_optimized_model);
  }

  LOG(INFO) << "input shape(NCHW):" << FLAGS_N << " " << FLAGS_C << " "
            << FLAGS_H << " " << FLAGS_W;
  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << sum_duration / FLAGS_repeats << " ms in average.";

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

  // Get target and check result
  VLOG(1) << "valid_places.size():" << valid_places.size();
  for (int i = 0; i < valid_places.size(); ++i) {
    auto p = valid_places[i];
    VLOG(1) << "valid_places[" << i << "]:" << p.DebugString();
  }
  auto first_target = valid_places[0].target;

  float relative_err_max = 0.f;
  if (first_target == TARGET(kOpenCL) || first_target == TARGET(kNPU)) {
    ASSERT_EQ(out->dims().production(), 1000);
    double eps = first_target == TARGET(kOpenCL) ? 0.13 : 0.1;
    for (int i = 0; i < ref.size(); ++i) {
      for (int j = 0; j < ref[i].size(); ++j) {
        auto idx = j * step + (out->dims()[1] * i);
        auto result = pdata[idx];
        auto relative_err = std::fabs((result - ref[i][j]) / ref[i][j]);
        VLOG(3) << lite::string_format(
            "relative_err[%d]: %f \tresult: %f \tref: %f",
            idx,
            relative_err,
            result,
            ref[i][j]);
        if (relative_err > relative_err_max) {
          relative_err_max = relative_err;
        }
      }
    }
    VLOG(3) << lite::string_format("max relative err: %f", relative_err_max);
    EXPECT_LT(relative_err_max, eps);
  } else {
    ASSERT_EQ(out->dims().size(), 2);
    ASSERT_EQ(out->dims()[0], 1);
    ASSERT_EQ(out->dims()[1], 1000);
    double eps = 1e-6;
    for (int i = 0; i < ref.size(); ++i) {
      for (int j = 0; j < ref[i].size(); ++j) {
        auto result = pdata[j * step + (out->dims()[1] * i)];
        EXPECT_NEAR(result, ref[i][j], eps);
      }
    }
  }

  // Get detailed result
  size_t output_tensor_num = predictor.GetOutputNames().size();
  VLOG(1) << "output tensor num:" << output_tensor_num;

  for (size_t tidx = 0; tidx < output_tensor_num; ++tidx) {
    auto* output_tensor = predictor.GetOutput(tidx);
    VLOG(1) << "============= output tensor " << tidx << " =============\n";
    auto out_dims = output_tensor->dims();
    auto out_data = output_tensor->data<float>();
    auto out_mean = compute_mean<float>(out_data, out_dims.production());
    auto out_std_dev = compute_standard_deviation<float>(
        out_data, out_dims.production(), true, out_mean);

    VLOG(1) << "output tensor dims:" << out_dims;
    VLOG(1) << "output tensor elements num:" << out_dims.production();
    VLOG(1) << "output tensor standard deviation:" << out_std_dev;
    VLOG(1) << "output tensor mean value:" << out_mean;

    // print result
    for (int i = 0; i < out_dims.production(); ++i) {
      VLOG(2) << "output_tensor->data<float>()[" << i
              << "]:" << output_tensor->data<float>()[i];
    }
  }
}

#ifdef LITE_WITH_NPU
TEST(MobileNetV1, test_npu) {
  std::vector<Place> valid_places({
      Place{TARGET(kARM), PRECISION(kFloat)},
      Place{TARGET(kNPU), PRECISION(kFloat)},
  });

  TestModel(valid_places, FLAGS_model_dir, true /* save_model*/);

  TestModel(valid_places, FLAGS_optimized_model, false /* save model */);
}
#endif  // LITE_WITH_NPU

TEST(MobileNetV1, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  TestModel(valid_places);
}

#ifdef LITE_WITH_OPENCL
TEST(MobileNetV1, test_opencl) {
  std::vector<Place> valid_places({
      Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)},
      Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)},
      Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)},
      TARGET(kARM),  // enable kARM CPU kernel when no opencl kernel
  });

  TestModel(valid_places);
}
#endif  // LITE_WITH_OPENCL

}  // namespace lite
}  // namespace paddle
