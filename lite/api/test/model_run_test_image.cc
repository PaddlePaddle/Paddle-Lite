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

namespace paddle {
namespace lite {

TEST(model, test) {
#ifdef LITE_WITH_ARM
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_NO_BIND, FLAGS_threads);
  lite::Predictor predictor;
  std::vector<Place> valid_places({Place{TARGET(kARM), PRECISION(kFloat)},
                                   Place{TARGET(kARM), PRECISION(kInt8)}});

  auto precision = PRECISION(kFloat);
  if (FLAGS_int8) {
    precision = PRECISION(kInt8);
  }
  predictor.Build(FLAGS_model_dir, "", "", valid_places);
  int im_width = FLAGS_im_width;
  int im_height = FLAGS_im_height;
  auto* input_tensor = predictor.GetInput(0);
  auto in_dims = input_tensor->dims();
  input_tensor->Resize(
      DDim(std::vector<DDim::value_type>({1, 3, im_width, im_height})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor.Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor.Run();
  }
  auto output_tensors = predictor.GetOutputs();

  LOG(INFO) << "======output:========";
  for (auto* t : output_tensors) {
    LOG(INFO) << *t;
  }
  LOG(INFO)
      << "=====RUN_finished!!============= Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";
#endif
}

}  // namespace lite
}  // namespace paddle
