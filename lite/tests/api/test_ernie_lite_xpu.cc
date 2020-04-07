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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

template <typename T>
lite::Tensor GetTensorWithShape(std::vector<int64_t> shape) {
  lite::Tensor ret;
  ret.Resize(shape);
  T* ptr = ret.mutable_data<T>();
  for (int i = 0; i < ret.numel(); ++i) {
    ptr[i] = (T)1;
  }
  return ret;
}

TEST(Ernie, test_ernie_lite_xpu) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  config.set_xpu_workspace_l3_size_per_thread();
  auto predictor = lite_api::CreatePaddlePredictor(config);

  int64_t batch_size = 1;
  int64_t seq_len = 64;
  Tensor sample_input = GetTensorWithShape<int64_t>({batch_size, seq_len, 1});
  std::vector<int64_t> input_shape{batch_size, seq_len, 1};
  predictor->GetInput(0)->Resize(input_shape);
  predictor->GetInput(1)->Resize(input_shape);
  predictor->GetInput(2)->Resize(input_shape);
  predictor->GetInput(3)->Resize(input_shape);

  memcpy(predictor->GetInput(0)->mutable_data<int64_t>(),
         sample_input.raw_data(),
         sizeof(int64_t) * batch_size * seq_len);
  memcpy(predictor->GetInput(1)->mutable_data<int64_t>(),
         sample_input.raw_data(),
         sizeof(int64_t) * batch_size * seq_len);
  memcpy(predictor->GetInput(2)->mutable_data<int64_t>(),
         sample_input.raw_data(),
         sizeof(int64_t) * batch_size * seq_len);
  memcpy(predictor->GetInput(3)->mutable_data<int64_t>(),
         sample_input.raw_data(),
         sizeof(int64_t) * batch_size * seq_len);

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  std::vector<std::vector<float>> results;
  results.emplace_back(std::vector<float>({0.108398}));
  auto out = predictor->GetOutput(0);
  ASSERT_EQ(out->shape().size(), 2);
  ASSERT_EQ(out->shape()[0], 1);
  ASSERT_EQ(out->shape()[1], 1);

  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = 0; j < results[i].size(); ++j) {
      EXPECT_NEAR(
          out->data<float>()[j + (out->shape()[1] * i)], results[i][j], 1e-5);
    }
  }
}

}  // namespace lite
}  // namespace paddle
