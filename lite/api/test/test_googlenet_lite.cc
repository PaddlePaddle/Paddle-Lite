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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test/lite_api_test_helper.h"
#include "lite/api/test/test_helper.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
#ifdef LITE_WITH_X86
TEST(CXXApi, test_lite_googlenet) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  auto predictor = lite_api::CreatePaddlePredictor(config);

  auto input_tensor = predictor->GetInput(0);
  std::vector<int64_t> input_shape{1, 3, 224, 224};
  input_tensor->Resize(input_shape);
  auto* data = input_tensor->mutable_data<float>();
  int input_num = 1;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    input_num *= input_shape[i];
  }
  for (int i = 0; i < input_num; i++) {
    data[i] = 1;
  }

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
  auto out = predictor->GetOutput(0);
  std::vector<float> results(
      {0.00034298553f, 0.0008200012f, 0.0005046297f, 0.000839279f,
       0.00052616704f, 0.0003447803f, 0.0010877076f, 0.00081762316f,
       0.0003941339f,  0.0011430943f, 0.0008892841f, 0.00080191303f,
       0.0004442384f,  0.000658702f,  0.0026721435f, 0.0013686896f,
       0.0005618166f,  0.0006556497f, 0.0006984528f, 0.0014619455f});
  for (size_t i = 0; i < results.size(); ++i) {
    EXPECT_NEAR(out->data<float>()[i * 51], results[i], 1e-5);
  }
  ASSERT_EQ(out->shape().size(), 2u);
  ASSERT_EQ(out->shape()[0], 1);
  ASSERT_EQ(out->shape()[1], 1000);
}
#endif
}  // namespace lite
}  // namespace paddle
