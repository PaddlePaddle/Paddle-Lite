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

TEST(Step_rnn, test_step_rnn_lite_x86) {
  std::string model_dir = FLAGS_model_dir;
  lite_api::CxxConfig config;
  config.set_model_dir(model_dir);
#ifdef LITE_WITH_X86
  config.set_x86_math_num_threads(1);
#endif
  config.set_valid_places({lite_api::Place{TARGET(kX86), PRECISION(kInt64)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  auto predictor = lite_api::CreatePaddlePredictor(config);

  std::vector<std::string> target_names = {"item_type_id",
                                           "mthid_id",
                                           "source_id_id",
                                           "layout_id",
                                           "mark_id",
                                           "category_id",
                                           "subcategory_id",
                                           "score_segment_id",
                                           "item_attention_id",
                                           "queue_num_id",
                                           "micro_video_id",
                                           "vertical_type_id"};

  for (size_t i = 0; i < target_names.size(); ++i) {
    auto input_tensor = predictor->GetInput(i);
    int size = 0;
    if (i == 6 || i == 8) {
      input_tensor->Resize(std::vector<int64_t>{5, 1});
      input_tensor->SetLoD({{0, 5}});
      size = 5;
    } else {
      input_tensor->Resize(std::vector<int64_t>{1, 1});
      input_tensor->SetLoD({{0, 1}});
      size = 1;
    }
    auto* data = input_tensor->mutable_data<int64_t>();
    for (int i = 0; i < size; i++) data[i] = 1;
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }

  LOG(INFO) << "warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  std::vector<std::vector<float>> results;
  // i = 1
  results.emplace_back(std::vector<float>({0.5030127f, 0.496987f}));
  auto out = predictor->GetOutput(0);

  std::vector<int64_t> out_shape = out->shape();

  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = 0; j < results[i].size(); ++j) {
      EXPECT_NEAR(
          out->data<float>()[j + (out_shape[1] * i)], results[i][j], 1e-6);
    }
  }
}

}  // namespace lite
}  // namespace paddle
