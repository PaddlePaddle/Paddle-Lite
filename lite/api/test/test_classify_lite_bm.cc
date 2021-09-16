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
#include <thread>  //NOLINT
#include <vector>
#include "lite/api/cxx_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test/test_helper.h"
#include "lite/core/op_registry.h"

DEFINE_string(input_img_txt_path,
              "",
              "if set input_img_txt_path, read the img filename as input.");

namespace paddle {
namespace lite {

const int g_batch_size = 1;
const int g_thread_num = 1;

void instance_run() {
  lite::Predictor predictor;
  std::vector<std::string> passes;
  std::vector<Place> valid_places({Place{TARGET(kBM), PRECISION(kFloat)},
                                   Place{TARGET(kX86), PRECISION(kFloat)}});
  predictor.Build(FLAGS_model_dir,
                  FLAGS_model_dir + "/model",
                  FLAGS_model_dir + "/params",
                  valid_places,
                  passes);
  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>(
      {g_batch_size, 3, FLAGS_im_height, FLAGS_im_width})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  if (FLAGS_input_img_txt_path.empty()) {
    for (int i = 0; i < item_size; i++) {
      data[i] = 1;
    }
  } else {
    for (int j = 0; j < g_batch_size; j++) {
      std::fstream fs(FLAGS_input_img_txt_path, std::ios::in);
      if (!fs.is_open()) {
        LOG(FATAL) << "open input_img_txt error.";
      }
      for (int i = 0; i < item_size / g_batch_size; i++) {
        fs >> data[i];
      }
      data += item_size / g_batch_size;
    }
  }
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

  auto out = predictor.GetOutputs();
  FILE* fp = fopen("result.txt", "wb");
  for (int i = 0; i < out.size(); i++) {
    auto* out_data = out[i]->data<float>();
    LOG(INFO) << out[i]->numel();
    for (int j = 0; j < out[i]->numel(); j++) {
      fprintf(fp, "%f\n", out_data[j]);
    }
  }
  fclose(fp);
}

void TestModel(const std::vector<Place>& valid_places) {
  std::vector<std::unique_ptr<std::thread>> instances_vec;
  for (int i = 0; i < g_thread_num; ++i) {
    instances_vec.emplace_back(new std::thread(&instance_run));
  }
  for (int i = 0; i < g_thread_num; ++i) {
    instances_vec[i]->join();
  }
}

TEST(Classify, test_bm) {
  std::vector<Place> valid_places({Place{TARGET(kBM), PRECISION(kFloat)},
                                   Place{TARGET(kX86), PRECISION(kFloat)}});

  TestModel(valid_places);
}

}  // namespace lite
}  // namespace paddle
