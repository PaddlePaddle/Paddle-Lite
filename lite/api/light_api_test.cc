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

#include "lite/api/light_api.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"

DEFINE_string(optimized_model, "", "");

namespace paddle {
namespace lite {
// SubFunction of read buffer from file
static size_t ReadBuffer(const char* file_name, char** out) {
  FILE* fp;
  fp = fopen(file_name, "rb");
  CHECK(fp != nullptr) << " %s open failed !";
  fseek(fp, 0, SEEK_END);
  auto size = static_cast<size_t>(ftell(fp));
  rewind(fp);
  LOG(INFO) << "model size: " << size;
  *out = reinterpret_cast<char*>(malloc(size));
  size_t cur_len = 0;
  size_t nread;
  while ((nread = fread(*out + cur_len, 1, size - cur_len, fp)) != 0) {
    cur_len += nread;
  }
  fclose(fp);
  return cur_len;
}

TEST(LightAPI, load) {
  if (FLAGS_optimized_model.empty()) {
    FLAGS_optimized_model = "lite_naive_model";
  }
  lite_api::MobileConfig config;
  config.set_model_dir(FLAGS_optimized_model);
  LightPredictor predictor(config, lite_api::LiteModelType::kNaiveBuffer);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<int64_t>({100, 100})));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor.Run();

  const auto* output = predictor.GetOutput(0);
  const float* raw_output = output->data<float>();

  for (int i = 0; i < 10; i++) {
    LOG(INFO) << "out " << raw_output[i];
  }
}

TEST(LightAPI, loadNaiveBuffer) {
  if (FLAGS_optimized_model.empty()) {
    FLAGS_optimized_model = "lite_naive_model";
  }

  auto model_path = std::string(FLAGS_optimized_model) + "/__model__.nb";
  auto params_path = std::string(FLAGS_optimized_model) + "/param.nb";
  char* bufModel = nullptr;
  size_t sizeBuf = ReadBuffer(model_path.c_str(), &bufModel);
  char* bufParams = nullptr;
  std::cout << "sizeBuf: " << sizeBuf << std::endl;
  size_t sizeParams = ReadBuffer(params_path.c_str(), &bufParams);
  std::cout << "sizeParams: " << sizeParams << std::endl;

  lite_api::MobileConfig config;
  config.set_model_buffer(bufModel, sizeBuf, bufParams, sizeParams);
  LightPredictor predictor(config, lite_api::LiteModelType::kNaiveBuffer);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<int64_t>({100, 100})));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor.Run();

  const auto* output = predictor.GetOutput(0);
  const float* raw_output = output->data<float>();

  for (int i = 0; i < 10; i++) {
    LOG(INFO) << "out " << raw_output[i];
  }
}

}  // namespace lite
}  // namespace paddle
