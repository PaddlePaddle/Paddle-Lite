/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include <iostream>
#include "../test_helper.h"
#include "io/paddle_inference_api.h"

static size_t ReadBuffer(const char *file_name, uint8_t **out) {
  FILE *fp;
  fp = fopen(file_name, "rb");
  PADDLE_MOBILE_ENFORCE(fp != nullptr, " %s open failed !", file_name);
  fseek(fp, 0, SEEK_END);
  auto size = static_cast<size_t>(ftell(fp));
  rewind(fp);
  DLOG << "model size: " << size;
  *out = reinterpret_cast<uint8_t *>(malloc(size));
  size_t cur_len = 0;
  size_t nread;
  while ((nread = fread(*out + cur_len, 1, size - cur_len, fp)) != 0) {
    cur_len += nread;
  }
  fclose(fp);
  return cur_len;
}

static char *Get_binary_data(std::string filename) {
  FILE *file = fopen(filename.c_str(), "rb");
  PADDLE_MOBILE_ENFORCE(file != nullptr, "can't open file: %s ",
                        filename.c_str());
  fseek(file, 0, SEEK_END);
  int64_t size = ftell(file);
  PADDLE_MOBILE_ENFORCE(size > 0, "size is too small");
  rewind(file);
  auto *data = new char[size];
  size_t bytes_read = fread(data, 1, size, file);
  PADDLE_MOBILE_ENFORCE(bytes_read == size,
                        "read binary file bytes do not match with fseek");
  fclose(file);
  return data;
}

paddle_mobile::PaddleMobileConfig GetConfig() {
  paddle_mobile::PaddleMobileConfig config;
  config.precision = paddle_mobile::PaddleMobileConfig::FP32;
  config.device = paddle_mobile::PaddleMobileConfig::kGPU_CL;
  const std::shared_ptr<paddle_mobile::PaddleModelMemoryPack> &memory_pack =
      std::make_shared<paddle_mobile::PaddleModelMemoryPack>();
  auto model_path = std::string(g_mobilenet_combined) + "/model";
  auto params_path = std::string(g_mobilenet_combined) + "/params";
  memory_pack->model_size =
      ReadBuffer(model_path.c_str(), &memory_pack->model_buf);
  std::cout << "sizeBuf: " << memory_pack->model_size << std::endl;
  memory_pack->combined_params_size =
      ReadBuffer(params_path.c_str(), &memory_pack->combined_params_buf);
  std::cout << "sizeParams: " << memory_pack->combined_params_size << std::endl;
  memory_pack->from_memory = true;
  config.memory_pack = *memory_pack;
  config.thread_num = 4;
  return config;
}
int main() {
  paddle_mobile::PaddleMobileConfig config = GetConfig();
  auto predictor = paddle_mobile::CreatePaddlePredictor<
      paddle_mobile::PaddleMobileConfig,
      paddle_mobile::PaddleEngineKind::kPaddleMobile>(config);
  return 0;
}
