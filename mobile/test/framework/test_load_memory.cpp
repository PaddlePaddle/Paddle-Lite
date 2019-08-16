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

#include <iostream>
#include <string>
#include "../test_helper.h"
#include "../test_include.h"

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

int main() {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  auto model_path = std::string(g_genet_combine) + "/model";
  auto params_path = std::string(g_genet_combine) + "/params";
  uint8_t *bufModel = nullptr;
  size_t sizeBuf = ReadBuffer(model_path.c_str(), &bufModel);
  uint8_t *bufParams = nullptr;

  std::cout << "sizeBuf: " << sizeBuf << std::endl;
  size_t sizeParams = ReadBuffer(params_path.c_str(), &bufParams);
  std::cout << "sizeParams: " << sizeParams << std::endl;

  paddle_mobile.LoadCombinedMemory(sizeBuf, bufModel, sizeParams, bufParams);
  return 0;
}
