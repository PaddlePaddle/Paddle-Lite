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

#include "common/util.h"

namespace paddle_mobile {

char *ReadFileToBuff(std::string filename) {
  FILE *file = fopen(filename.c_str(), "rb");
  PADDLE_MOBILE_ENFORCE(file != nullptr, "can't open file: %s ",
                        filename.c_str());
  fseek(file, 0, SEEK_END);
  int64_t size = ftell(file);
  PADDLE_MOBILE_ENFORCE(size > 0, "file should not be empty");
  rewind(file);
  char *data = new char[size];
  size_t bytes_read = fread(data, 1, size, file);
  PADDLE_MOBILE_ENFORCE(bytes_read == size,
                        "read binary file bytes do not match with fseek");
  fclose(file);
  return data;
}

int GetFileLength(std::string filename) {
  FILE *file = fopen(filename.c_str(), "rb");
  PADDLE_MOBILE_ENFORCE(file != nullptr, "can't open file: %s ",
                        filename.c_str());
  fseek(file, 0, SEEK_END);
  int size = ftell(file);
  PADDLE_MOBILE_ENFORCE(size > 0, "file should not be empty");
  fclose(file);
  return size;
}

}  // namespace paddle_mobile
