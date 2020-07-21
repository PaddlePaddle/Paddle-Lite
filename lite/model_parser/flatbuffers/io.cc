// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/model_parser/flatbuffers/io.h"
#include <memory>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace fbs {

void LoadModel(const std::string& path, ProgramDesc* prog) {
  CHECK(prog != nullptr);
  FILE* file = fopen(path.c_str(), "rb");
  fseek(file, 0, SEEK_END);
  int64_t length = ftell(file);
  rewind(file);
  std::vector<char> buf(length);
  CHECK(fread(buf.data(), 1, length, file));
  fclose(file);
  prog->Init(std::move(buf));
}

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
