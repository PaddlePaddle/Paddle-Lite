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

namespace paddle {
namespace lite {
namespace fbs {

void LoadModel(const std::string& path, ProgramDesc* prog) {
  std::ifstream infile;
  infile.open(path, std::ios::binary | std::ios::in);
  infile.seekg(0, std::ios::end);
  int length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  char* buff = new char[length];
  std::unique_ptr<char[]> buf(buff);
  infile.read(buf.get(), length);
  infile.close();
  prog->Init(std::move(buf));
}

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
