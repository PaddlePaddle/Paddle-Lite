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

#include "lite/model_parser/immediate.h"

namespace paddle {
namespace lite {
namespace model_parser {

template <typename T>
T BytesReader::ReadForward() {
  T tmp;
  ReadForward(&tmp, sizeof(T));
  return tmp;
}

template int64_t BytesReader::ReadForward<int64_t>();
template int32_t BytesReader::ReadForward<int32_t>();
template uint32_t BytesReader::ReadForward<uint32_t>();
template uint64_t BytesReader::ReadForward<uint64_t>();

}  // namespace model_parser
}  // namespace lite
}  // namespace paddle
