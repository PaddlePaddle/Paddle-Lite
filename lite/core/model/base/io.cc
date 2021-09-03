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

#include "lite/core/model/base/io.h"

namespace paddle {
namespace lite {
namespace model_parser {

void Buffer::CopyDataFrom(const Buffer& other) {
  const auto* other_raw = other.raw();
  CHECK(other_raw);
  raw_->CopyDataFrom(*other_raw, other.size());
}

void Buffer::ResetLazy(size_t size) {
  if (size == 0) {
    size = 1;
  }
  CHECK(raw_);
  raw_->ResetLazy(TargetType::kHost, size);
  size_ = size;
}

std::string ByteReader::ReadToString(size_t size) const {
  std::string tmp;
  tmp.resize(size);
  Read(&tmp[0], size);
  tmp.shrink_to_fit();
  return tmp;
}

BinaryFileReader::BinaryFileReader(const std::string& path, size_t offset) {
  file_ = fopen(path.c_str(), "rb");
  CHECK(file_) << "Unable to open file: " << path;
  fseek(file_, 0L, SEEK_END);
  length_ = ftell(file_) - offset;
  fseek(file_, offset, SEEK_SET);
}

void BinaryFileReader::Read(void* dst, size_t size) const {
  CHECK(dst);
  CHECK_EQ(fread(dst, 1, size, file_), size) << "Failed to read " << size
                                             << " bytes.";
  cur_ += size;
}

void BinaryFileWriter::Write(const void* src, size_t size) const {
  CHECK(src);
  CHECK_EQ(fwrite(src, 1, size, file_), size) << "Failed to read " << size
                                              << "bytes.";
  cur_ += size;
}

void StringBufferReader::Read(void* dst, size_t size) const {
  CHECK(dst);
  lite::TargetCopy(TargetType::kHost, dst, buf_ + cur_, size);
  cur_ += size;
}

}  // namespace model_parser
}  // namespace lite
}  // namespace paddle
