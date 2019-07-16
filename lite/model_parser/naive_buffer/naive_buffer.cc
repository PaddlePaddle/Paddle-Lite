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

#include "lite/model_parser/naive_buffer/naive_buffer.h"

namespace paddle {
namespace lite {
namespace naive_buffer {

void BinaryTable::Require(size_t size) {
  CHECK(is_mutable_mode_);
  if (free_size() < size) {
    bytes_.resize(cursor_ + size);
  }
}

void BinaryTable::Consume(size_t bytes) {
  CHECK_LE(bytes, free_size()) << "No free memory of " << bytes
                               << ", should Require the memory first";
  cursor_ += bytes;
  // Consume is used in both readonly and mutable mode to move the write/read
  // cursor, so we don't check mutable mode here.
}

void BinaryTable::SaveToFile(const std::string &filename) const {
  std::ofstream file(filename, std::ios::binary);
  CHECK(file.is_open()) << "failed to open " << filename;
  file.write(reinterpret_cast<const char *>(data()), size());
  file.close();
}

void BinaryTable::LoadFromFile(const std::string &filename) {
  // get file size
  std::ifstream file(filename, std::ios::binary);
  CHECK(file.is_open()) << "Unable to open file: " << filename;
  const auto fbegin = file.tellg();
  file.seekg(0, std::ios::end);
  const auto fend = file.tellg();
  size_t file_size = fend - fbegin;

  // load data.
  LOG(INFO) << "file size " << file_size;
  file.seekg(0, std::ios::beg);
  Require(file_size);
  file.read(reinterpret_cast<char *>(&bytes_[0]), file_size);

  // Set readonly.
  is_mutable_mode_ = false;
}

void StringBuilder::Save() {
  // memory format: [size][string data]
  size_t mem_size = sizeof(size_t) + data_.size();
  table()->Require(mem_size);
  size_t str_len = data_.size();

  // write meta data of size.
  memcpy(table()->cursor(), &str_len, sizeof(size_t));
  table()->Consume(sizeof(size_t));

  // write the string data.
  memcpy(table()->cursor(),
         reinterpret_cast<const byte_t *>(data_.c_str()),
         str_len);
  table()->Consume(str_len);
}

void StringBuilder::Load() {
  // load meta data of size
  size_t str_len{};
  memcpy(&str_len, table()->cursor(), sizeof(size_t));
  table()->Consume(sizeof(size_t));

  // load string data.
  data_.resize(str_len);
  memcpy(&data_[0], table()->cursor(), str_len);
  table()->Consume(str_len);
}

#define NEW_PRIMARY_BUILDER_IMPL(T, name__)                                   \
  PrimaryBuilder<T> *StructBuilder::New##name__(const std::string &name,      \
                                                T val) {                      \
    using type = PrimaryBuilder<T>;                                           \
    field_builders_.Set(name, std::unique_ptr<type>(new type(table(), val))); \
    return static_cast<type *>(field_builders_.Get(name).get());              \
  }
NEW_PRIMARY_BUILDER_IMPL(bool, Bool);
NEW_PRIMARY_BUILDER_IMPL(char, Char);
NEW_PRIMARY_BUILDER_IMPL(int32_t, Int32);
NEW_PRIMARY_BUILDER_IMPL(uint32_t, UInt32);
NEW_PRIMARY_BUILDER_IMPL(int64_t, Int64);
NEW_PRIMARY_BUILDER_IMPL(uint64_t, UInt64);
NEW_PRIMARY_BUILDER_IMPL(float, Float32);
NEW_PRIMARY_BUILDER_IMPL(double, Float64);
#undef NEW_PRIMARY_BUILDER_IMPL

StringBuilder *StructBuilder::NewStr(const std::string &name,
                                     const std::string &val) {
  using type = StringBuilder;
  field_builders_.Set(name, std::unique_ptr<type>(new type(table(), val)));
  return static_cast<type *>(field_builders_.Get(name).get());
}

void StructBuilder::Save() {
  for (auto &elem : field_builders_.elements()) {
    elem->Save();
  }
}

void StructBuilder::Load() {
  for (auto &elem : field_builders_.elements()) {
    elem->Load();
  }
}

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
