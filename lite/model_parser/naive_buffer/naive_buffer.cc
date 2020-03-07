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
#include <stdio.h>

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
  FILE *fp = fopen(filename.c_str(), "wb");
  CHECK(fp) << "Unable to open file: " << filename;
  if (fwrite(reinterpret_cast<const char *>(data()), 1, size(), fp) != size()) {
    fclose(fp);
    LOG(FATAL) << "Write file error: " << filename;
  }
  fclose(fp);
}

void BinaryTable::AppendToFile(const std::string &filename) const {
  FILE *fp = fopen(filename.c_str(), "ab");
  CHECK(fp) << "Unable to open file: " << filename;
  if (fwrite(reinterpret_cast<const char *>(data()), 1, size(), fp) != size()) {
    fclose(fp);
    LOG(FATAL) << "Write file error: " << filename;
  }
  fclose(fp);
}

void BinaryTable::LoadFromFile(const std::string &filename,
                               const size_t &offset,
                               const size_t &size) {
  // open file in readonly mode
  FILE *fp = fopen(filename.c_str(), "rb");
  CHECK(fp) << "Unable to open file: " << filename;
  // move fstream pointer backward for size of offset
  size_t buffer_size = size;
  if (size == 0) {
    fseek(fp, 0L, SEEK_END);
    buffer_size = ftell(fp) - offset;
  }
  fseek(fp, offset, SEEK_SET);
  Require(buffer_size);
  // read data of `size` into binary_data_variable:`bytes_`
  if (fread(reinterpret_cast<char *>(&bytes_[0]), 1, buffer_size, fp) !=
      buffer_size) {
    fclose(fp);
    LOG(FATAL) << "Read file error: " << filename;
  }
  fclose(fp);
  // Set readonly.
  is_mutable_mode_ = false;
}

void BinaryTable::LoadFromMemory(const char *buffer, size_t buffer_size) {
  // get buffer
  bytes_.resize(buffer_size);
  memcpy(reinterpret_cast<char *>(&bytes_[0]), buffer, buffer_size);
  // Set readonly.
  is_mutable_mode_ = false;
}

void StringBuilder::Save() {
  // memory format: [size][string data]
  uint64_t mem_size = sizeof(uint64_t) + data_.size();
  table()->Require(mem_size);
  uint64_t str_len = data_.size();

  // write meta data of size.
  memcpy(table()->cursor(), &str_len, sizeof(uint64_t));
  table()->Consume(sizeof(uint64_t));

  // write the string data.
  memcpy(table()->cursor(),
         reinterpret_cast<const byte_t *>(data_.c_str()),
         str_len);
  table()->Consume(str_len);
}

void StringBuilder::Load() {
  // load meta data of size
  uint64_t str_len{};
  memcpy(&str_len, table()->cursor(), sizeof(uint64_t));
  table()->Consume(sizeof(uint64_t));

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
