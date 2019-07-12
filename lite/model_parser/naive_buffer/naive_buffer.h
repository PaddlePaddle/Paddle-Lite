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

#pragma once
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/types.h"
#include "lite/utils/container.h"
#include "lite/utils/cp_logging.h"

/*
 * This file contains the implementation of NaiveBuffer. We implement the basic
 * interfaces for serialization and de-serialization for a PaddlePaddle model to
 * avoid using the third-party libraries such as protobuf, and make the lite
 * dependencies small and easy to compile and deploy.
 */

namespace paddle {
namespace lite {
namespace naive_buffer {
using core::Type;

using byte_t = uint8_t;

/*
 * BinaryTable is a binary buffer, it holds all the fields of a NaiveBuffer
 * object.
 * A BinaryTable can only support write or read in its lifetime, it is mutable
 * by default, but the `Load` method will get a readonly BinaryTable.
 */
struct BinaryTable {
 private:
  std::vector<byte_t> bytes_;
  size_t cursor_{};
  bool is_mutable_mode_{true};  // true for mutable, false for readonly.

 public:
  /// Require free memory of `size` bytes.
  void Require(size_t size);

  /// Consume some memory.
  void Consume(size_t bytes);

  /// The current position of cursor for save or load.
  byte_t* cursor() { return &bytes_[cursor_]; }
  const byte_t* data() const { return bytes_.data(); }
  size_t size() const { return bytes_.size(); }
  size_t free_size() const { return bytes_.size() - cursor_; }

  /// Serialize the table to a binary buffer.
  void SaveToFile(const std::string& filename) const;

  void LoadFromFile(const std::string& filename);
};

/*
 * Base class of all the fields.
 */
class FieldBuilder {
  BinaryTable* table_{};

 protected:
  BinaryTable* table() { return table_; }

 public:
  explicit FieldBuilder(BinaryTable* table) : table_(table) {}

  // Write data to table and update the overall cursor.
  virtual void Save() = 0;
  // Load data from table and update the overall cursor.
  virtual void Load() = 0;

  virtual Type type() const = 0;

  virtual ~FieldBuilder() = default;
};

/*
 * Builder for all the primary types. int32, float, bool and so on.
 */
template <typename Primary>
class PrimaryBuilder : public FieldBuilder {
  Primary data_;

 public:
  using value_type = Primary;

  explicit PrimaryBuilder(BinaryTable* table) : FieldBuilder(table) {}

  /// Set data.
  void set(Primary x) { data_ = x; }

  Primary data() const { return data_; }

  /// Save information to the corresponding BinaryTable.
  void Save() override;

  /// Load information from the corresponding BinaryTable.
  void Load() override;

  Type type() const override { return core::StdTypeToRepr<Primary>(); }

  ~PrimaryBuilder() = default;
};

using Int32Builder = PrimaryBuilder<int32_t>;
using Int64Builder = PrimaryBuilder<int64_t>;
using Float32Builder = PrimaryBuilder<float>;
using Float64Builder = PrimaryBuilder<double>;
using BoolBuilder = PrimaryBuilder<bool>;

class StringBuilder : public FieldBuilder {
  std::string data_;

 public:
  explicit StringBuilder(BinaryTable* table) : FieldBuilder(table) {}

  void set(const std::string& x) { data_ = x; }

  const std::string& data() const { return data_; }

  void Save() override;

  void Load() override;

  Type type() const override { return Type::_string; }
};

/*
 * This is a data structure. A composion of multiple fields.
 *
 * Usage:
 *
 * class MyStruct : public StructBuilder {
 *   public:
 *     MyStruct(BinaryTable* table) : StructBuilder(table) {
 *       NewStr("name");
 *       NewInt32("age");
 *     }
 * };
 *
 * One can retrive a field with the specific field name.
 * e.g.
 * GetField<Int32Builder>("age") will get the age field declared in `MyStruct`
 */
class StructBuilder : public FieldBuilder {
  OrderedMap<std::unique_ptr<FieldBuilder>> field_builders_;

 public:
  explicit StructBuilder(BinaryTable* table) : FieldBuilder(table) {}

  /// Create a Int32 field called `name`.
  PrimaryBuilder<int32_t>* NewInt32(const std::string& name);

  /// Create a Int64 field called `name`.
  PrimaryBuilder<int64_t>* NewInt64(const std::string& name);

  /// Create a string field called `name`.
  StringBuilder* NewStr(const std::string& name);

  /// Create a user-defined field, this can build a complex composed struct.
  template <typename CustomBuilder>
  CustomBuilder* New(const std::string& name);

  /// Save the fields' information to the corresponding BinaryTable.
  void Save() override;

  /// Load the fields' information from the corresponding BinaryTable.
  void Load() override;

  /// Type of this struct.
  // TODO(Superjomn) The customized type is not supported yet.
  Type type() const override { return Type::_unk; }

  /// Get a field by `name`.
  template <typename T>
  T* GetField(const std::string& name) {
    auto& builder = field_builders_.Get(name);
    return static_cast<T*>(builder.get());
  }
};

/*
 * Builder of a Struct List.
 *
 * Such as
 *
 * ListBuilder<Int32Builder> is equal to a vector<int32>
 */
template <typename Builder>
class ListBuilder : public FieldBuilder {
  std::vector<Builder> builders_;

 public:
  explicit ListBuilder(BinaryTable* table) : FieldBuilder(table) {}

  // Create a new element.
  Builder* New() {
    builders_.emplace_back(table());
    return &builders_.back();
  }

  // Get i-th element.
  Builder* Get(int i) {
    CHECK_LT(i, builders_.size());
    return &builders_[i];
  }

  // Get element type.
  Type type() const override { return Type::_list; }

  /// Persist information to the corresponding BinaryTable.
  void Save() override;

  /// Load information from the corresponding BinaryTable.
  void Load() override;

  /// Number of elements.
  size_t size() const { return builders_.size(); }
};

template <typename Builder>
void ListBuilder<Builder>::Save() {
  // store number of elements in the head.
  size_t num_elems = size();
  table()->Require(sizeof(size_t));
  memcpy(table()->cursor(), &num_elems, sizeof(size_t));
  table()->Consume(sizeof(size_t));

  // Save all the elements.
  for (auto& elem : builders_) {
    elem.Save();
  }
}

template <typename Builder>
void ListBuilder<Builder>::Load() {
  CHECK(builders_.empty()) << "Duplicate load";
  // Load number of elements first.
  size_t num_elems{};
  memcpy(&num_elems, table()->cursor(), sizeof(num_elems));
  table()->Consume(sizeof(size_t));

  // Load all the elements.
  for (size_t i = 0; i < num_elems; i++) {
    builders_.emplace_back(table());
    builders_.back().Load();
  }
}

template <typename Primary>
void PrimaryBuilder<Primary>::Save() {
  table()->Require(sizeof(value_type));
  memcpy(
      table()->cursor(), reinterpret_cast<byte_t*>(&data_), sizeof(value_type));
  table()->Consume(sizeof(value_type));
}

template <typename Primary>
void PrimaryBuilder<Primary>::Load() {
  memcpy(&data_, table()->cursor(), sizeof(value_type));
  table()->Consume(sizeof(value_type));
}

template <typename CustomBuilder>
CustomBuilder* StructBuilder::New(const std::string& name) {
  using type = CustomBuilder;
  field_builders_.Set(name, std::unique_ptr<CustomBuilder>(new type(table())));
  return static_cast<type*>(field_builders_.Get(name).get());
}

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
