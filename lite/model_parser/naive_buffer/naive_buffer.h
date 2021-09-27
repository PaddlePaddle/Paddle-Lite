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
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/types.h"
#include "lite/utils/container.h"
#include "lite/utils/log/cp_logging.h"

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
  void AppendToFile(const std::string& filename) const;

  //  void LoadFromFile(const std::string& filename);
  void LoadFromFile(const std::string& filename,
                    const size_t& offset = 0,
                    const size_t& size = 0);
  void LoadFromMemory(const char* buffer, size_t buffer_size);
};

/*
 * Base class of all the fields.
 */
class FieldBuilder {
  BinaryTable* table_{};

 public:
  explicit FieldBuilder(BinaryTable* table) : table_(table) {}

  // Write data to table and update the overall cursor.
  virtual void Save() = 0;
  // Load data from table and update the overall cursor.
  virtual void Load() = 0;

  virtual Type type() const = 0;

  BinaryTable* table() { return table_; }

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
  PrimaryBuilder(BinaryTable* table, const Primary& val)
      : FieldBuilder(table), data_(val) {}

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

using BoolBuilder = PrimaryBuilder<bool>;
using CharBuilder = PrimaryBuilder<char>;
using Int32Builder = PrimaryBuilder<int32_t>;
using UInt32Builder = PrimaryBuilder<uint32_t>;
using Int64Builder = PrimaryBuilder<int64_t>;
using UInt64Builder = PrimaryBuilder<uint64_t>;
using Float32Builder = PrimaryBuilder<float>;
using Float64Builder = PrimaryBuilder<double>;

template <typename Primary>
class PrimaryListBuilder : public FieldBuilder {
  const Primary* data_{nullptr};
  int size_{0};

 public:
  using value_type = Primary;

  explicit PrimaryListBuilder(BinaryTable* table) : FieldBuilder(table) {}
  PrimaryListBuilder(BinaryTable* table, const Primary* val, int size)
      : FieldBuilder(table), data_(val), size_(size) {}

  /// Set data.
  void set(const Primary* x, int size) {
    data_ = x;
    size_ = size;
  }

  const Primary* data() const { return data_; }

  /// Save information to the corresponding BinaryTable.
  void Save() override;

  /// Load information from the corresponding BinaryTable.
  void Load() override;

  /// Number of elements.
  size_t size() const { return size_; }

  Type type() const override { return core::StdTypeToRepr<const Primary*>(); }

  /// clear builder
  void Clear() { size_ = 0; }

  ~PrimaryListBuilder() = default;
};

/*
 * Builder for all the primary types. int32, float, bool and so on.
 */
template <typename EnumType>
class EnumBuilder : public FieldBuilder {
  EnumType data_;

 public:
  using value_type = int32_t;

  explicit EnumBuilder(BinaryTable* table) : FieldBuilder(table) {}

  /// Set data.
  void set(EnumType x) { data_ = x; }

  EnumType data() const { return data_; }

  /// Save information to the corresponding BinaryTable.
  void Save() override;

  /// Load information from the corresponding BinaryTable.
  void Load() override;

  ~EnumBuilder() = default;

  Type type() const override { return Type::ENUM; }
};

class StringBuilder : public FieldBuilder {
  std::string data_;

 public:
  explicit StringBuilder(BinaryTable* table) : FieldBuilder(table) {}
  StringBuilder(BinaryTable* table, const std::string& val)
      : FieldBuilder(table), data_(val) {}

  void set(const std::string& x) { data_ = x; }

  const std::string& data() const { return data_; }

  void Save() override;

  void Load() override;

  Type type() const override { return Type::STRING; }
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
 * GetMutableField<Int32Builder>("age") will get the mutable age field declared
 * in `MyStruct`
 */
class StructBuilder : public FieldBuilder {
  OrderedMap<std::unique_ptr<FieldBuilder>> field_builders_;

 public:
  explicit StructBuilder(BinaryTable* table) : FieldBuilder(table) {}

#define NEW_PRIMARY_BUILDER_DECLARE(T, name__, dft_val__) \
  PrimaryBuilder<T>* New##name__(const std::string& name, T val = dft_val__);
  NEW_PRIMARY_BUILDER_DECLARE(bool, Bool, false);
  NEW_PRIMARY_BUILDER_DECLARE(char, Char, 0);
  NEW_PRIMARY_BUILDER_DECLARE(int32_t, Int32, 0);
  NEW_PRIMARY_BUILDER_DECLARE(uint32_t, UInt32, 0);
  NEW_PRIMARY_BUILDER_DECLARE(int64_t, Int64, 0);
  NEW_PRIMARY_BUILDER_DECLARE(uint64_t, UInt64, 0);
  NEW_PRIMARY_BUILDER_DECLARE(float, Float32, 0.0);
  NEW_PRIMARY_BUILDER_DECLARE(double, Float64, 0.0);
#undef NEW_PRIMARY_BUILDER_DECLARE

  /// Create a string field called `name`.
  StringBuilder* NewStr(const std::string& name, const std::string& val = "");

  /// Create a user-defined field, this can build a complex composed struct.
  template <typename CustomBuilder>
  CustomBuilder* New(const std::string& name);

  /// Save the fields' information to the corresponding BinaryTable.
  void Save() override;

  /// Load the fields' information from the corresponding BinaryTable.
  void Load() override;

  /// Type of this struct.
  // TODO(Superjomn) The customized type is not supported yet.
  Type type() const override { return Type::UNK; }

  /// Get a field by `name`.
  template <typename T>
  const T& GetField(const std::string& name) const {
    auto& builder = field_builders_.Get(name);
    return *(static_cast<const T*>(builder.get()));
  }

  /// Get a mutable field by `name`.
  template <typename T>
  T* GetMutableField(const std::string& name) {
    auto& builder = field_builders_.GetMutable(name);
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
  std::deque<Builder> builders_;

 public:
  explicit ListBuilder(BinaryTable* table) : FieldBuilder(table) {}

  // Create a new element.
  Builder* New() {
    builders_.emplace_back(table());
    return &builders_.back();
  }

  // Get i-th element.
  const Builder& Get(int i) const {
    CHECK_LT(i, builders_.size());
    return builders_[i];
  }

  Builder* GetMutable(int i) {
    CHECK_LT(i, builders_.size());
    return &builders_[i];
  }

  typename std::deque<Builder>::iterator begin() { return builders_.begin(); }

  typename std::deque<Builder>::iterator end() { return builders_.end(); }

  typename std::deque<Builder>::const_iterator begin() const {
    return builders_.begin();
  }

  typename std::deque<Builder>::const_iterator end() const {
    return builders_.end();
  }

  // Get element type.
  Type type() const override { return Type::LIST; }

  /// Persist information to the corresponding BinaryTable.
  void Save() override;

  /// Load information from the corresponding BinaryTable.
  void Load() override;

  /// Number of elements.
  size_t size() const { return builders_.size(); }

  /// clear builders
  void Clear() { builders_.clear(); }
};

template <typename Builder>
void ListBuilder<Builder>::Save() {
  // store number of elements in the head.
  uint64_t num_elems = size();
  table()->Require(sizeof(uint64_t));
  memcpy(table()->cursor(), &num_elems, sizeof(uint64_t));
  table()->Consume(sizeof(uint64_t));

  // Save all the elements.
  for (auto& elem : builders_) {
    elem.Save();
  }
}

template <typename Builder>
void ListBuilder<Builder>::Load() {
  CHECK(builders_.empty()) << "Duplicate load";
  // Load number of elements first.
  uint64_t num_elems{};
  memcpy(&num_elems, table()->cursor(), sizeof(uint64_t));
  table()->Consume(sizeof(uint64_t));

  // Load all the elements.
  for (uint64_t i = 0; i < num_elems; i++) {
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

template <typename Primary>
void PrimaryListBuilder<Primary>::Load() {
  CHECK(data_ == nullptr) << "Duplicate load";
  // Load number of elements first.
  uint64_t num_elems{};
  memcpy(&num_elems, table()->cursor(), sizeof(uint64_t));
  table()->Consume(sizeof(uint64_t));

  set(reinterpret_cast<Primary*>(table()->cursor()), num_elems);
  table()->Consume(num_elems * sizeof(value_type));
}

template <typename Primary>
void PrimaryListBuilder<Primary>::Save() {
  // store number of elements in the head.
  uint64_t num_elems = size();
  table()->Require(sizeof(uint64_t));
  memcpy(table()->cursor(), &num_elems, sizeof(uint64_t));
  table()->Consume(sizeof(uint64_t));

  table()->Require(num_elems * sizeof(value_type));
  memcpy(table()->cursor(),
         reinterpret_cast<const byte_t*>(data_),
         num_elems * sizeof(value_type));
  table()->Consume(num_elems * sizeof(value_type));
}

template <typename EnumType>
void EnumBuilder<EnumType>::Save() {
  value_type holder = static_cast<value_type>(data_);
  table()->Require(sizeof(value_type));
  memcpy(table()->cursor(),
         reinterpret_cast<byte_t*>(&holder),
         sizeof(value_type));
  table()->Consume(sizeof(value_type));
}

template <typename EnumType>
void EnumBuilder<EnumType>::Load() {
  value_type holder;
  memcpy(&holder, table()->cursor(), sizeof(value_type));
  table()->Consume(sizeof(value_type));
  data_ = static_cast<EnumType>(holder);
}

template <typename CustomBuilder>
CustomBuilder* StructBuilder::New(const std::string& name) {
  using type = CustomBuilder;
  field_builders_.Set(name, std::unique_ptr<CustomBuilder>(new type(table())));
  return static_cast<type*>(field_builders_.GetMutable(name).get());
}

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
