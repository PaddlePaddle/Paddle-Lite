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
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

// Ported from github:dmlc-core
class Any {
 public:
  inline Any() = default;
  inline explicit Any(Any&& other);
  inline explicit Any(const Any& other);

  template <typename T>
  void set();

  template <typename T>
  void set(T&& other);

  template <typename T>
  const T& get() const;

  template <typename T>
  T* get_mutable();

  template <typename T>
  inline explicit Any(T&& other);

  inline ~Any();

  inline Any& operator=(Any&& other);
  inline Any& operator=(const Any& other);

  template <typename T>
  inline Any& operator=(T&& other);

  inline bool empty() const;
  inline bool valid() const;
  inline void clear();
  inline void swap(Any& other);
  inline const std::type_info& type() const;

  template <typename T, typename... Args>
  inline void construct(Args&&... args);

  template <typename T>
  inline bool is_type() const;

 private:
  template <typename T>
  class TypeOnHeap;

  template <typename T>
  class TypeOnStack;

  template <typename T>
  class TypeInfo;

  static const size_t kStack = sizeof(void*) * 3;
  static const size_t kAlign = sizeof(void*);

  union Data {
    std::aligned_storage<kStack, kAlign>::type stack;
    void* pheap;
  };

  struct Type {
    void (*destroy)(Data* data);
    void (*create_from_data)(Data* dst, const Data& src);
    const std::type_info* ptype_info;
  };

  template <typename T>
  struct data_on_stack {
    static const bool value = ((alignof(T) <= kAlign) && (sizeof(T) <= kStack));
  };

  inline void construct(Any&& other);
  inline void construct(const Any& other);

  template <typename T>
  inline void check_type() const;

  template <typename T>
  inline void check_type_by_name() const;

  const Type* type_{nullptr};
  Data data_;
};

template <typename T>
inline Any::Any(T&& other) {
  typedef typename std::decay<T>::type DT;
  if (std::is_same<DT, Any>::value) {
    construct(std::forward<T>(other));
  } else {
    static_assert(std::is_copy_constructible<DT>::value,
                  "Any can only hold value that is copy constructable");
    type_ = TypeInfo<DT>::get_type();
    if (data_on_stack<DT>::value) {
#pragma GCC diagnostic push
#if 6 <= __GNUC__
#pragma GCC diagnostic ignored "-Wplacement-new"
#endif
      new (&(data_.stack)) DT(std::forward<T>(other));
#pragma GCC diagnostic pop
    } else {
      data_.pheap = new DT(std::forward<T>(other));
    }
  }
}

inline Any::Any(Any&& other) { construct(std::move(other)); }

inline Any::Any(const Any& other) { construct(other); }

inline void Any::construct(Any&& other) {
  type_ = other.type_;
  data_ = other.data_;
  other.type_ = nullptr;
}

inline void Any::construct(const Any& other) {
  type_ = other.type_;
  if (type_ != nullptr) {
    type_->create_from_data(&data_, other.data_);
  }
}

template <typename T, typename... Args>
inline void Any::construct(Args&&... args) {
  clear();
  typedef typename std::decay<T>::type DT;
  type_ = TypeInfo<DT>::get_type();
  if (data_on_stack<DT>::value) {
#pragma GCC diagnostic push
#if 6 <= __GNUC__
#pragma GCC diagnostic ignored "-Wplacement-new"
#endif
    new (&(data_.stack)) DT(std::forward<Args>(args)...);
#pragma GCC diagnostic pop
  } else {
    data_.pheap = new DT(std::forward<Args>(args)...);
  }
}

template <typename T>
void Any::set() {
  construct<T>();
}

template <typename T>
void Any::set(T&& other) {
  construct<T>(std::forward<T>(other));
}

inline Any::~Any() { clear(); }

inline Any& Any::operator=(Any&& other) {
  Any(std::move(other)).swap(*this);
  return *this;
}

inline Any& Any::operator=(const Any& other) {
  Any(other).swap(*this);
  return *this;
}

template <typename T>
inline Any& Any::operator=(T&& other) {
  Any(std::forward<T>(other)).swap(*this);
  return *this;
}

inline void Any::swap(Any& other) {
  std::swap(type_, other.type_);
  std::swap(data_, other.data_);
}

inline void Any::clear() {
  if (type_ != nullptr) {
    if (type_->destroy != nullptr) {
      type_->destroy(&data_);
    }
    type_ = nullptr;
  }
}

inline bool Any::empty() const { return type_ == nullptr; }

inline bool Any::valid() const { return empty() == false; }

inline const std::type_info& Any::type() const {
  if (type_ != nullptr) {
    return *(type_->ptype_info);
  } else {
    return typeid(void);
  }
}

template <typename T>
inline bool Any::is_type() const {
  if ((type_ == nullptr) || (*(type_->ptype_info) != typeid(T))) {
    return false;
  }
  return true;
}

template <typename T>
inline void Any::check_type() const {
  CHECK_EQ((type_ == nullptr), false);
  CHECK_EQ((*(type_->ptype_info) == typeid(T)), true)
      << "Any struct is stored in the type " << type_->ptype_info->name()
      << ", but trying to obtain the type " << typeid(T).name() << ".";
}

template <typename T>
inline void Any::check_type_by_name() const {
  CHECK_EQ((type_ == nullptr), false);
  CHECK_EQ(strcmp(type_->ptype_info->name(), typeid(T).name()), 0);
}

template <typename T>
inline const T& Any::get() const {
  check_type<T>();
  return *Any::TypeInfo<T>::get_ptr(&data_);
}

template <typename T>
T* Any::get_mutable() {
  check_type<T>();
  return Any::TypeInfo<T>::get_ptr(&data_);
}

template <typename T>
class Any::TypeOnHeap {
 public:
  inline static T* get_ptr(Any::Data* data) {
    return static_cast<T*>(data->pheap);
  }
  inline static const T* get_ptr(const Any::Data* data) {
    return static_cast<const T*>(data->pheap);
  }
  inline static void create_from_data(Any::Data* dst, const Any::Data& data) {
    dst->pheap = new T(*get_ptr(&data));
  }
  inline static void destroy(Data* data) {
    delete static_cast<T*>(data->pheap);
  }
};

template <typename T>
class Any::TypeOnStack {
 public:
  inline static T* get_ptr(Any::Data* data) {
    return reinterpret_cast<T*>(&(data->stack));
  }
  inline static const T* get_ptr(const Any::Data* data) {
    return reinterpret_cast<const T*>(&(data->stack));
  }
  inline static void create_from_data(Any::Data* dst, const Any::Data& data) {
    new (&(dst->stack)) T(*get_ptr(&data));
  }
  inline static void destroy(Data* data) {
    T* dptr = reinterpret_cast<T*>(&(data->stack));
    dptr->~T();
  }
};

template <typename T>
class Any::TypeInfo : public std::conditional<Any::data_on_stack<T>::value,
                                              Any::TypeOnStack<T>,
                                              Any::TypeOnHeap<T>>::type {
 public:
  inline static const Type* get_type() {
    static TypeInfo<T> tp;
    return &(tp.type_);
  }

 private:
  Type type_;
  TypeInfo() {
    if (std::is_pod<T>::value && data_on_stack<T>::value) {
      type_.destroy = nullptr;
    } else {
      type_.destroy = TypeInfo<T>::destroy;
    }
    type_.create_from_data = TypeInfo<T>::create_from_data;
    type_.ptype_info = &typeid(T);
  }
};

}  // namespace lite
}  // namespace paddle
