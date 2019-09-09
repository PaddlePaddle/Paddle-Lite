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

#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include "common/enforce.h"
#include "common/log.h"
#include "common/type_define.h"

namespace paddle_mobile {

template <int ID, typename Type>
struct IDToType {
  typedef Type type_t;
};

template <typename F, typename... Ts>
struct VariantHelper {
  inline static void Destroy(kTypeId_t type, void *raw_ptr) {
    if (type == type_id<F>()) {
      auto ptr = reinterpret_cast<F *>(raw_ptr);
      delete ptr;
    } else {
      VariantHelper<Ts...>::Destroy(type, raw_ptr);
    }
  }
};

template <typename F>
struct VariantHelper<F> {
  inline static void Destroy(kTypeId_t type, void *raw_ptr) {
    if (type == type_id<F>()) {
      auto ptr = reinterpret_cast<F *>(raw_ptr);
      delete ptr;
    }
  }
};

template <typename... Ts>
struct VariantDeleter {
  kTypeId_t type_ = type_id<void>().hash_code();
  explicit VariantDeleter(kTypeId_t type) { type_ = type; }
  void operator()(void *raw_ptr) {
    // DLOG << "variant delete: " << type_ << " " << raw_ptr;
    VariantHelper<Ts...>::Destroy(type_, raw_ptr);
  }
};

template <typename... Ts>
struct Variant {
  Variant() : type_(invalid_type()) {}

  Variant(const Variant &variant) {
    type_ = variant.type_;
    data_ = variant.data_;
  }

  virtual ~Variant() {
    // DLOG << "variant deinit: " << type_ << " " << (void *)data_.get();
    data_.reset();
  }

  template <typename T, typename... Args>
  void Set(Args &&... args) {
    auto raw_ptr = new T(std::forward<Args>(args)...);
    type_ = type_id<T>().hash_code();
    // DLOG << "variant new: " << type_ << " " << (void *)raw_ptr;
    data_.reset(raw_ptr, VariantDeleter<Ts...>(type_));
  }

  template <typename T>
  T &Get() const {
    return *const_cast<T *>(reinterpret_cast<const T *>(data_.get()));
  }

  kTypeId_t TypeId() const { return type_; }

 private:
  static inline kTypeId_t invalid_type() { return type_id<void>().hash_code(); }
  typedef VariantHelper<Ts...> helper;
  kTypeId_t type_ = type_id<void>().hash_code();
  std::shared_ptr<void> data_;
};

template <typename T>
struct Vistor {
  typedef T type_t;
};

}  // namespace paddle_mobile
