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
#include <string>
#include "common/enforce.h"
#include "common/log.h"

namespace paddle_mobile {

template <int ID, typename Type>
struct IDToType {
  typedef Type type_t;
};

template <typename F, typename... Ts>
struct VariantHelper {
  static const size_t size = sizeof(F) > VariantHelper<Ts...>::size
                                 ? sizeof(F)
                                 : VariantHelper<Ts...>::size;

  inline static void Destroy(size_t id, void *data) {
    if (id == typeid(F).hash_code()) {
      reinterpret_cast<F *>(data)->~F();
    } else {
      VariantHelper<Ts...>::Destroy(id, data);
    }
  }
};

template <typename F>
struct VariantHelper<F> {
  static const size_t size = sizeof(F);
  inline static void Destroy(size_t id, void *data) {
    if (id == typeid(F).hash_code()) {
      //              reinterpret_cast<F*>(data)->~F();
    } else {
      //              std::cout << "未匹配到 " << std::endl;
    }
  }
};

template <size_t size>
class RawData {
 public:
  char data[size];
  RawData() {}
  RawData(const RawData &raw_data) { memcpy(data, raw_data.data, size); }

  RawData &operator=(const RawData &raw_data) {
    memcpy(data, raw_data.data, size);
    return *this;
  }
};

template <typename... Ts>
struct Variant {
  Variant(const Variant &variant) {
    type_id = variant.type_id;
    data = variant.data;
  }

  Variant() : type_id(invalid_type()) {}
  ~Variant() {
    //        helper::Destroy(type_id, &data);
  }

  template <typename T, typename... Args>
  void Set(Args &&... args) {
    helper::Destroy(type_id, data.data);
    new (data.data) T(std::forward<Args>(args)...);
    type_id = typeid(T).hash_code();
  }

  void SetString(std::string &string) {
    helper::Destroy(type_id, data.data);
    type_id = typeid(std::string).hash_code();
    strcpy(data.data, string.c_str());
  }

  std::string GetString() const {
    if (type_id == typeid(std::string).hash_code()) {
      return std::string(data.data);
    } else {
      PADDLE_MOBILE_THROW_EXCEPTION(
          " bad cast in variant data type not a string ");
      exit(0);
    }
  }

  template <typename T>
  T &Get() const {
    if (type_id == typeid(std::string).hash_code()) {
      PADDLE_MOBILE_THROW_EXCEPTION(
          "Please use getString to get an string (to avoid of an issue with "
          "gcc "
          "stl lib with string copy)");
      exit(0);
    } else if (type_id == typeid(T).hash_code()) {
      return *const_cast<T *>(reinterpret_cast<const T *>(data.data));
    } else {
      PADDLE_MOBILE_THROW_EXCEPTION(" bad cast in variant");
      exit(0);
    }
  }

  size_t TypeId() const { return type_id; }

 private:
  static inline size_t invalid_type() { return typeid(void).hash_code(); }
  typedef VariantHelper<Ts...> helper;
  size_t type_id;
  // todo use an anto size to suite this.
  RawData<64> data;
};

template <typename T>
struct Vistor {
  typedef T type_t;
};

}  // namespace paddle_mobile
