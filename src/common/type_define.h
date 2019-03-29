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

#include <functional>
#include <string>
#include <vector>

namespace paddle_mobile {

typedef enum {
  _void = 0,
  _float,
  _int,
  _double,
  _int64_t,
  _size_t,
  _int16_t,
  _int8_t,
  _uint8_t,
  _bool,
  _string,
  _floats = 100,
  _ints,
  _int64_ts,
  _size_ts,
  _bools,
  _strings,
  _const_float = 200,
  _const_int,
  _block = 300,
  _tensor,
  _lod_tensor,
  _blocks,
  _tensors,
  _lod_tensors,
  _p_block = 400,
  _p_tensor,
  _p_lod_tensor,
  _p_blocks,
  _p_tensors,
  _p_lod_tensors,
  _scopes = 500,
  _selected_rows,
  _dim0 = 600,
  _dim1,
  _dim2,
  _dim3,
  _dim4,
  _dim5,
  _dim6,
  _dim7,
  _dim8,
  _dim9,
} kTypeId_t;

template <typename T>
struct TypeIdWrapper {
  inline std::string name();
  inline kTypeId_t hash_code();
};

template <typename T>
struct type_id {
  const kTypeId_t hash_code() const { return TypeIdWrapper<T>().hash_code(); }
  const std::string name() const { return TypeIdWrapper<T>().name(); }

  template <typename OtherType>
  bool operator==(const type_id<OtherType> &operand) const {
    return this->hash_code() == operand.hash_code();
  }
};

template <typename T>
inline bool operator==(const kTypeId_t &t0, const type_id<T> &t1) {
  return t0 == t1.hash_code();
}

template <typename T>
inline bool operator==(const type_id<T> &t0, const kTypeId_t &t1) {
  return t1 == t0.hash_code();
}

namespace framework {
class BlockDesc;
class Tensor;
class LoDTensor;
class SelectedRows;
class Scope;

template <int>
struct Dim;
}  // namespace framework

#define REGISTER_TYPE_ID(Type, TypeName)                         \
  template <>                                                    \
  struct TypeIdWrapper<Type> {                                   \
    inline std::string name() { return std::string(#TypeName); } \
    inline kTypeId_t hash_code() { return kTypeId_t::TypeName; } \
  };

REGISTER_TYPE_ID(void, _void)
REGISTER_TYPE_ID(float, _float)
REGISTER_TYPE_ID(int, _int)
REGISTER_TYPE_ID(double, _double)
REGISTER_TYPE_ID(int64_t, _int64_t)
REGISTER_TYPE_ID(size_t, _size_t)
REGISTER_TYPE_ID(int16_t, _int16_t)
REGISTER_TYPE_ID(int8_t, _int8_t)
REGISTER_TYPE_ID(uint8_t, _uint8_t)
REGISTER_TYPE_ID(bool, _bool)
REGISTER_TYPE_ID(std::string, _string)
REGISTER_TYPE_ID(std::vector<float>, _floats)
REGISTER_TYPE_ID(std::vector<int>, _ints)
REGISTER_TYPE_ID(std::vector<int64_t>, _int64_ts)
REGISTER_TYPE_ID(std::vector<size_t>, _size_ts)
REGISTER_TYPE_ID(std::vector<bool>, _bools)
REGISTER_TYPE_ID(std::vector<std::string>, _strings)

REGISTER_TYPE_ID(float const, _const_float)
REGISTER_TYPE_ID(int const, _const_int)

REGISTER_TYPE_ID(framework::BlockDesc, _block)
REGISTER_TYPE_ID(framework::Tensor, _tensor)
REGISTER_TYPE_ID(framework::LoDTensor, _lod_tensor)
REGISTER_TYPE_ID(std::vector<framework::BlockDesc>, _blocks)
REGISTER_TYPE_ID(std::vector<framework::Tensor>, _tensors)
REGISTER_TYPE_ID(std::vector<framework::LoDTensor>, _lod_tensors)

REGISTER_TYPE_ID(framework::BlockDesc *, _p_block)
REGISTER_TYPE_ID(framework::Tensor *, _p_tensor)
REGISTER_TYPE_ID(framework::LoDTensor *, _p_lod_tensor)
REGISTER_TYPE_ID(std::vector<framework::BlockDesc *>, _p_blocks)
REGISTER_TYPE_ID(std::vector<framework::Tensor *>, _p_tensors)
REGISTER_TYPE_ID(std::vector<framework::LoDTensor *>, _p_lod_tensors)

REGISTER_TYPE_ID(std::vector<framework::Scope *>, _scopes);
REGISTER_TYPE_ID(framework::SelectedRows, _selected_rows)
REGISTER_TYPE_ID(framework::Dim<0>, _dim0)
REGISTER_TYPE_ID(framework::Dim<1>, _dim1)
REGISTER_TYPE_ID(framework::Dim<2>, _dim2)
REGISTER_TYPE_ID(framework::Dim<3>, _dim3)
REGISTER_TYPE_ID(framework::Dim<4>, _dim4)
REGISTER_TYPE_ID(framework::Dim<5>, _dim5)
REGISTER_TYPE_ID(framework::Dim<6>, _dim6)
REGISTER_TYPE_ID(framework::Dim<7>, _dim7)
REGISTER_TYPE_ID(framework::Dim<8>, _dim8)
REGISTER_TYPE_ID(framework::Dim<9>, _dim9)

}  // namespace paddle_mobile

namespace std {

template <>
struct hash<paddle_mobile::kTypeId_t> {
  size_t operator()(const paddle_mobile::kTypeId_t &t) const {
    return std::hash<int>{}(static_cast<int>(t));
  }
};

}  // namespace std
