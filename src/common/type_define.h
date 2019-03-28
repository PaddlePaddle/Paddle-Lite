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

#include <string>
#include <vector>

namespace paddle_mobile {

template <typename T>
struct TypeIdWrapper {
  std::string type();
};

template <typename T>
struct type_id {
  const std::string type_ = TypeIdWrapper<T>().type();

  template <typename OtherType>
  bool operator==(const type_id<OtherType> &operand) {
    return this->name() == operand.name();
  }

  const std::string name() { return type_; }
};

namespace framework {
class BlockDesc;
class Tensor;
class LoDTensor;
class SelectedRows;
class Scope;

template <int>
struct Dim;
}  // namespace framework

#define REGISTER_TYPE_ID(Type, TypeName)                  \
  template <>                                             \
  struct TypeIdWrapper<Type> {                            \
    std::string type() { return std::string(#TypeName); } \
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
