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
#include <typeindex>

#include "framework/program/tensor_desc.h"

namespace paddle_mobile {
namespace framework {

inline VarType_Type ToDataType(std::type_index type) {
  /*if (typeid(platform::float16).hash_code() == type.hash_code()) {
    return proto::VarType::FP16;
  } else */
  if (typeid(const float).hash_code() == type.hash_code()) {
    // CPPLint complains Using C-style cast.  Use
    // static_cast<float>() instead
    // One fix to this is to replace float with const float because
    // typeid(T) == typeid(const T)
    // http://en.cppreference.com/w/cpp/language/typeid
    return VARTYPE_TYPE_FP32;
  } else if (typeid(const double).hash_code() == type.hash_code()) {
    return VARTYPE_TYPE_FP64;
  } else if (typeid(const int).hash_code() == type.hash_code()) {
    return VARTYPE_TYPE_INT32;
  } else if (typeid(const int64_t).hash_code() == type.hash_code()) {
    return VARTYPE_TYPE_INT64;
  } else if (typeid(const bool).hash_code() == type.hash_code()) {
    return VARTYPE_TYPE_BOOL;
  } else {
    //    PADDLE_THROW("Not supported");
    //    std::cout << "Not supported";
  }
}

inline std::type_index ToTypeIndex(VarType_Type type) {
  switch (type) {
    //    case proto::VarType::FP16:
    //      return typeid(platform::float16);
    case VARTYPE_TYPE_FP32:
      return typeid(float);
    case VARTYPE_TYPE_FP64:
      return typeid(double);
    case VARTYPE_TYPE_INT32:
      return typeid(int);
    case VARTYPE_TYPE_INT64:
      return typeid(int64_t);
    case VARTYPE_TYPE_BOOL:
      return typeid(bool);
    default:
      //      PADDLE_THROW("Not support type %d", type);
      printf("Not support type %d", type);
  }
}

template <typename Visitor>
inline void VisitDataType(VarType_Type type, Visitor visitor) {
  switch (type) {
    //    case proto::VarType::FP16:
    //      visitor.template operator()<platform::float16>();
    //      break;
    case VARTYPE_TYPE_FP32:
      visitor.template operator()<float>();
      break;
    case VARTYPE_TYPE_FP64:
      visitor.template operator()<double>();
      break;
    case VARTYPE_TYPE_INT32:
      visitor.template operator()<int>();
      break;
    case VARTYPE_TYPE_INT64:
      visitor.template operator()<int64_t>();
      break;
    case VARTYPE_TYPE_BOOL:
      visitor.template operator()<bool>();
      break;
    default:
      //      PADDLE_THROW("Not supported");
      printf("Not supported");
  }
}

inline std::string DataTypeToString(const VarType_Type type) {
  switch (type) {
    case VARTYPE_TYPE_FP16:
      return "float16";
    case VARTYPE_TYPE_FP32:
      return "float32";
    case VARTYPE_TYPE_FP64:
      return "float64";
    case VARTYPE_TYPE_INT16:
      return "int16";
    case VARTYPE_TYPE_INT32:
      return "int32";
    case VARTYPE_TYPE_INT64:
      return "int64";
    case VARTYPE_TYPE_BOOL:
      return "bool";
    default:
      //      PADDLE_THROW("Not support type %d", type);
      printf("Not support type %d", type);
  }
}

inline std::ostream &operator<<(std::ostream &out,
                                const VarType_Type &type) {
  out << DataTypeToString(type);
  return out;
}

}  // namespace framework
}  // namespace paddle_mobile
