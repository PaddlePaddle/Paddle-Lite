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

#include "framework/data_type.h"
#include <stdint.h>
#include <string>
#include <unordered_map>
#include "common/type_define.h"

namespace paddle_mobile {
namespace framework {

struct DataTypeMap {
  std::unordered_map<std::string,
                     _PaddleMobile__Framework__Proto__VarType__Type>
      cpp_to_proto_;
  std::unordered_map<int, std::string> proto_to_cpp_;
  std::unordered_map<int, std::string> proto_to_str_;
  std::unordered_map<std::string, size_t> cpp_to_size_;
};

static DataTypeMap* InitDataTypeMap();
// C++11 removes the need for manual locking. Concurrent execution shall wait if
// a static local variable is already being initialized.
// https://stackoverflow.com/questions/11711920/how-to-implement-multithread-safe-singleton-in-c11-without-using-mutex
static DataTypeMap& gDataTypeMap() {
  static DataTypeMap* g_data_type_map_ = InitDataTypeMap();
  return *g_data_type_map_;
}

template <typename T>
static inline void RegisterType(
    DataTypeMap* map, _PaddleMobile__Framework__Proto__VarType__Type proto_type,
    const std::string& name) {
  map->proto_to_cpp_.emplace(static_cast<int>(proto_type), type_id<T>().name());
  map->cpp_to_proto_.emplace(type_id<T>().name(), proto_type);
  map->proto_to_str_.emplace(static_cast<int>(proto_type), name);
  map->cpp_to_size_.emplace(type_id<T>().name(), sizeof(T));
}

static DataTypeMap* InitDataTypeMap() {
  auto retv = new DataTypeMap();

#define RegType(cc_type, proto_type) \
  RegisterType<cc_type>(retv, proto_type, #cc_type)

  // NOTE: Add your customize type here.
  // RegType(float16, PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__FP16);
  RegType(float, PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__FP32);
  RegType(double, PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__FP64);
  RegType(int, PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__INT32);
  RegType(int64_t, PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__INT64);
  RegType(bool, PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__BOOL);
  RegType(size_t, PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__SIZE_T);
  RegType(int16_t, PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__INT16);
  RegType(uint8_t, PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__UINT8);
  RegType(int8_t, PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__INT8);

#undef RegType
  return retv;
}

_PaddleMobile__Framework__Proto__VarType__Type ToDataType(std::string type) {
  auto it = gDataTypeMap().cpp_to_proto_.find(type);
  if (it != gDataTypeMap().cpp_to_proto_.end()) {
    return it->second;
  }
  PADDLE_MOBILE_THROW_EXCEPTION("Not support %s as tensor type", type.c_str());
}

std::string ToTypeIndex(_PaddleMobile__Framework__Proto__VarType__Type type) {
  auto it = gDataTypeMap().proto_to_cpp_.find(static_cast<int>(type));
  if (it != gDataTypeMap().proto_to_cpp_.end()) {
    return it->second;
  }
  PADDLE_MOBILE_THROW_EXCEPTION(
      "Not support _PaddleMobile__Framework__Proto__VarType__Type(%d) as "
      "tensor type",
      static_cast<int>(type));
}

std::string DataTypeToString(
    const _PaddleMobile__Framework__Proto__VarType__Type type) {
  auto it = gDataTypeMap().proto_to_str_.find(static_cast<int>(type));
  if (it != gDataTypeMap().proto_to_str_.end()) {
    return it->second;
  }
  PADDLE_MOBILE_THROW_EXCEPTION(
      "Not support _PaddleMobile__Framework__Proto__VarType__Type(%d) as "
      "tensor type",
      static_cast<int>(type));
}

}  // namespace framework
}  // namespace paddle_mobile
