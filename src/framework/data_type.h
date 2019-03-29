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
#include "common/enforce.h"
#include "common/type_define.h"
#include "framework/framework.pb-c.h"

namespace paddle_mobile {

namespace framework {

_PaddleMobile__Framework__Proto__VarType__Type ToDataType(kTypeId_t type);

kTypeId_t ToTypeIndex(_PaddleMobile__Framework__Proto__VarType__Type type);

inline _PaddleMobile__Framework__Proto__VarType__Type ToDataType(int type) {
  return static_cast<_PaddleMobile__Framework__Proto__VarType__Type>(type);
}

template <typename Visitor>
inline void VisitDataType(_PaddleMobile__Framework__Proto__VarType__Type type,
                          Visitor visitor) {
  switch (type) {
    // case PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__FP16:
    //   visitor.template apply<float16>();
    //   break;
    case PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__FP32:
      visitor.template apply<float>();
      break;
    case PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__FP64:
      visitor.template apply<double>();
      break;
    case PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__INT32:
      visitor.template apply<int>();
      break;
    case PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__INT64:
      visitor.template apply<int64_t>();
      break;
    case PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__BOOL:
      visitor.template apply<bool>();
      break;
    case PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__UINT8:
      visitor.template apply<uint8_t>();
      break;
    case PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__INT16:
      visitor.template apply<int16_t>();
      break;
    case PADDLE_MOBILE__FRAMEWORK__PROTO__VAR_TYPE__TYPE__INT8:
      visitor.template apply<int8_t>();
      break;
    default:
      PADDLE_MOBILE_THROW_EXCEPTION("Not supported %d", type);
  }
}

extern std::string DataTypeToString(
    const _PaddleMobile__Framework__Proto__VarType__Type type);
inline std::ostream& operator<<(
    std::ostream& out,
    const _PaddleMobile__Framework__Proto__VarType__Type& type) {
  out << DataTypeToString(type);
  return out;
}

}  // namespace framework
}  // namespace paddle_mobile
