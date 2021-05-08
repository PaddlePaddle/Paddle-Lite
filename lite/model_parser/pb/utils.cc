// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/model_parser/pb/utils.h"

namespace paddle {
namespace lite {
namespace pb {

lite::VarDataType ConvertVarType(
    ::paddle::framework::proto::VarType_Type pb_type) {
  typedef ::paddle::framework::proto::VarType_Type VarType_Type;
  lite::VarDataType type{};
  switch (pb_type) {
#define CASE(vtype)                        \
  case VarType_Type::VarType_Type_##vtype: \
    type = lite::VarDataType::vtype;       \
    break
    CASE(FP64);
    CASE(FP32);
    CASE(INT8);
    CASE(UINT8);
    CASE(INT16);
    CASE(INT32);
    CASE(INT64);
#undef CASE
    default:
      LOG(FATAL) << "unknown type " << pb_type;
  }
  return type;
}

::paddle::framework::proto::VarType_Type ConvertVarType(
    lite::VarDataType var_type) {
  typedef ::paddle::framework::proto::VarType_Type VarType_Type;
  VarType_Type type{};
  switch (var_type) {
#define CASE(vtype)                            \
  case lite::VarDataType::vtype:               \
    type = VarType_Type::VarType_Type_##vtype; \
    break
    CASE(FP64);
    CASE(FP32);
    CASE(FP16);
    CASE(INT8);
    CASE(UINT8);
    CASE(INT16);
    CASE(INT32);
    CASE(INT64);
#undef CASE
    default:
      LOG(FATAL) << "unknown type " << static_cast<int>(var_type);
  }
  return type;
}

}  // namespace pb
}  // namespace lite
}  // namespace paddle
