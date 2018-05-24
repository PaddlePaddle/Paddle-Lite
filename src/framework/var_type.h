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
#include "framework.pb.h"
#include "lod_tensor.h"
#include "selected_rows.h"
#include "variable.h"

namespace paddle_mobile {
namespace framework {
inline proto::VarType::Type ToVarType(std::type_index type) {
  if (type.hash_code() == typeid(LoDTensor).hash_code()) {
    return proto::VarType_Type_LOD_TENSOR;
  } else if (type.hash_code() == typeid(SelectedRows).hash_code()) {
    return proto::VarType_Type_SELECTED_ROWS;
  } else {
    //    PADDLE_THROW("ToVarType:Unsupported type %s",
    //    type.name());
  }
}

}  // namespace framework
}  // namespace paddle_mobile
