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

#pragma once

#include "lite/core/framework.pb.h"
#include "lite/model_parser/base/traits.h"
#include "lite/utils/logging.h"
namespace paddle {
namespace lite {
namespace pb {

lite::VarDataType ConvertVarType(
    ::paddle::framework::proto::VarType_Type pb_type);

::paddle::framework::proto::VarType_Type ConvertVarType(
    lite::VarDataType var_type);

}  // namespace pb
}  // namespace lite
}  // namespace paddle
