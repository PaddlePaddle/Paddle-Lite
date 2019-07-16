// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * This file implements the interface to manipute the protobuf message. We use
 * macros to make a compatible interface with the framework::XXDesc and
 * lite::pb::XXDesc.
 */

#include "lite/core/framework.pb.h"
#include "lite/model_parser/cpp/op_desc.h"
#include "lite/model_parser/naive_buffer/op_desc.h"
#include "lite/model_parser/pb/op_desc.h"

namespace paddle {
namespace lite {

/// Transform an OpDesc from OpDescType to cpp format.
template <typename OpDescType>
void TransformOpDescAnyToCpp(const OpDescType& any_desc, cpp::OpDesc* cpp_desc);

/// Transform an OpDesc from cpp to OpDescType format.
template <typename OpDescType>
void TransformOpDescCppToAny(const cpp::OpDesc& cpp_desc, OpDescType* any_desc);

}  // namespace lite
}  // namespace paddle
