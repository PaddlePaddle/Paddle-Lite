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

namespace paddle_mobile {
namespace framework {

//    inline proto::VarType::Type ToDataType(std::type_index type) {
//        using namespace paddle_mobile::framework::proto;
//        if (typeid(float).hash_code() == type.hash_code()) {
//            return proto::VarType::FP32;
//        } else if (typeid(double).hash_code() == type.hash_code()) {
//            return proto::VarType::FP64;
//        } else if (typeid(int).hash_code() == type.hash_code()) {
//            return proto::VarType::INT32;
//        } else if (typeid(int64_t).hash_code() == type.hash_code()) {
//            return proto::VarType::INT64;
//        } else if (typeid(bool).hash_code() == type.hash_code()) {
//            return proto::VarType::BOOL;
//        } else {
////            PADDLE_THROW("Not supported");
//        }
//    }
}  // namespace framework
}  // namespace paddle_mobile
