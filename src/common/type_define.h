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
#include <map>
#include <string>
#include <vector>
#include "framework/attribute.h"
#include "framework/scope.h"

namespace paddle_mobile {

namespace framework {
template <typename Dtype>
class OperatorBase;
class OpDesc;
class BlockDesc;
class InferShapeContext;
}  // namespace framework

using VariableNameMap = std::map<std::string, std::vector<std::string>>;

template <typename Dtype>
using OpCreator = std::function<framework::OperatorBase<Dtype> *(
    const std::string & /*type*/, const VariableNameMap & /*inputs*/,
    const VariableNameMap & /*outputs*/,
    const framework::AttributeMap & /*attrs*/, framework::Scope * /*scope*/)>;

using InferVarTypeFN = std::function<void(const framework::OpDesc & /*op_desc*/,
                                          framework::BlockDesc * /*block*/)>;

using InferShapeFN = std::function<void(framework::InferShapeContext *)>;
};  // namespace paddle_mobile
