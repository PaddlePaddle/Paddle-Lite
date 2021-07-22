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

#include <set>
#include <string>
#include "lite/core/mir/pass.h"

namespace paddle {
namespace lite {

// Query if the specified kernel has been registered.
bool KernelRegistered(const std::string name, const Place& place);

// Check if the pass hits the hardware target.
bool PassMatchesTarget(const mir::Pass& pass,
                       const std::set<TargetType>& targets);

// Check if the pass hits all necessary operators.
bool PassMatchesKernels(const mir::Pass& pass);

}  // namespace lite
}  // namespace paddle
