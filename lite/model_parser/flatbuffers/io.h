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

#include <set>
#include <string>
#include <vector>
#include "lite/core/scope.h"
#include "lite/core/tensor.h"
#include "lite/model_parser/flatbuffers/param_desc.h"
#include "lite/model_parser/flatbuffers/program_desc.h"

namespace paddle {
namespace lite {
namespace fbs {

std::vector<char> LoadFile(const std::string& path,
                           const size_t& offset = 0,
                           const size_t& size = 0);
void SaveFile(const std::string& path, const std::vector<char>& cache);

void SetScopeWithCombinedParams(lite::Scope* scope,
                                const CombinedParamsDescReadAPI& params);

void SetCombinedParamsWithScope(const lite::Scope& scope,
                                const std::set<std::string>& params_name,
                                CombinedParamsDescWriteAPI* params);

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
