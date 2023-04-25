// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/target_wrapper.h"
#include "lite/utils/env.h"
namespace paddle {
namespace lite {
namespace mir {

class Support0DimTensor : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph> &graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
