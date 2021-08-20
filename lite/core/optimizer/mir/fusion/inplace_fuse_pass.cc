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

#include "lite/core/optimizer/mir/fusion/inplace_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/fusion/inplace_fuser.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void InplaceFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::vector<std::string> inplace_type_cases{"reshape",
                                              "reshape2",
                                              "flatten",
                                              "flatten2",
                                              "squeeze",
                                              "squeeze2",
                                              "unsqueeze",
                                              "unsqueeze2"};
  for (auto type : inplace_type_cases) {
    fusion::InplaceFuser inplace_fuser(type);
    inplace_fuser(graph.get());
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_inplace_fuse_pass, paddle::lite::mir::InplaceFusePass)
    .BindTargets({TARGET(kAny)})
    .ExcludeTargets({TARGET(kNPU)});
