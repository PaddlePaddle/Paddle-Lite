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

#include "lite/core/mir/fusion/interpolate_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/mir/fusion/interpolate_fuser.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void InterpolateFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::vector<std::string> Interpolate_type_cases{"bilinear_interp",
                                                  "nearest_interp"};
  for (auto type_ : Interpolate_type_cases) {
    fusion::InterpolateFuser interp_fuser(type_);
    interp_fuser(graph.get());

    fusion::InterpolateFuser2 interp_fuser2(type_);
    interp_fuser2(graph.get());
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_interpolate_fuse_pass,
                  paddle::lite::mir::InterpolateFusePass)
    .BindTargets({TARGET(kAny)});
