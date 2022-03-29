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

#include "lite/core/optimizer/mir/fusion/fc_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/fusion/fc_fuser.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void FcFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::vector<std::string> mul_types{"mul"};
  std::vector<bool> act_types;
  for (auto& place : graph->valid_places()) {
    if (place.target != TARGET(kMLU)) {
      act_types.push_back(true);
    }
    if (place.target == TARGET(kARM)) {
      mul_types.push_back("matmul");
      mul_types.push_back("matmul_v2");
    }
  }
  act_types.push_back(false);
  for (auto op_type : mul_types) {
    for (auto act_type : act_types) {
      fusion::FcFuser fuser(op_type, act_type);
      fuser(graph.get());
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_fc_fuse_pass, paddle::lite::mir::FcFusePass)
    .BindTargets({TARGET(kAny)})
    .ExcludeTargets({TARGET(kXPU)})
#if (!defined(LITE_WITH_MLU) && !defined(LITE_WITH_NNADAPTER) && \
     !defined(LITE_WITH_METAL) && !defined(LITE_WITH_X86))
    .ExcludeTargets({TARGET(kX86)})
#endif
    .ExcludeTargets({TARGET(kBM)})
    .BindKernel("fc");
