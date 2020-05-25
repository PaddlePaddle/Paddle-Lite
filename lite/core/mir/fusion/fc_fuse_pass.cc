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

#include "lite/core/mir/fusion/fc_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/mir/fusion/fc_fuser.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void FcFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
#ifdef LITE_WITH_X86
  fusion::FcFuser fuser(true);
  fuser(graph.get());
#endif

  fusion::FcFuser fuser2(false);
  fuser2(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_fc_fuse_pass, paddle::lite::mir::FcFusePass)
    .BindTargets({TARGET(kAny)})
    .ExcludeTargets({TARGET(kXPU), TARGET(kX86)})
    .ExcludeTargets({TARGET(kBM)})
    .ExcludeTargets({TARGET(kCUDA)})
    .BindKernel("fc");
