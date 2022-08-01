// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

//
// Created by chenyaohuang on 2021/12/17.
//

#include "lite/core/optimizer/mir/fusion/unsqueeze2_pad3d_squeeze2_fuse.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/fusion/unsqueeze2_pad3d_squeeze2_fuse_pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void Unsqueeze2Pad3dSqueeze2FusePass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  VLOG(4) << "start";
  fusion::Unsqueeze2Pad3dSqueeze2Fuser fuser("unsqueeze2", "pad3d", "squeeze2");
  fuser(graph.get());
  VLOG(4) << "end";
}
}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_unsqueeze2_pad3d_squeeze2_fuse_pass,
                  paddle::lite::mir::Unsqueeze2Pad3dSqueeze2FusePass)
    .BindTargets({TARGET(kAny)})
    .ExcludeTargets({TARGET(kXPU), TARGET(kBM)});
