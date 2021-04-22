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
#if defined(LITE_WITH_X86) || defined(LITE_WITH_CUDA)
#ifdef LITE_WITH_MLU
  fusion::FcFuser fuser(false);
  fuser(graph.get());
#else
  fusion::FcFuser fuser(true);
  fuser(graph.get());
#endif
#endif
  fusion::FcFuser fuser2(false);
  fuser2(graph.get());
#ifdef LITE_WITH_FPGA
  fusion::FcFuser fpga_fuser(true);
  fpga_fuser(graph.get());
#endif
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_fc_fuse_pass, paddle::lite::mir::FcFusePass)
    .BindTargets({TARGET(kAny)})
    .ExcludeTargets({TARGET(kXPU)})
#if (!defined(LITE_WITH_MLU) && !defined(LITE_WITH_HUAWEI_ASCEND_NPU))
    .ExcludeTargets({TARGET(kX86)})
#endif
    .ExcludeTargets({TARGET(kBM)})
    .BindKernel("fc");
