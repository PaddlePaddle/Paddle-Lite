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

#include "lite/core/optimizer/mir/fusion/matmul_elementwise_add_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/fusion/matmul_elementwise_add_fuser.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void MatmulElementwiseAddFusePass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  for (auto& place : graph->valid_places()) {
    if (place.precision == PRECISION(kInt8)) {
      return;
    }
  }
#if defined(LITE_WITH_X86) || defined(LITE_WITH_CUDA) || defined(LITE_WITH_ARM)
#ifdef LITE_WITH_MLU
  fusion::MatmulElementwiseAddFuser fuser(false, graph);
  fuser(graph.get());
#else
  fusion::MatmulElementwiseAddFuser fuser(true, graph);
  fuser(graph.get());
#endif
#endif
  fusion::MatmulElementwiseAddFuser fuser2(false, graph);
  fuser2(graph.get());
#ifdef LITE_WITH_FPGA
  fusion::MatmulElementwiseAddFuser fpga_fuser(true, graph);
  fpga_fuser(graph.get());
#endif
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_matmul_element_add_fuse_pass,
                  paddle::lite::mir::MatmulElementwiseAddFusePass)
    .BindTargets({TARGET(kAny)})
    .ExcludeTargets({TARGET(kXPU)})
    .BindKernel("fc");
