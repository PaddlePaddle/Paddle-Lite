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

#include "lite/core/mir/fusion/shuffle_channel_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/mir/fusion/shuffle_channel_fuser.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void ShuffleChannelFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  for (std::string reshape_type : {"reshape", "reshape2"}) {
    for (std::string transpose_type : {"transpose", "transpose2"}) {
      for (std::string sub_structure : {"r_t_r", "s_t_r"}) {
        fusion::ShuffleChannelFuser fuser(
            reshape_type, transpose_type, sub_structure);
        fuser(graph.get());
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_shuffle_channel_fuse_pass,
                  paddle::lite::mir::ShuffleChannelFusePass)
    .BindTargets({TARGET(kAny)})
    .BindKernel("shuffle_channel");
