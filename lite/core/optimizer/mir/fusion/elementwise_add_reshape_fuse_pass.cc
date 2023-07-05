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

#include "lite/core/optimizer/mir/fusion/elementwise_add_reshape_fuse_pass.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/fusion/elementwise_add_reshape_fuser.h"
#include "lite/core/optimizer/mir/graph_visualize_pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void ElementwiseReshapeFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // initialze fuser params
  std::vector<std::string> elt_types{"elementwise_add"};
  std::vector<std::string> reshape_type{"reshape2"};

  // start fuse using params
  for (auto elt_type : elt_types) {
    for (auto rs_type : reshape_type) {
      fusion::ElementwiseReshapeFuser fuser(elt_type, rs_type);
      fuser(graph.get());
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_elementwise_reshape_fuse_pass,
                  paddle::lite::mir::ElementwiseReshapeFusePass)
    .BindTargets({TARGET(kXPU)});
