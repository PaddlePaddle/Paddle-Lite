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

#include "lite/core/optimizer/mir/adaptive_1x1_pool2d_convert_global_pass.h"
#include <set>
#include "lite/core/optimizer/mir/graph_visualize_pass.h"
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher.h"
#include "lite/model_parser/cpp_desc.h"

namespace paddle {
namespace lite {
namespace mir {

void Adaptive1x1Pool2dConvertGlobalPass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  auto check_adaptive_1x1 = [](Node* p) -> bool {
    auto op_desc = p->stmt()->mutable_op_info();
    if (!op_desc->HasAttr("adaptive")) {
      VLOG(1) << "skip. Pool has no attribute named adaptive.";
      return false;
    }

    auto ksize = op_desc->GetAttr<std::vector<int>>("ksize");
    auto ksize_one = ksize[0] == ksize[1] && ksize[0] == 1;
    auto global_pooling = op_desc->GetAttr<bool>("global_pooling");
    auto adaptive = op_desc->GetAttr<bool>("adaptive");
    VLOG(1) << "check adaptive:" << adaptive;
    VLOG(1) << "check global_pooling:" << global_pooling;
    VLOG(1) << "check ksize:" << ksize[0] << "," << ksize[1]
            << " | ksize_one:" << ksize_one;
    if (adaptive && ksize_one && !global_pooling) {
      return true;
    }
    return false;
  };

  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (op_node->AsStmt().op_type() == "pool2d") {
      Node* pool = op_node;
      bool is_adaptive_1x1 = check_adaptive_1x1(pool);
      if (is_adaptive_1x1) {
        // modify its `global pooling` attribute
        auto op_desc = pool->stmt()->mutable_op_info();
        op_desc->SetAttr<bool>("global_pooling", true);
        op_desc->SetAttr<bool>("adaptive", false);
        // read && check
        VLOG(1) << "check adaptive:" << op_desc->GetAttr<bool>("adaptive");
        VLOG(1) << "check global_pooling:"
                << op_desc->GetAttr<bool>("global_pooling");
        VLOG(1) << "check ksize:"
                << op_desc->GetAttr<std::vector<int>>("ksize")[0] << ","
                << op_desc->GetAttr<std::vector<int>>("ksize")[1];
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(adaptive_1x1_pool2d_convert_global_pass,
                  paddle::lite::mir::Adaptive1x1Pool2dConvertGlobalPass)
    .BindTargets({TARGET(kARM)});
