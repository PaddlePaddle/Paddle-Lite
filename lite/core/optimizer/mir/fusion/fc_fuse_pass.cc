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
#include <list>
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/fusion/fc_fuser.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void FcFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::vector<std::string> mul_types{"mul"};
  std::vector<std::string> act_types;
  bool has_int8 = false;
  bool has_arm = false;
  bool has_weight_quant = false;
  bool is_nnadapter = false;
  for (auto& place : graph->valid_places()) {
    if (place.target != TARGET(kMLU)) {
      act_types.push_back("relu");
    }
    if (place.target == TARGET(kARM)) {
      has_arm = true;
      act_types.push_back("relu6");
      if (place.precision == PRECISION(kInt8)) {
        has_int8 = true;
      }
    }
    if (place.target == TARGET(kNNAdapter)) {
      is_nnadapter = true;
    }
  }
  act_types.push_back("");
  const std::list<mir::Node>& nodes = graph->nodes();
  for (auto& node : nodes) {
    if (node.IsStmt()) {
      auto* op_info = (node.stmt())->op_info();
      bool enable_int8 = op_info->HasAttr("enable_int8") ? true : false;
      if (enable_int8) {
        has_weight_quant = true;
        break;
      }
    }
  }
  if (!(has_int8 && has_weight_quant) && has_arm && !is_nnadapter) {
    // only support FP32/FP16
    mul_types.push_back("matmul");
    mul_types.push_back("matmul_v2");
  }
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
