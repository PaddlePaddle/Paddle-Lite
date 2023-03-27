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

#include "lite/core/optimizer/mir/fusion/transformer_attention_fuse_pass.h"
#include <list>
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/fusion/transformer_attention_fuser.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void TransformerAttentionFusePass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  bool has_int8 = false;
  for (auto& place : graph->valid_places()) {
    if (place.precision == PRECISION(kInt8)) {
      has_int8 = true;
    }
  }
  std::vector<bool> reshape_has_xshapes = {false, true};
  std::vector<bool> transpose_has_xshapes = {false, true};
  std::vector<bool> dropout_masks = {false, true};
  std::vector<std::string> mul_types = {"matmul", "matmul_v2"};
  for (auto reshape_has_xshape : reshape_has_xshapes) {
    for (auto transpose_has_xshape : transpose_has_xshapes) {
      for (auto dropout_mask : dropout_masks) {
        for (auto mul_type : mul_types) {
          fusion::TransformerAttentionFuser fuser(
              reshape_has_xshape, transpose_has_xshape, dropout_mask, mul_type);
          if ((has_int8)) {
            fuser(graph.get());
          }
        }
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(transformer_attention_fuse_pass,
                  paddle::lite::mir::TransformerAttentionFusePass)
    .BindTargets({TARGET(kARM)})
    .ExcludeTargets(
        {TARGET(kXPU), TARGET(kOpenCL), TARGET(kMetal), TARGET(kNNAdapter)})
    .BindKernel("fused_attention");
