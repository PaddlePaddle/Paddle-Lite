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

#include "lite/core/mir/subgraph/subgraph_pass.h"
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/subgraph/subgraph_detector.h"
#include "lite/utils/env.h"

namespace paddle {
namespace lite {
namespace mir {

void NPUSubgraphPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::unordered_set<std::string> supported_lists;
#define USE_SUBGRAPH_BRIDGE(op_type, target) supported_lists.insert(#op_type);
#include "lite/kernels/npu/bridges/paddle_use_bridges.h"
#undef USE_SUBGRAPH_BRIDGE
  auto teller = [&](Node* node) {
    if (!node->IsStmt()) return false;
    auto& stmt = node->AsStmt();
    return supported_lists.count(stmt.op_type()) != 0;
  };
  SubgraphFuser fuser(graph.get(), teller, 1 /* min_subgraph_size */);
  fuser();
}

void XPUSubgraphPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  if (!GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
  std::unordered_set<std::string> supported_lists;
#define USE_SUBGRAPH_BRIDGE(op_type, target) supported_lists.insert(#op_type);
#include "lite/kernels/xpu/bridges/paddle_use_bridges.h"
#undef USE_SUBGRAPH_BRIDGE
  auto teller = [&](Node* node) {
    if (!node->IsStmt()) return false;
    auto& stmt = node->AsStmt();
    return supported_lists.count(stmt.op_type()) != 0;
  };
  SubgraphFuser fuser(graph.get(), teller, 1 /* min_subgraph_size */);
  fuser();
}

void BMSubgraphPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::unordered_set<std::string> supported_lists;
#define USE_SUBGRAPH_BRIDGE(op_type, target) supported_lists.insert(#op_type);
#include "lite/kernels/bm/bridges/paddle_use_bridges.h"
#undef USE_SUBGRAPH_BRIDGE
  auto teller = [&](Node* node) {
    if (!node->IsStmt()) return false;
    auto& stmt = node->AsStmt();
    return supported_lists.count(stmt.op_type()) != 0;
  };
  SubgraphFuser fuser(graph.get(), teller, 1 /* min_subgraph_size */);
  fuser();
}

void RKNPUSubgraphPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::unordered_set<std::string> supported_lists;
#define USE_SUBGRAPH_BRIDGE(op_type, target) supported_lists.insert(#op_type);
#include "lite/kernels/rknpu/bridges/paddle_use_bridges.h"
#undef USE_SUBGRAPH_BRIDGE
  auto teller = [&](Node* node) {
    if (!node->IsStmt()) return false;
    auto& stmt = node->AsStmt();
    return supported_lists.count(stmt.op_type()) != 0;
  };
  SubgraphFuser fuser(graph.get(), teller, 1 /* min_subgraph_size */);
  fuser();
}

void MLUSubgraphPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
#ifdef LITE_WITH_MLU
  // remove invalid places, since only support X86, host, MLU
  auto v_places = graph->valid_places();
  for (auto it = v_places.begin(); it != v_places.end();) {
    if (it->target != TARGET(kMLU) && it->target != TARGET(kHost) &&
        it->target != TARGET(kX86)) {
      it = v_places.erase(it);
    } else {
      ++it;
    }
  }
  // add x86 NHWC place
  std::vector<paddle::lite_api::PrecisionType> precisions{PRECISION(kFloat),
                                                          PRECISION(kFP16)};
  if (lite::TargetWrapperMlu::UseFirstConv())
    precisions.emplace_back(PRECISION(kInt8));
  for (auto& prec : precisions) {
    auto is_x86_nhwc = [prec](const Place& it) {
      return it.layout == DATALAYOUT(kNHWC) && it.target == TARGET(kX86) &&
             it.precision == prec;
    };
    if (std::find_if(v_places.cbegin(), v_places.cend(), is_x86_nhwc) ==
        v_places.end()) {
      v_places.emplace_back(Place{TARGET(kX86), prec, DATALAYOUT(kNHWC)});
    }
  }
  graph->SetValidPlaces(v_places);
  VLOG(4) << "valid places after modified:";
  for (auto& p : v_places) {
    VLOG(4) << p.DebugString();
  }
#endif

  std::unordered_set<std::string> supported_lists;
#define USE_SUBGRAPH_BRIDGE(op_type, target) supported_lists.insert(#op_type);
#include "lite/kernels/mlu/bridges/paddle_use_bridges.h"
#undef USE_SUBGRAPH_BRIDGE
  auto teller = [&](Node* node) {
    if (!node->IsStmt()) return false;
    auto& stmt = node->AsStmt();
    return supported_lists.count(stmt.op_type()) != 0;
  };
  SubgraphFuser fuser(graph.get(), teller, 1 /* min_subgraph_size */);
  fuser();
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(npu_subgraph_pass, paddle::lite::mir::NPUSubgraphPass)
    .BindTargets({TARGET(kNPU)});
REGISTER_MIR_PASS(xpu_subgraph_pass, paddle::lite::mir::XPUSubgraphPass)
    .BindTargets({TARGET(kXPU)});
REGISTER_MIR_PASS(bm_subgraph_pass, paddle::lite::mir::BMSubgraphPass)
    .BindTargets({TARGET(kBM)});
REGISTER_MIR_PASS(rknpu_subgraph_pass, paddle::lite::mir::RKNPUSubgraphPass)
    .BindTargets({TARGET(kRKNPU)});
REGISTER_MIR_PASS(mlu_subgraph_pass, paddle::lite::mir::MLUSubgraphPass)
    .BindTargets({TARGET(kMLU)});
