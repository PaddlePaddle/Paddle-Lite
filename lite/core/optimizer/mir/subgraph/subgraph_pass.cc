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

#include "lite/core/optimizer/mir/subgraph/subgraph_pass.h"
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/subgraph/subgraph_detector.h"
#include "lite/utils/env.h"

namespace paddle {
namespace lite {
namespace mir {

void NPUSubgraphPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::set<std::string> supported_lists;
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

void HuaweiAscendNPUSubgraphPass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  std::set<std::string> supported_lists;
#define USE_SUBGRAPH_BRIDGE(op_type, target) supported_lists.insert(#op_type);
#include "lite/kernels/huawei_ascend_npu/bridges/paddle_use_bridges.h"
#undef USE_SUBGRAPH_BRIDGE
  auto teller = [&](Node* node) {
    if (!node->IsStmt()) return false;
    auto& stmt = node->AsStmt();
    return supported_lists.count(stmt.op_type()) != 0;
  };
  SubgraphFuser fuser(graph.get(), teller, 1 /* min_subgraph_size */);
  fuser();
}

void APUSubgraphPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::set<std::string> supported_lists;
#define USE_SUBGRAPH_BRIDGE(op_type, target) \
  supported_lists.insert(#op_type);          \
  LOG(INFO) << #op_type
#include "lite/kernels/apu/bridges/paddle_use_bridges.h"
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
  std::set<std::string> supported_lists;
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
  std::set<std::string> supported_lists;
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
  std::set<std::string> supported_lists;
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
  std::set<std::string> supported_lists;
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

void ImaginationNNASubgraphPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::set<std::string> supported_lists;
#define USE_SUBGRAPH_BRIDGE(op_type, target) supported_lists.insert(#op_type);
#include "lite/kernels/imagination_nna/bridges/paddle_use_bridges.h"
#undef USE_SUBGRAPH_BRIDGE
  auto teller = [&](Node* node) {
    if (!node->IsStmt()) return false;
    auto& stmt = node->AsStmt();
    return supported_lists.count(stmt.op_type()) != 0;
  };
  SubgraphFuser fuser(graph.get(), teller, 1 /* min_subgraph_size */);
  fuser();
}

void NNAdapterSubgraphPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  auto has_intersection = [](const std::vector<std::string>& a,
                             const std::vector<std::string>& b) -> bool {
    std::set<std::string> a_set(a.begin(), a.end());
    std::set<std::string> b_set(b.begin(), b.end());
    std::set<std::string> c_set;
    std::set_intersection(a_set.begin(),
                          a_set.end(),
                          b_set.begin(),
                          b_set.end(),
                          std::inserter(c_set, c_set.begin()));
    return !c_set.empty();
  };
  // Filter the supported operators for the selected devices according to the
  // registered op bridges
  std::vector<std::string> selected_device_names;
#if defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || defined(LITE_WITH_PYTHON) || \
    defined(LITE_WITH_NNADAPTER)
  Scope* scope = nullptr;
  for (auto& any_op_node : graph->StmtTopologicalOrder()) {
    scope = any_op_node->AsStmt().op()->scope();
    if (scope) break;
  }
  CHECK(scope != nullptr);
  selected_device_names =
      Context<TargetType::kNNAdapter>::NNAdapterDeviceNames(scope);
#endif
  std::set<std::string> supported_ops;
  std::vector<std::string> supported_device_names;
  std::string device_names;
#define USE_SUBGRAPH_BRIDGE(op_type_, target_, device_names_)            \
  device_names = device_names_;                                          \
  device_names.erase(                                                    \
      std::remove(device_names.begin(), device_names.end(), ' '),        \
      device_names.end());                                               \
  supported_device_names = Split(device_names, ",");                     \
  if (has_intersection(selected_device_names, supported_device_names)) { \
    supported_ops.insert(#op_type_);                                     \
  }
#include "lite/kernels/nnadapter/bridges/paddle_use_bridges.h"
#undef USE_SUBGRAPH_BRIDGE
  auto teller = [&](Node* node) {
    if (!node->IsStmt()) return false;
    auto& stmt = node->AsStmt();
    return supported_ops.count(stmt.op_type()) != 0;
  };
  SubgraphFuser fuser(graph.get(), teller, 1 /* min_subgraph_size */);
  fuser();
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(npu_subgraph_pass, paddle::lite::mir::NPUSubgraphPass)
    .BindTargets({TARGET(kNPU)});
REGISTER_MIR_PASS(huawei_ascend_npu_subgraph_pass,
                  paddle::lite::mir::HuaweiAscendNPUSubgraphPass)
    .BindTargets({TARGET(kHuaweiAscendNPU)});
REGISTER_MIR_PASS(apu_subgraph_pass, paddle::lite::mir::APUSubgraphPass)
    .BindTargets({TARGET(kAPU)});
REGISTER_MIR_PASS(xpu_subgraph_pass, paddle::lite::mir::XPUSubgraphPass)
    .BindTargets({TARGET(kXPU)});
REGISTER_MIR_PASS(bm_subgraph_pass, paddle::lite::mir::BMSubgraphPass)
    .BindTargets({TARGET(kBM)});
REGISTER_MIR_PASS(rknpu_subgraph_pass, paddle::lite::mir::RKNPUSubgraphPass)
    .BindTargets({TARGET(kRKNPU)});
REGISTER_MIR_PASS(mlu_subgraph_pass, paddle::lite::mir::MLUSubgraphPass)
    .BindTargets({TARGET(kMLU)});
REGISTER_MIR_PASS(imagination_nna_subgraph_pass,
                  paddle::lite::mir::ImaginationNNASubgraphPass)
    .BindTargets({TARGET(kImaginationNNA)});
REGISTER_MIR_PASS(nnadapter_subgraph_pass,
                  paddle::lite::mir::NNAdapterSubgraphPass)
    .BindTargets({TARGET(kNNAdapter)});
