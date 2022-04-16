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
#include <map>
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

static std::string ReadSubgraphPartitionConfigsFromEnv() {
  std::string configs;
  auto path = GetStringFromEnv(SUBGRAPH_CUSTOM_PARTITION_CONFIG_FILE);
  if (!path.empty()) {
    std::vector<char> buffer;
    if (ReadFile(path, &buffer, false)) {
      if (!buffer.empty()) {
        configs.insert(configs.begin(), buffer.begin(), buffer.end());
      }
    } else {
      LOG(WARNING)
          << "Missing the subgraph custom partition configuration file "
          << path;
    }
  }
  return configs;
}

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
  auto subgraph_partition_configs = ReadSubgraphPartitionConfigsFromEnv();
  SubgraphFuser fuser(graph.get(),
                      teller,
                      1 /* min_subgraph_size */,
                      subgraph_partition_configs);
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
  auto subgraph_partition_configs = ReadSubgraphPartitionConfigsFromEnv();
  SubgraphFuser fuser(graph.get(),
                      teller,
                      1 /* min_subgraph_size */,
                      subgraph_partition_configs);
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
  auto subgraph_partition_configs = ReadSubgraphPartitionConfigsFromEnv();
  SubgraphFuser fuser(graph.get(),
                      teller,
                      1 /* min_subgraph_size */,
                      subgraph_partition_configs);
  fuser();
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(npu_subgraph_pass, paddle::lite::mir::NPUSubgraphPass)
    .BindTargets({TARGET(kNPU)});
REGISTER_MIR_PASS(bm_subgraph_pass, paddle::lite::mir::BMSubgraphPass)
    .BindTargets({TARGET(kBM)});
REGISTER_MIR_PASS(mlu_subgraph_pass, paddle::lite::mir::MLUSubgraphPass)
    .BindTargets({TARGET(kMLU)});
