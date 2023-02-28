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

bool NNAdapterSubgraphOpTeller(const std::string& device_name,
                               SSAGraph* graph,
                               Node* node,
                               Scope* scope) {
  // auto op_info = node->AsStmt().op_info();
  // auto op_type = op_info->Type();
  // // Add extra op filter for each device
  // if (device_name == "nvidia_tensorrt") {
  //   if (op_type == "depthwise_conv2d" || op_type == "conv2d") {
  //     auto filter_name = op_info->Input("Filter").front();
  //     auto filter_tensor = scope->FindMutableTensor(filter_name);
  //     auto filter_dims = filter_tensor->dims();
  //     auto filter_width = filter_dims[3];
  //     auto filter_height = filter_dims[2];
  //     if (filter_width != filter_height) return false;
  //     auto output_channel_size = filter_dims[0];
  //     auto groups = op_info->GetAttr<int>("groups");
  //     int multiplier = output_channel_size / groups;
  //     if (groups != 1 && multiplier != 1) return false;
  //   }
  // }
  return true;
}

void NNAdapterSubgraphPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Filter the supported operators for the selected devices according to the
  // registered op converters
  std::string subgraph_partition_configs;
  std::vector<std::string> selected_device_names;
  Scope* scope = nullptr;
  for (auto& any_op_node : graph->StmtTopologicalOrder()) {
    scope = any_op_node->AsStmt().op()->scope();
    if (scope) break;
  }
  CHECK(scope != nullptr);
#if defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || defined(LITE_WITH_PYTHON) || \
    defined(LITE_WITH_NNADAPTER)
  selected_device_names =
      Context<TargetType::kNNAdapter>::NNAdapterDeviceNames(scope);
  // Load the partition configurations from APIs
  subgraph_partition_configs =
      Context<TargetType::kNNAdapter>::NNAdapterSubgraphPartitionConfigBuffer(
          scope);
  if (subgraph_partition_configs.empty()) {
    auto path =
        Context<TargetType::kNNAdapter>::NNAdapterSubgraphPartitionConfigPath(
            scope);
    if (!path.empty()) {
      std::vector<char> buffer;
      if (ReadFile(path, &buffer, false)) {
        if (!buffer.empty()) {
          subgraph_partition_configs.insert(
              subgraph_partition_configs.begin(), buffer.begin(), buffer.end());
        }
      } else {
        LOG(WARNING)
            << "Missing the subgraph custom partition configuration file "
            << path;
      }
    }
  }
#endif
  // Read the config path from environment and load the partition configurations
  if (subgraph_partition_configs.empty()) {
    subgraph_partition_configs = GetConfigsFromEnv(
        SUBGRAPH_PARTITION_CONFIG_FILE, SUBGRAPH_PARTITION_CONFIG_BUFFER);
  }
  std::set<std::string> all_supported_ops;
  std::map<std::string, std::set<std::string>> device_supported_ops;
#define REGISTER_CONVERTER(__op_type__, __func_name__, __device_names__) \
  {                                                                      \
    std::string device_names = __device_names__;                         \
    device_names.erase(                                                  \
        std::remove(device_names.begin(), device_names.end(), ' '),      \
        device_names.end());                                             \
    auto supported_device_names = Split(device_names, ",");              \
    for (const auto& supported_device_name : supported_device_names) {   \
      device_supported_ops[supported_device_name].insert(#__op_type__);  \
    }                                                                    \
    all_supported_ops.insert(#__op_type__);                              \
  }
#include "lite/kernels/nnadapter/converter/all.h"
#undef __NNADAPTER_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

  auto teller = [&](Node* node) {
    if (!node->IsStmt()) return false;
    auto op_info = node->AsStmt().op_info();
    auto op_type = op_info->Type();
    // As long as a device does not register the supported operators in
    // lite/kernels/nnadapter/converter/all.h, we assume that all operators are
    // supported by default, and subgraph partition will be performed online (or
    // at the execution time).
    for (const auto& selected_device_name : selected_device_names) {
      if (!device_supported_ops.count(selected_device_name)) {
        return all_supported_ops.count(op_type) != 0;
      }
    }
    for (const auto& selected_device_name : selected_device_names) {
      if (device_supported_ops[selected_device_name].count(op_type) != 0 &&
          NNAdapterSubgraphOpTeller(
              selected_device_name, graph.get(), node, scope)) {
        return true;
      }
    }
    return false;
  };
  SubgraphFuser fuser(graph.get(),
                      teller,
                      1 /* min_subgraph_size */,
                      subgraph_partition_configs,
                      true);
  fuser();
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(nnadapter_subgraph_pass,
                  paddle::lite::mir::NNAdapterSubgraphPass)
    .BindTargets({TARGET(kNNAdapter)});
