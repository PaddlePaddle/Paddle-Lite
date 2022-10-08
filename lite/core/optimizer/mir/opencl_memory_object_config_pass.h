// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/target_wrapper.h"
#include "lite/utils/env.h"
namespace paddle {
namespace lite {
namespace mir {

class OpenCLMemoryObjectConfigPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph> &graph) override;

 private:
  std::string ReadMemoryObjectConfigsFromEnv() {
    std::string configs;
    auto path = GetStringFromEnv(OPENCL_MEMORY_CONFIG_FILE);
    if (!path.empty()) {
      std::vector<char> buffer;
      if (ReadFile(path, &buffer, false)) {
        if (!buffer.empty()) {
          configs.insert(configs.begin(), buffer.begin(), buffer.end());
        }
      } else {
        LOG(WARNING) << "Missing the opencl memory configuration file " << path;
      }
    }
    return configs;
  }

  std::set<Node *> GetBufferNodesFromOpenCLBufferConfig(
      SSAGraph *graph, const std::string &ocl_buffer_configs) {
    // Get the buffer nodes from the opencl memory configurations
    std::set<Node *> buffer_nodes;
    std::vector<std::string> lines = Split(ocl_buffer_configs, "\n");
    for (const auto &line : lines) {
      if (line.empty()) continue;
      std::vector<std::string> node_info = Split(line, ":");
      std::string op_type = node_info.at(0);
      std::vector<std::string> in_vars_name;
      if (node_info.size() > 1) {
        in_vars_name = Split(node_info.at(1), ",");
      }
      std::vector<std::string> out_vars_name;
      if (node_info.size() > 2) {
        out_vars_name = Split(node_info.at(2), ",");
      }

      for (auto &node : graph->mutable_nodes()) {
        if (node.IsArg()) continue;
        auto stmt = node.stmt();
        if (op_type != stmt->op_type()) continue;
        auto in_nodes = node.inlinks;
        auto out_nodes = node.outlinks;
        if (in_vars_name.size() > in_nodes.size() ||
            out_vars_name.size() > out_nodes.size()) {
          continue;
        }

        bool matched = true;

        for (auto in_var_name : in_vars_name) {
          bool find_var = false;
          for (auto *in_node : in_nodes) {
            if (in_node->arg()->name == in_var_name) {
              find_var = true;
              break;
            }
          }
          if (!find_var) {
            matched = false;
            break;
          }
        }

        for (auto out_var_name : out_vars_name) {
          bool find_var = false;
          for (auto *out_node : out_nodes) {
            if (out_node->arg()->name == out_var_name) {
              find_var = true;
              break;
            }
          }
          if (!find_var) {
            matched = false;
            break;
          }
        }

        if (matched) {
          buffer_nodes.insert(&node);
        }
      }
    }
    return buffer_nodes;
  }

  void MemoryObjectConfig(SSAGraph *graph) {
    auto ocl_memory_object_configs = ReadMemoryObjectConfigsFromEnv();
    auto buffer_nodes =
        GetBufferNodesFromOpenCLBufferConfig(graph, ocl_memory_object_configs);
    VLOG(4) << "opencl buffer nodes size:" << buffer_nodes.size();
    for (auto &x : graph->mutable_nodes()) {
      if (buffer_nodes.count(&x) != 0) {
        auto &inst = x.AsStmt();
        auto new_place = inst.place();

        new_place.target = TARGET(kOpenCL);
        new_place.precision = PRECISION(kFP16);
        new_place.layout = DATALAYOUT(kNCHW);

        std::vector<Place> places;
        places.push_back(new_place);
        inst.ResetKernels(places);
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
