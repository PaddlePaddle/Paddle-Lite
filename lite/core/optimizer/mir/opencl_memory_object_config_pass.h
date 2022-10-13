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

  void UpdateTensor(mir::Node::Stmt &inst,  // NOLINT
                    Node *in,
                    Node *out,
                    TargetType new_target = TargetType::kUnk) {
    auto get_argname = [&](const std::string &node_name,
                           const std::map<std::string, std::vector<std::string>>
                               &argname_map) -> std::string {
      for (auto &ele : argname_map) {
        auto it = std::find(ele.second.begin(), ele.second.end(), node_name);
        if (it != ele.second.end()) return ele.first;
      }
      return "";
    };

    std::string arg_name =
        get_argname(out->AsArg().name, inst.op_info()->outputs());
    std::string in_name =
        get_argname(in->AsArg().name, inst.op_info()->inputs());

    auto type = inst.picked_kernel().GetInputDeclType(in_name);
    auto tmp_ptype = in->AsArg().type->precision();
    auto tmp_target = type->target();
    auto tmp_layout = type->layout();

    if (new_target == TargetType::kARM) {
      tmp_target = TargetType::kARM;
      tmp_ptype = PrecisionType::kFloat;
      tmp_layout = DataLayoutType::kNCHW;
    }

    if (new_target == TargetType::kX86) {
      tmp_target = TargetType::kX86;
      tmp_ptype = PrecisionType::kFloat;
      tmp_layout = DataLayoutType::kNCHW;
    }

    if (new_target == TargetType::kHost) {
      tmp_target = TargetType::kHost;
      tmp_ptype = PrecisionType::kFloat;
      tmp_layout = DataLayoutType::kNCHW;
    }

    out->AsArg().type =
        LiteType::GetTensorTy(tmp_target, tmp_ptype, tmp_layout);
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

    if (ocl_memory_object_configs.size() < 1) {
      return;
    }

    std::string buffer_flag = "device:gpu buffer";
    std::string cpu_flag = "device:cpu";
    int buffer_flag_size = buffer_flag.size();
    int cpu_flag_size = cpu_flag.size();
    auto gpu_op_begin = ocl_memory_object_configs.find("device:gpu buffer");
    auto cpu_op_begin = ocl_memory_object_configs.find("device:cpu");
    auto config_file_end = ocl_memory_object_configs.rfind("\n");

    std::string opencl_buffer_op_config;
    std::string cpu_op_config;
    if (gpu_op_begin != std::string::npos &&
        cpu_op_begin == std::string::npos) {
      opencl_buffer_op_config = ocl_memory_object_configs.substr(
          gpu_op_begin + buffer_flag_size, config_file_end - buffer_flag_size);
    } else if (gpu_op_begin == std::string::npos &&
               cpu_op_begin != std::string::npos) {
      cpu_op_config = ocl_memory_object_configs.substr(
          cpu_op_begin + cpu_flag_size, config_file_end - cpu_flag_size);
    } else if (gpu_op_begin != std::string::npos &&
               cpu_op_begin != std::string::npos) {
      if (cpu_op_begin > gpu_op_begin) {
        opencl_buffer_op_config = ocl_memory_object_configs.substr(
            gpu_op_begin + buffer_flag_size,
            cpu_op_begin - buffer_flag_size - 1);
        cpu_op_config = ocl_memory_object_configs.substr(
            cpu_op_begin + cpu_flag_size,
            config_file_end - cpu_op_begin - cpu_flag_size);
      } else {
        cpu_op_config = ocl_memory_object_configs.substr(
            cpu_op_begin + cpu_flag_size, gpu_op_begin - cpu_flag_size - 1);
        opencl_buffer_op_config = ocl_memory_object_configs.substr(
            gpu_op_begin + buffer_flag_size,
            config_file_end - gpu_op_begin - buffer_flag_size);
      }
    }

    auto buffer_nodes =
        GetBufferNodesFromOpenCLBufferConfig(graph, opencl_buffer_op_config);
    auto cpu_nodes = GetBufferNodesFromOpenCLBufferConfig(graph, cpu_op_config);

    VLOG(4) << "opencl buffer nodes size:" << buffer_nodes.size();
    VLOG(4) << "opencl cpu nodes size:" << cpu_nodes.size();
    for (auto &x : graph->mutable_nodes()) {
      // slice reduce op unable to select the correct kernel temporarily
      if (cpu_nodes.count(&x) != 0) {
        auto &inst = x.AsStmt();
        auto new_place = inst.place();
        auto in = x.inlinks.front();
        if (!in) {
          continue;
        }
        auto out = x.outlinks.front();
        const auto &op_type = inst.op_type();
        auto tmp_ptype = in->AsArg().type->precision();
        TargetType new_target = TARGET(kARM);
        const auto &valid_places = graph->valid_places();
        for (const auto &place : valid_places) {
          if (place.target == TARGET(kARM)) {
            new_target = place.target;
            break;
          } else if (place.target == TARGET(kX86)) {
            new_target = place.target;
            break;
          }
        }
        new_place.target = new_target;
        new_place.precision = PRECISION(kFloat);
        if (op_type.find("elementwise") != std::string::npos) {
          new_place.precision = tmp_ptype;
        }
        new_place.layout = DATALAYOUT(kNCHW);

        std::vector<Place> places;
        places.push_back(new_place);
        inst.ResetKernels(places);

        UpdateTensor(inst, in, out, new_target);
      } else if (buffer_nodes.count(&x) != 0) {
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
