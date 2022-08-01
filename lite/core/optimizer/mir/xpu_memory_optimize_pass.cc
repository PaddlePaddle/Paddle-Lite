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

#include "lite/core/optimizer/mir/xpu_memory_optimize_pass.h"
#include <algorithm>
#include <cctype>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/optimizer/mir/graph_visualize_pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace mir {

void XPUMemoryOptimizePass::CollectLifeCycleByDevice(SSAGraph* graph) {
  max_lifecycle_ = 0;
  std::map<std::string, lifecycle_map_t> lifecycles;

  auto is_host = [](TargetType x) -> bool {
    return x == TARGET(kHost) || x == TARGET(kX86) || x == TARGET(kARM);
  };

  // The all of input and output variables of the Ops will not be reused.
  std::set<std::string> invalid_op_nodes = {
      "while",
      "conditional_block",
      "conditional_block_infer",
      "merge_lod_tensor_infer",
      "merge_lod_tensor",
      "flatten_contiguous_range",
      "equal",
      "lod_reset",
      "yolo_box",
      "subgraph",
      "feed",
      "fetch",
      "cast",
      "expand",
      "io_copy_once",
      "scale",
      "__xpu__resnet50",
      "softmax",
  };

  auto insert_invalid_op_nodes_for_specific_target =
      [&](std::set<std::string> op_node_set) {
        for (auto& op_node : graph->StmtTopologicalOrder()) {
          if (!op_node->IsStmt()) continue;
          TargetType op_target_type = op_node->AsStmt().place().target;
          if (op_target_type != TARGET(kXPU)) {
            invalid_op_nodes.insert(op_node->AsStmt().op_info()->Type());
          }
        }
      };

  insert_invalid_op_nodes_for_specific_target(invalid_op_nodes);

  // Collect the invalid input and output variables that will not be reused.
  std::set<std::string> invalid_var_names;
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    // variables of invalid_op_nodes wil not be reused
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().op_info();
    auto op_type = op_info->Type();
    auto invalid_op_node = invalid_op_nodes.find(op_type);
    if (invalid_op_node != invalid_op_nodes.end()) {
      for (auto in_var_node : op_node->inlinks) {
        CHECK(in_var_node->IsArg());
        invalid_var_names.insert(in_var_node->AsArg().name);
      }
      for (auto out_var_node : op_node->outlinks) {
        CHECK(out_var_node->IsArg());
        invalid_var_names.insert(out_var_node->AsArg().name);
      }
      continue;
    }
  }

  // non-tensor(like tensor_array) variables will not be reused
  for (auto& node : graph->nodes()) {
    if (node.IsArg() && (node.arg()->type != nullptr) &&
        !node.arg()->type->IsTensor()) {
      invalid_var_names.insert(node.arg()->name);
    }
  }

  std::map<std::string, std::pair<std::set<std::string>, std::set<std::string>>>
      inplace_op_nodes = {{"reshape", {{"X"}, {"Out"}}},
                          {"reshape2", {{"X"}, {"Out"}}},
                          {"flatten", {{"X"}, {"Out"}}},
                          {"flatten2", {{"X"}, {"Out"}}},
                          {"squeeze", {{"X"}, {"Out"}}},
                          {"squeeze2", {{"X"}, {"Out"}}},
                          {"unsqueeze", {{"X"}, {"Out"}}},
                          {"unsqueeze2", {{"X"}, {"Out"}}}};
  std::vector<std::vector<std::string>> inpalce_reuse_var_names;
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    auto op_type = op_node->AsStmt().op_info()->Type();
    auto op_info = op_node->AsStmt().op_info();
    if (op_node->IsStmt()) {
      if (op_type == "io_copy_once") {
        continue;
      }
      VLOG(4) << op_type << " life is " << max_lifecycle_;
      std::vector<Node*> var_nodes(op_node->inlinks.begin(),
                                   op_node->inlinks.end());
      var_nodes.insert(
          var_nodes.end(), op_node->outlinks.begin(), op_node->outlinks.end());
      TargetType target_type;

      auto inplace_op_node = inplace_op_nodes.find(op_type);
      if (inplace_op_node != inplace_op_nodes.end()) {
        bool inplace = false;
        if (op_info->HasAttr("inplace")) {
          inplace = op_info->GetAttr<bool>("inplace");
        }
        if (inplace) {
          auto in_arg_name = op_info->Input("X")[0];
          auto out_arg_name = op_info->Output("Out")[0];
          if (invalid_var_names.count(in_arg_name)) continue;
          if (invalid_var_names.count(out_arg_name)) continue;
          bool reuse = false;
          int i = 0;
          for (const auto& reuse_var_names : inpalce_reuse_var_names) {
            auto in_iter = std::find(
                reuse_var_names.begin(), reuse_var_names.end(), in_arg_name);
            if (in_iter != reuse_var_names.end()) {
              reuse = true;
              inpalce_reuse_var_names[i].push_back(out_arg_name);
              break;
            }
            ++i;
          }
          if (!reuse) {
            std::vector<std::string> tmp_reuse_var_name{in_arg_name,
                                                        out_arg_name};
            inpalce_reuse_var_names.push_back(tmp_reuse_var_name);
          }
        }
      }

      for (auto* var_node : var_nodes) {
        CHECK(var_node->IsArg());
        auto& arg = var_node->AsArg();
        if (arg.is_weight || arg.is_persist) continue;
        std::string var_name = arg.name;
        VLOG(4) << "OP VAR NAME IS " << var_name;
        if (var_name.find("_xpu_max") != std::string::npos) continue;
        if (invalid_var_names.count(var_name)) continue;
        target_type = arg.type->target();
        if (is_host(target_type)) target_type = TARGET(kHost);

        if (!lifecycles[TargetToStr(target_type)].count(var_name)) {
          lifecycles[TargetToStr(target_type)].emplace(
              var_name, std::make_pair(max_lifecycle_, max_lifecycle_));
        } else {
          int cur_life = lifecycles[TargetToStr(target_type)][var_name].second;
          lifecycles[TargetToStr(target_type)][var_name].second =
              (std::max)(max_lifecycle_, cur_life);
        }
      }
      ++max_lifecycle_;
    }
  }

  for (const auto& reuse_var_names : inpalce_reuse_var_names) {
    if (!lifecycles["xpu"].count(reuse_var_names.front()) ||
        !lifecycles["xpu"].count(reuse_var_names.back())) {
      for (const auto& reuse_var_name : reuse_var_names) {
        VLOG(4) << "inplace node var name is not in lifecycles:"
                << reuse_var_name;
      }
      continue;
    }

    int min_life = lifecycles["xpu"][reuse_var_names.front()].first;
    int max_life = lifecycles["xpu"][reuse_var_names.back()].second;
    for (const auto& reuse_var_name : reuse_var_names) {
      VLOG(4) << "inplace node var name:" << reuse_var_name
              << "origin life time is :"
              << lifecycles["xpu"][reuse_var_name].first << " --> "
              << lifecycles["xpu"][reuse_var_name].second;
      lifecycles["xpu"][reuse_var_name].first = min_life;
      lifecycles["xpu"][reuse_var_name].second = max_life;
      VLOG(4) << "inplace node var name:" << reuse_var_name
              << "new life time is :" << lifecycles["xpu"][reuse_var_name].first
              << " --> " << lifecycles["xpu"][reuse_var_name].second;
    }
  }

  LOG(INFO) << "There are " << lifecycles.size() << " types device var.";
  for (auto& ele : lifecycles) {
    if (ele.first != "xpu") {
      continue;
    }

    for (auto& data : ele.second) {
      XPUMemNode temp_node;
      temp_node.name = data.first;
      temp_node.cluster = -1;
      temp_node.lifetime = data.second;
      temp_node.life_interval = data.second.second - data.second.first;

      for (const auto& reuse_var_names : inpalce_reuse_var_names) {
        auto in_iter = std::find(
            reuse_var_names.begin() + 1, reuse_var_names.end(), data.first);
        if (in_iter != reuse_var_names.end()) {
          temp_node.cluster = 1;
        }
      }
      mem_nodes_.push_back(temp_node);
    }
  }
}

void XPUMemoryOptimizePass::MakeReusePlan(
    std::map<std::string, std::string>* node2cluster) {
  std::vector<std::string> cluster;

  // Sort Node with life_interval to optimize L3 usage by Greedy Way later
  struct {
    bool operator()(XPUMemNode a, XPUMemNode b) const {
      if (a.life_interval < b.life_interval) {
        return true;
      } else if (a.life_interval == b.life_interval) {
        return a.lifetime.first < b.lifetime.first;
      } else {
        return false;
      }
    }
  } customLess;

  std::sort(mem_nodes_.begin(), mem_nodes_.end(), customLess);

  auto overlap = [](std::pair<int, int> a, std::pair<int, int> b) -> bool {
    return b.second >= a.first && a.second >= b.first;
  };
  // If the lifetime of two nodes is overwritten, we set them as adjacent nodes.
  for (size_t i = 0; i < mem_nodes_.size(); i++) {
    for (size_t j = i + 1; j < mem_nodes_.size(); j++) {
      if (overlap(mem_nodes_[i].lifetime, mem_nodes_[j].lifetime)) {
        mem_nodes_[i].adj.insert(mem_nodes_[j].name);
        mem_nodes_[j].adj.insert(mem_nodes_[i].name);
      }
    }
  }

  // Generating XPUMemory Reuse Strategy Based on Greedy Way
  // The vars can be reused if there is no overlap between them.
  for (size_t i = 0; i < mem_nodes_.size(); i++) {
    if (mem_nodes_[i].cluster >= 0 || mem_nodes_[i].life_interval == 0)
      continue;
    int cluster_index = cluster.size();
    mem_nodes_[i].cluster = cluster_index;
    (*node2cluster)[mem_nodes_[i].name] = mem_nodes_[i].name;
    VLOG(4) << "Mapping Tensor Cluster: " << mem_nodes_[i].name
            << ", life time is " << mem_nodes_[i].lifetime.first << " --> "
            << mem_nodes_[i].lifetime.second;
    cluster.push_back(mem_nodes_[i].name);
    std::set<std::string> cluster_adj = mem_nodes_[i].adj;
    for (size_t j = i + 1; j < mem_nodes_.size(); j++) {
      if (mem_nodes_[j].cluster < 0 &&
          (cluster_adj.find(mem_nodes_[j].name) == cluster_adj.end())) {
        (*node2cluster)[mem_nodes_[j].name] = mem_nodes_[i].name;
        mem_nodes_[j].cluster = cluster_index;
        VLOG(4) << mem_nodes_[j].name << ", life time is "
                << mem_nodes_[j].lifetime.first << " --> "
                << mem_nodes_[j].lifetime.second;
        for (auto& n : mem_nodes_[j].adj) {
          cluster_adj.insert(n);
        }
      }
    }
  }
  for (auto& name : cluster) {
    LOG(INFO) << "cluster: " << name;
  }
}

void XPUMemoryOptimizePass::PerformReusePlan(
    SSAGraph* graph, const std::map<std::string, std::string>& reuse_table) {
  int node_append_idx = 0;
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto& stmt = op_node->AsStmt();
    auto* op_info = stmt.mutable_op_info();
    std::map<std::string, std::vector<std::string>> in_args, out_args;
    // replace the op's input according the reuse table.
    for (auto argument : op_info->inputs()) {
      for (const auto& x : argument.second) {
        auto name = x;
        if (reuse_table.count(x) && reuse_table.at(x) != x) {
          name = reuse_table.at(x);
        }
        in_args[argument.first].push_back(name);
        VLOG(4) << op_info->Type() << " input " << x << " -> " << name;
      }
    }

    // modify the graph
    for (Node* input_node : op_node->inlinks) {
      CHECK(input_node->IsArg()) << "The op node's inputs should be var node.";
      std::string name = input_node->AsArg().name;
      if (reuse_table.count(name) && reuse_table.at(name) != name) {
        auto replace_name = reuse_table.at(name);
        input_node->AsArg().name =
            replace_name + "(" + paddle::lite::to_string(node_append_idx) + ")";
        node_append_idx++;
      }
    }

    // replace the op's output according the reuse table.
    for (auto argument : op_info->outputs()) {
      for (const auto& x : argument.second) {
        auto name = x;
        if (reuse_table.count(x) && reuse_table.at(x) != x) {
          name = reuse_table.at(x);
        }
        out_args[argument.first].push_back(name);
        VLOG(4) << op_info->Type() << " output " << x << " -> " << name;
      }
    }

    // modify the graph
    for (Node* out_node : op_node->outlinks) {
      CHECK(out_node->IsArg()) << "The op node's outputs should be var node.";
      std::string name = out_node->AsArg().name;
      if (reuse_table.count(name) && reuse_table.at(name) != name) {
        auto replace_name = reuse_table.at(name);
        out_node->AsArg().name =
            replace_name + "(" + paddle::lite::to_string(node_append_idx) + ")";
        node_append_idx++;
      }
    }

    for (auto& arg : in_args) {
      op_info->SetInput(arg.first, arg.second);
    }
    for (auto& arg : out_args) {
      op_info->SetOutput(arg.first, arg.second);
    }

    auto original_selected_kernel = std::move(stmt.kernels().front());
    auto updated_op_info = *stmt.mutable_op_info();
    stmt.ResetOp(updated_op_info, graph->valid_places());
    stmt.kernels().clear();
    stmt.kernels().emplace_back(std::move(original_selected_kernel));
    for (auto& kernel : stmt.kernels()) {
      VLOG(4) << "kernel info: " << kernel->name();
      stmt.op()->AttachKernel(kernel.get());
    }
    graph->CheckValid();
  }
}

void XPUMemoryOptimizePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Memory optimization.
  // We will perform the following operation:
  // 1. Collect all var's lifetime, then classify them according to the device.
  // Only the vars on the same device can be reused.
  // 2. Make reuse plan: the vars can be reused if there is no overlap between
  // them.
  // The final plan is a mapping table in which the key represents the original
  // name of var and the value in the table represents the current name of var.
  // 3. Perform reuse plan: Replace all var's name in the model according to the
  // mapping table.
  // std::map<std::string, lifecycle_map_t> lifecycles;
  CollectLifeCycleByDevice(graph.get());
  std::map<std::string, std::string> node2cluster;
  MakeReusePlan(&node2cluster);
  PerformReusePlan(graph.get(), node2cluster);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(xpu_memory_optimize_pass,
                  paddle::lite::mir::XPUMemoryOptimizePass)
    .BindTargets({TARGET(kXPU)})
    .ExcludeTargets({TARGET(kARM),
                     TARGET(kNPU),
                     TARGET(kOpenCL),
                     TARGET(kBM),
                     TARGET(kRKNPU),
                     TARGET(kMLU),
                     TARGET(kMetal),
                     TARGET(kNNAdapter)});
