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

#include "lite/core/mir/xpu_memory_optimize_pass.h"
#include <memory>
#include <utility>
#include <vector>
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace mir {

typedef struct {
  std::string name;
  int cluster;
  std::pair<int, int> lifetime;
  int life_interval;
  std::set<std::string> adj;
} XPUMemNode;

void XPUMemoryOptimizePass::CollectLifeCycleByDevice(
    std::map<std::string, lifecycle_map_t>* lifecycles, SSAGraph* graph) {
  max_lifecycle_ = 0;

  auto is_host = [](TargetType x) -> bool {
    return x == TARGET(kHost) || x == TARGET(kX86) || x == TARGET(kARM);
  };

  auto has_x86_opencl = [&]() -> bool {
    bool has_x86{false};
    bool has_opencl{false};
    for (auto& op_node : graph->StmtTopologicalOrder()) {
      if (!op_node->IsStmt()) continue;
      TargetType op_target_type = op_node->AsStmt().place().target;
      if (has_opencl && has_x86) {
        return true;
      }
      if (op_target_type == TARGET(kOpenCL)) {
        has_opencl = true;
      }
      if (op_target_type == TARGET(kX86)) {
        has_x86 = true;
      }
    }
    return has_x86 && has_opencl;
  };

  // The all of input and output variables of the Ops will not be reused.
  std::set<std::string> invalid_op_nodes = {
      "while",
      "conditional_block",
      "conditional_block_infer",
      "merge_lod_tensor_infer",
      "merge_lod_tensor",
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

  VLOG(4) << "invalid_op_nodes.size();" << invalid_op_nodes.size();
  insert_invalid_op_nodes_for_specific_target(invalid_op_nodes);
  VLOG(4) << "invalid_op_nodes.size();" << invalid_op_nodes.size();

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
    // The specified input and output variables of the Ops whose 'inplace' attr
    // is true will not be reused, such as reshape/reshape2's X and Out
    // variables
    std::map<std::string,
             std::pair<std::set<std::string>, std::set<std::string>>>
        inplace_op_nodes = {{"reshape", {{"X"}, {"Out"}}},
                            {"reshape2", {{"X"}, {"Out"}}},
                            {"flatten", {{"X"}, {"Out"}}},
                            {"flatten2", {{"X"}, {"Out"}}},
                            {"squeeze", {{"X"}, {"Out"}}},
                            {"squeeze2", {{"X"}, {"Out"}}},
                            {"unsqueeze", {{"X"}, {"Out"}}},
                            {"unsqueeze2", {{"X"}, {"Out"}}}};
    auto inplace_op_node = inplace_op_nodes.find(op_type);
    if (inplace_op_node != inplace_op_nodes.end()) {
      bool inplace = false;
      if (op_info->HasAttr("inplace")) {
        inplace = op_info->GetAttr<bool>("inplace");
      }
      if (inplace) {
        for (auto& in_param_name : inplace_op_node->second.first) {
          const auto& in_arg_names = op_info->Input(in_param_name);
          invalid_var_names.insert(in_arg_names.begin(), in_arg_names.end());
        }
        for (auto& out_param_name : inplace_op_node->second.second) {
          const auto& out_arg_names = op_info->Output(out_param_name);
          invalid_var_names.insert(out_arg_names.begin(), out_arg_names.end());
        }
      }
    }
  }

  // non-tensor(like tensor_array) variables will not be reused
  for (auto& node : graph->nodes()) {
    if (node.IsArg() && (node.arg()->type != nullptr) &&
        !node.arg()->type->IsTensor()) {
      invalid_var_names.insert(node.arg()->name);
    }
  }

  lite::TargetWrapperXPU::inside_kernel_l3_size = 0;
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (op_node->IsStmt()) {
      if (op_node->AsStmt().op_info()->Type() == "io_copy_once") {
        continue;
      }
      VLOG(4) << op_node->AsStmt().op_info()->Type() << " life is "
              << max_lifecycle_;
      if (op_node->AsStmt().op_info()->HasAttr("xpu_l3_size") &&
          op_node->AsStmt().op_info()->GetAttr<int>("xpu_l3_size") >
              lite::TargetWrapperXPU::inside_kernel_l3_size) {
        lite::TargetWrapperXPU::inside_kernel_l3_size = static_cast<size_t>(
            op_node->AsStmt().op_info()->GetAttr<int>("xpu_l3_size"));
      }
      std::vector<Node*> var_nodes(op_node->inlinks.begin(),
                                   op_node->inlinks.end());
      var_nodes.insert(
          var_nodes.end(), op_node->outlinks.begin(), op_node->outlinks.end());
      for (auto* var_node : var_nodes) {
        CHECK(var_node->IsArg());
        auto& arg = var_node->AsArg();
        if (arg.is_weight || arg.is_persist) continue;
        std::string var_name = arg.name;
        VLOG(4) << "OP VAR NAME IS " << var_name;
        if (var_name.find("_xpu_max") != std::string::npos) continue;
        if (invalid_var_names.count(var_name)) continue;
        TargetType target_type = arg.type->target();
        if (is_host(target_type)) target_type = TARGET(kHost);

        if (!(*lifecycles)[TargetToStr(target_type)].count(var_name)) {
          (*lifecycles)[TargetToStr(target_type)].emplace(
              var_name, std::make_pair(max_lifecycle_, max_lifecycle_));
        } else {
          int cur_life =
              (*lifecycles)[TargetToStr(target_type)][var_name].second;
          (*lifecycles)[TargetToStr(target_type)][var_name].second =
              (std::max)(max_lifecycle_, cur_life);
        }
      }
      ++max_lifecycle_;
    }
  }
  LOG(INFO) << "There are " << (*lifecycles).size() << " types device var.";
}

void XPUMemoryOptimizePass::MakeReusePlan(
    const lifecycle_map_t& lifecycles,
    std::map<std::string, std::string>* node2cluster) {
  std::vector<XPUMemNode> mem_nodes;
  std::vector<std::string> cluster;
  for (auto& data : lifecycles) {
    XPUMemNode temp_node;
    temp_node.name = data.first;
    temp_node.cluster = -1;
    temp_node.lifetime = data.second;
    temp_node.life_interval = data.second.second - data.second.first;
    mem_nodes.push_back(temp_node);
  }
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

  std::sort(mem_nodes.begin(), mem_nodes.end(), customLess);

  auto overlap = [](std::pair<int, int> a, std::pair<int, int> b) -> bool {
    return b.second >= a.first && a.second >= b.first;
  };
  // If the lifetime of two nodes is overwritten, we set them as adjacent nodes.
  for (size_t i = 0; i < mem_nodes.size(); i++) {
    for (size_t j = i + 1; j < mem_nodes.size(); j++) {
      if (overlap(mem_nodes[i].lifetime, mem_nodes[j].lifetime)) {
        mem_nodes[i].adj.insert(mem_nodes[j].name);
        mem_nodes[j].adj.insert(mem_nodes[i].name);
      }
    }
  }

  // Generating XPUMemory Reuse Strategy Based on Greedy Way
  // The vars can be reused if there is no overlap between them.
  for (size_t i = 0; i < mem_nodes.size(); i++) {
    if (mem_nodes[i].cluster >= 0 || mem_nodes[i].life_interval == 0) continue;
    int cluster_index = cluster.size();
    mem_nodes[i].cluster = cluster_index;
    (*node2cluster)[mem_nodes[i].name] = mem_nodes[i].name;
    VLOG(4) << "Mapping Tensor Cluster: " << mem_nodes[i].name
            << ", life time is " << mem_nodes[i].lifetime.first << " --> "
            << mem_nodes[i].lifetime.second;
    cluster.push_back(mem_nodes[i].name);
    std::set<std::string> cluster_adj = mem_nodes[i].adj;
    for (size_t j = i + 1; j < mem_nodes.size(); j++) {
      if (mem_nodes[j].cluster < 0 &&
          (cluster_adj.find(mem_nodes[j].name) == cluster_adj.end())) {
        (*node2cluster)[mem_nodes[j].name] = mem_nodes[i].name;
        mem_nodes[j].cluster = cluster_index;
        VLOG(4) << mem_nodes[j].name << ", life time is "
                << mem_nodes[j].lifetime.first << " --> "
                << mem_nodes[j].lifetime.second;
        for (auto& n : mem_nodes[j].adj) {
          cluster_adj.insert(n);
        }
      }
    }
  }
  std::vector<float> cluster_proportion;
  switch (cluster.size()) {
    case 1: {
      cluster_proportion = {1.0f};
      break;
    }
    case 2: {
      cluster_proportion = {0.5f, 0.5f};
      break;
    }
    case 3: {
      cluster_proportion = {0.45f, 0.3f, 0.25f};
      break;
    }
    case 4: {
      cluster_proportion = {0.4f, 0.25f, 0.25f, 0.1f};
      break;
    }
    default: {
      cluster_proportion = {0.35f, 0.2f, 0.2f, 0.15f, 0.1f};
      break;
    }
  }
  for (auto i = 0; i < cluster_proportion.size(); i++) {
    XPUL3CacheBlock cache_block;
    cache_block.init(cluster_proportion[i]);
    lite::TargetWrapperXPU::l3_block_dict[cluster[i]] = cache_block;
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
    auto* scope = stmt.op()->scope();
    std::map<std::string, std::vector<std::string>> in_args, out_args;
    // replace the op's input according the reuse table.
    for (auto argument : op_info->inputs()) {
      for (const auto& x : argument.second) {
        auto name = x;
        if (reuse_table.count(x) && reuse_table.at(x) != x) {
          name = reuse_table.at(x);
        }
        if (reuse_table.count(x) && reuse_table.at(x) == x) {
          auto* reuse_tensor = scope->FindMutableTensor(x);
          reuse_tensor->SetXPUL3CacheBlock(x);
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
        if (reuse_table.count(x) && reuse_table.at(x) == x) {
          auto* reuse_tensor = scope->FindMutableTensor(x);
          reuse_tensor->SetXPUL3CacheBlock(x);
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
  // const char* xpu_mem_optimize = std::getenv("XPU_MEMORY_OPTIMIZE");
  // if (xpu_mem_optimize == nullptr || std::strlen(xpu_mem_optimize) != 1 ||
  //    std::isdigit(*xpu_mem_optimize) == 0 ||
  //    std::atoi(xpu_mem_optimize) <= 0) {
  //  return;
  //}
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
  std::map<std::string, lifecycle_map_t> lifecycles;
  CollectLifeCycleByDevice(&lifecycles, graph.get());
  for (auto& ele : lifecycles) {
    if (ele.first != "xpu") {
      continue;
    }
    std::map<std::string, std::string> node2cluster;
    MakeReusePlan(ele.second, &node2cluster);
    PerformReusePlan(graph.get(), node2cluster);
  }
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
                     TARGET(kAPU),
                     TARGET(kMLU),
                     TARGET(kHuaweiAscendNPU),
                     TARGET(kImaginationNNA),
                     TARGET(kMetal),
                     TARGET(kNNAdapter)});
