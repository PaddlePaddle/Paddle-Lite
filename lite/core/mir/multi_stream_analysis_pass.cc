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

#include "lite/core/mir/multi_stream_analysis_pass.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace mir {

void MultiStreamAnalysisPass::Init(SSAGraph* graph) {
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (op_node->IsStmt()) {
      // Set all outputs of op to inaccessible state.
      auto outputs = op_node->outlinks;
      for (Node* node : outputs) {
        CHECK(node->IsArg());
        auto& arg = node->AsArg();
        if (!resources_.count(arg.name)) {
          resources_[arg.name] = false;
        }
      }
      // Set the weight input of op to be accessible.
      auto inputs = op_node->inlinks;
      for (Node* node : inputs) {
        CHECK(node->IsArg());
        auto& arg = node->AsArg();
        if (arg.is_weight || arg.is_persist) {
          resources_[arg.name] = true;
        }
      }

      // feed op has no dependencies and can be launched directly.
      // Other ops are put into the waiting queue.
      if (op_node->AsStmt().op_type() == "feed") {
        exec_que_.push(op_node);
      } else {
        wait_que_.push_back(op_node);
      }
    }
  }

  // Set the stream id according to the number of feed ops, and set the output
  // of the feed op to be accessible.
  uint32_t lane = 0;
  auto nodes = graph->inputs();

  for (auto& node : nodes) {
    std::string::size_type idx = node->AsArg().name.find("feed");
    if (idx != std::string::npos) {
      for (auto& feed_ops : node->outlinks) {
        // feed op doesn't need to wait sync.
        feed_ops->AsStmt().need_sync_ = false;
        CHECK_EQ(feed_ops->outlinks.size(), 1)
            << "feed op must have one output.";
        for (auto& var : feed_ops->outlinks) {
          var->AsArg().lane = lane;
          map_arg_to_lane_[var->AsArg().name] = lane;
          resources_[var->AsArg().name] = true;
        }
        ops_in_streams_.push_back({feed_ops});
        ++lane;
      }
    }
  }
}

bool MultiStreamAnalysisPass::IsPrepared(Node* stmt_node) {
  // feed op are prepared when init.
  std::string op_name = stmt_node->AsStmt().op_type();
  if (op_name == "feed") {
    return true;
  }

  // Check is op's input are all accessible.
  std::vector<std::string> args;
  for (auto* ins : stmt_node->inlinks) {
    args.push_back(ins->AsArg().name);
  }
  return CheckAccess(args);
}

bool MultiStreamAnalysisPass::CheckAccess(
    const std::vector<std::string>& args) {
  if (args.size() == 0) {
    return true;
  }
  for (auto& name : args) {
    if (resources_[name]) {
      continue;
    } else {
      return false;
    }
  }
  return true;
}

uint32_t MultiStreamAnalysisPass::SelectStreamId(
    const std::vector<uint32_t>& lanes) {
  if (lanes.size() == 0) {
    return 0;
  }

  uint32_t res = lanes[0];
  size_t min_num = ops_in_streams_[lanes[0]].size();
  for (size_t i = 1; i < lanes.size(); ++i) {
    if (ops_in_streams_[lanes[i]].size() < min_num) {
      res = lanes[i];
      min_num = ops_in_streams_[lanes[i]].size();
    }
  }

  return res;
}

void MultiStreamAnalysisPass::Launch(Node* stmt_node) {
  // record ops launch order.
  exec_que_.push(stmt_node);
  std::vector<uint32_t> lanes;
  for (auto& in_arg : stmt_node->inlinks) {
    // Weight parameter does not involve stream id, so just skip it.
    if (in_arg->AsArg().is_weight || in_arg->AsArg().is_persist) {
      continue;
    }

    if (std::find(lanes.begin(), lanes.end(), in_arg->AsArg().lane) ==
        lanes.end()) {
      lanes.push_back(in_arg->AsArg().lane);
    }
  }

  // If all inputs of the op are on multiple streams, they need to be
  // synchronized
  if (lanes.size() > 1) {
    stmt_node->AsStmt().need_sync_ = true;
  }

  // set output lane and set the output of op to be accessible.
  uint32_t stream_id = SelectStreamId(lanes);
  for (auto& out_arg : stmt_node->outlinks) {
    out_arg->AsArg().lane = stream_id;
    resources_[out_arg->AsArg().name] = true;
  }
  ops_in_streams_[stream_id].push_back(stmt_node);
}

void MultiStreamAnalysisPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Find the correct startup sequence for op.
  Init(graph.get());
  size_t prev_size;

  while (!(this->wait_que_.empty())) {
    prev_size = this->wait_que_.size();
    // launch the acessible op and remove it from wait que.
    for (auto it = this->wait_que_.begin(); it != this->wait_que_.end();) {
      if (IsPrepared(*it)) {
        Launch(*it);
        it = wait_que_.erase(it);
      } else {
        ++it;
      }
    }

    if (this->wait_que_.size() == prev_size) {
      LOG(FATAL) << "network topo error!";
    }
  }

  // Get exec ops order.
  while (!exec_que_.empty()) {
    auto* node = exec_que_.front();
    exec_ops_.push_back(node);
    VLOG(3) << node->AsStmt().op_type()
            << " sync: " << node->AsStmt().need_sync_;
    exec_que_.pop();
  }

  for (size_t i = 0; i < ops_in_streams_.size(); ++i) {
    VLOG(3) << "stream " << i << " has " << ops_in_streams_[i].size()
            << " ops.";
  }

  // VLOG(4) << Visualize(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(multi_stream_analysis_pass,
                  paddle::lite::mir::MultiStreamAnalysisPass)
    .BindTargets({TARGET(kCUDA)});
