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

#include "lite/core/device_info.h"
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace mir {

void MultiStreamAnalysisPass::CleanUp() {
  exec_ops_.clear();
  wait_que_.clear();
  wait_que_cpu_.clear();
  std::queue<int> empty_queue;
  while (!exec_que_.empty()) {
    exec_que_.pop();
  }
  ops_in_streams_.clear();
  resources_.clear();
  map_arg_to_lane_.clear();
  op_types_set_.clear();
  io_copy_once_num_ = 0;
}

void MultiStreamAnalysisPass::Init(SSAGraph* graph) {
  // If not cleaned, the clone will overlay the previous state
  CleanUp();
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

      // feed and io_copy_once op has no dependencies and can be launched
      // directly. Other ops are put into the waiting queue.
      if (op_node->AsStmt().op_type() == "feed" ||
          op_node->AsStmt().op_type() == "io_copy_once") {
        exec_que_.push(op_node);
      } else {
        auto tgt = op_node->AsStmt().kernels().front()->target();
        if (tgt == TargetType::kCUDA) {
          wait_que_.push_back(op_node);
        } else {
          wait_que_cpu_.push_back(op_node);
        }
      }
      op_types_set_.insert(op_node->AsStmt().op_type());
    }
  }

  // Set the stream id according to the number of feed ops, and set the output
  // of the feed op to be accessible.
  int lane = 0;
  auto nodes = graph->inputs();
  ops_in_streams_.resize(max_stream_);

  for (auto& node : nodes) {
    std::string::size_type idx = node->AsArg().name.find("feed");
    if (idx != std::string::npos) {
      for (auto& feed_ops : node->outlinks) {
        if (feed_ops->AsStmt().op_type() == "feed") {
          // feed op doesn't need to wait sync.
          feed_ops->AsStmt().need_sync_ = false;
          CHECK_EQ(static_cast<int>(feed_ops->outlinks.size()), 1)
              << "feed op must have one output.";
          for (auto& var : feed_ops->outlinks) {
            var->AsArg().lane = lane;
            map_arg_to_lane_[var->AsArg().name] = lane;
            resources_[var->AsArg().name] = true;
          }
          feed_ops->AsStmt().stream_id_ = lane;
          ops_in_streams_[lane].push_back(feed_ops);
          ++lane;
          if (lane >= max_stream_) {
            lane = 0;
          }
        }
      }
    }
    // set all io_copy_once op in the first stream
    for (auto& io_copy_once_ops : node->outlinks) {
      if (io_copy_once_ops->AsStmt().op_type() == "io_copy_once") {
        ops_in_streams_[0].push_back(io_copy_once_ops);
        io_copy_once_ops->AsStmt().stream_id_ = 0;
        io_copy_once_ops->AsStmt().need_sync_ = false;
        ++io_copy_once_num_;
      }
    }
  }
}

bool MultiStreamAnalysisPass::CheckOpSupport() {
  std::set<std::string> invalid_op = {
      "while", "conditional_block", "conditional_block_infer", "graph_op"};
  for (auto& op_type : op_types_set_) {
    if (invalid_op.count(op_type)) {
      LOG(INFO) << "multi_stream_analysis_pass don't support " << op_type
                << ", just return.";
      return false;
    }
  }
  return true;
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

int MultiStreamAnalysisPass::SelectStreamId(const std::vector<int>& lanes) {
  if (lanes.size() == 0) {
    return 0;
  }

  int res = lanes[0];
  int exclude_io_copy_once_num = ops_in_streams_[0].size() - io_copy_once_num_;
  int min_num = lanes[0] == 0 ? exclude_io_copy_once_num
                              : ops_in_streams_[lanes[0]].size();
  for (size_t i = 1; i < lanes.size(); ++i) {
    int ith_num = lanes[i] == 0 ? exclude_io_copy_once_num
                                : ops_in_streams_[lanes[i]].size();
    if (ith_num < min_num) {
      res = lanes[i];
      min_num = ith_num;
    }
  }

  return res;
}

void MultiStreamAnalysisPass::Launch(Node* stmt_node) {
  // record ops launch order.
  exec_que_.push(stmt_node);
  std::vector<int> lanes;
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

  int stream_id = SelectStreamId(lanes);

  // If all inputs of the op are on multiple streams, they need to be
  // synchronized
  if (lanes.size() > 1) {
    for (size_t i = 0; i < lanes.size(); ++i) {
      if (lanes[i] != stream_id) {
        stmt_node->AsStmt().sync_streams_.push_back(lanes[i]);
      }
    }
    stmt_node->AsStmt().need_sync_ = true;
  }
  // io_copy are nodes inserted across devices and need to be synced.
  if (stmt_node->AsStmt().op_type() == "io_copy") {
    stmt_node->AsStmt().need_sync_ = true;
  }
  stmt_node->AsStmt().stream_id_ = stream_id;

  // set output lane and set the output of op to be accessible.
  for (auto& out_arg : stmt_node->outlinks) {
    out_arg->AsArg().lane = stream_id;
    resources_[out_arg->AsArg().name] = true;
  }
  ops_in_streams_[stream_id].push_back(stmt_node);
}

void MultiStreamAnalysisPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
#ifdef LITE_WITH_CUDA
  typename Env<TargetType::kCUDA>::Devs& devs =
      Env<TargetType::kCUDA>::Global();
  int dev_id = TargetWrapper<TargetType::kCUDA>::GetCurDevice();
  max_stream_ = devs[dev_id].max_stream();
#else
  LOG(FATAL) << "Please re-compile by setting the cmake flag LITE_WITH_CUDA=ON";
#endif

  // Find the correct startup sequence for op.
  Init(graph.get());
  bool is_valid = CheckOpSupport();
  if (!is_valid) {
    return;
  }
  size_t prev_size;

  while (!(this->wait_que_.empty() && this->wait_que_cpu_.empty())) {
    prev_size = this->wait_que_.size() + this->wait_que_cpu_.size();
    // launch the acessible cuda kernel and remove it from wait que.
    for (auto it = this->wait_que_.begin(); it != this->wait_que_.end();) {
      if (IsPrepared(*it)) {
        Launch(*it);
        it = wait_que_.erase(it);
      } else {
        ++it;
      }
    }
    // launch the accessible cpu kernel and remove it from wait que.
    for (auto cpu_it = this->wait_que_cpu_.begin();
         cpu_it != this->wait_que_cpu_.end();) {
      if (IsPrepared(*cpu_it)) {
        Launch(*cpu_it);
        cpu_it = wait_que_cpu_.erase(cpu_it);
      } else {
        ++cpu_it;
      }
    }

    if (this->wait_que_.size() + this->wait_que_cpu_.size() == prev_size) {
      LOG(FATAL) << "network topo error!";
    }
  }

  // Get exec ops order.
  while (!exec_que_.empty()) {
    auto* node = exec_que_.front();
    exec_ops_.push_back(node);
    VLOG(4) << node->AsStmt().op_type()
            << " stream: " << node->AsStmt().stream_id_
            << ", sync: " << node->AsStmt().need_sync_;
    if (node->AsStmt().need_sync_) {
      for (size_t i = 0; i < node->AsStmt().sync_streams_.size(); ++i) {
        VLOG(4) << "        " << node->AsStmt().sync_streams_[i];
      }
    }
    exec_que_.pop();
  }

  // Set attribute parameters, for passing parameters between passes
  const std::string attr_name{"nodes_in_order"};
  SetAttr<std::vector<Node*>>(attr_name, &exec_ops_);

  LOG(INFO) << "stream " << 0 << " has "
            << ops_in_streams_[0].size() - io_copy_once_num_
            << " ops. (exclude io_copy_once).";
  for (size_t i = 1; i < ops_in_streams_.size(); ++i) {
    LOG(INFO) << "stream " << i << " has " << ops_in_streams_[i].size()
              << " ops.";
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(multi_stream_analysis_pass,
                  paddle::lite::mir::MultiStreamAnalysisPass)
    .BindTargets({TARGET(kCUDA)});
