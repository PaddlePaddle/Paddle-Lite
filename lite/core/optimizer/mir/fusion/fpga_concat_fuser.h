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

#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

struct NodeInfo {
 public:
  NodeInfo(Node* node,
           bool wd_enable,
           int wd_offset,
           int fuse_idx = -1,
           int original_out_channel = 0)
      : node_(node),
        wd_enable_(wd_enable),
        wd_offset_(wd_offset),
        fuse_idx_(fuse_idx),
        original_out_channel_(original_out_channel) {}
  Node* node_;
  bool wd_enable_;
  int wd_offset_;
  int fuse_idx_;
  int original_out_channel_ = 0;
  int start_idx_ = 0;
  int end_idx_ = 0;
};

class FpgaConcatFuser : public FuseBase {
 public:
  FpgaConcatFuser() {}
  size_t operator()(SSAGraph* graph);
  // pure virtual function must has implementation although it is useless here
  void BuildPattern() override{};
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override{};

 private:
  std::vector<std::vector<NodeInfo>> PatternMatch(SSAGraph* graph);
  void ExtractInputsOutputs(const std::vector<NodeInfo>& patterns,
                            std::set<Node*>* input_var_nodes,
                            std::set<Node*>* weight_var_nodes,
                            std::set<Node*>* output_var_nodes);
  void InsertNewNode(SSAGraph* graph,
                     const std::vector<std::vector<NodeInfo>>& patterns);
  void DeleteInterNodes(SSAGraph* graph,
                        const std::vector<std::vector<NodeInfo>>& patterns);
  std::vector<std::vector<NodeInfo>> select_candidate(
      std::vector<NodeInfo> subgraph);
  void fuse_accumulate(std::vector<std::vector<NodeInfo>>* groups);
  bool enable_fuse(Node* varnode);
  int enable_jump(Node* opnode);
  std::string DebugPatternInfo(const std::vector<NodeInfo>& pattern);
  //  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
