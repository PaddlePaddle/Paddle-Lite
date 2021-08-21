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

#include "lite/core/optimizer/mir/fusion/fpga_concat_fuser.h"
#include <map>
#include <memory>
#include "lite/core/optimizer/mir/pattern_matcher.h"
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

// more jump write supported op will be added here
std::map<std::string, std::string> FPGA_SUBOP_MAP{{"conv2d", "fpga_conv2d"}};

std::string FpgaConcatFuser::DebugPatternInfo(
    const std::vector<NodeInfo>& pattern) {
  std::stringstream ss;
  for (auto& nodeinfo : pattern) {
    auto node = nodeinfo.node_;
    if (node->IsStmt()) {
      auto type = node->AsStmt().op_info()->Type();
      bool wd_enable = nodeinfo.wd_enable_;
      int fuse_idx = nodeinfo.fuse_idx_;
      int start_offset = nodeinfo.original_out_channel_;
      if (type == "concat") {
        auto output_name = node->outlinks.front()->AsArg().name;
        ss << type << "/" << output_name << "/";
      } else {
        ss << type << "/" << wd_enable << "/" << fuse_idx << "/" << start_offset
           << "/";
      }
    }
  }
  return ss.str();
}

int FpgaConcatFuser::enable_jump(Node* opnode) {
  auto op = opnode->stmt()->op();
  auto* scope = op->scope();

  auto opdesc = opnode->AsStmt().op_info();
  int groups = 1;
  if (opdesc->HasAttr("groups")) groups = opdesc->GetAttr<int>("groups");
  const std::string& type = opdesc->Type();
  if (type == "conv2d") {
    auto filter = opdesc->Input("Filter").front();
    auto filter_tensor = scope->FindVar(filter)->GetMutable<lite::Tensor>();
    auto filter_dims = filter_tensor->dims();
    auto cout = filter_dims[0];
    auto cin = filter_dims[3];
    if (cin <= 2047 && cout % 16 == 0 && groups == 1) return cout;
  }
  return 0;
}

bool FpgaConcatFuser::enable_fuse(Node* varnode) {
  if (varnode->outlinks.size() == 1) return true;
  return false;
}

void FpgaConcatFuser::fuse_accumulate(
    std::vector<std::vector<NodeInfo>>* groups) {
  for (auto& group : *groups) {
    int total_output_channel = 0;
    int start_offset = 0;
    std::vector<int> ori_channel;
    int start_idx = group[0].fuse_idx_;
    int end_idx = group[group.size() - 1].fuse_idx_;

    ori_channel.push_back(0);
    for (NodeInfo& nodeinfo : group) {
      total_output_channel += nodeinfo.wd_offset_;
      ori_channel.push_back(nodeinfo.wd_offset_);
    }
    int i = 0;
    for (NodeInfo& nodeinfo : group) {
      nodeinfo.wd_offset_ = total_output_channel;
      start_offset += ori_channel[i];
      nodeinfo.original_out_channel_ = start_offset;
      nodeinfo.start_idx_ = start_idx;
      nodeinfo.end_idx_ = end_idx;
      ++i;
    }
  }
}

std::vector<std::vector<NodeInfo>> FpgaConcatFuser::select_candidate(
    std::vector<NodeInfo> subgraph) {
  std::vector<std::vector<NodeInfo>> groups;
  std::vector<NodeInfo> cur_group;

  int i = 0;
  int length = subgraph.size() - 1;
  while (i < length) {
    cur_group.clear();
    cur_group.push_back(subgraph[i]);
    int last_idx = subgraph[i].fuse_idx_;
    for (int j = i + 1; j < subgraph.size(); ++j) {
      if (last_idx + 1 != subgraph[j].fuse_idx_) {
        i = j;
        break;
      } else {
        cur_group.push_back(subgraph[j]);
        last_idx = subgraph[j].fuse_idx_;
        ++i;
      }
    }
    groups.push_back(cur_group);
  }
  // if there is only one conv in a group, it means no other conv can be fused
  // with just remove it
  auto iter = std::remove_if(
      groups.begin(), groups.end(), [](std::vector<NodeInfo> item) {
        return item.size() == 1;
      });
  groups.erase(iter, groups.end());
  fuse_accumulate(&groups);
  return groups;
}

std::vector<std::vector<NodeInfo>> FpgaConcatFuser::PatternMatch(
    SSAGraph* graph) {
  std::vector<std::vector<NodeInfo>> patterns;
  Node* upstream_op_node;
  std::vector<NodeInfo> subgraph;
  for (auto node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().op_info()->Type() == "concat") {
      subgraph.clear();

      int idx = 0;
      for (auto inconcat_node : node->inlinks) {
        upstream_op_node = inconcat_node->inlinks.front();
        int jump_info = enable_jump(upstream_op_node);
        if (jump_info && enable_fuse(inconcat_node)) {
          subgraph.push_back(NodeInfo(upstream_op_node, true, jump_info, idx));
          ++idx;
        } else {
          // TODO(chengruichang) Currently only the patterns that all upstream
          // ops of concat
          // support jump write are considered
          subgraph.clear();
          break;
        }
      }
      std::vector<std::vector<NodeInfo>> grouped_candidate =
          select_candidate(subgraph);
      // TODO(chengruichang) if concat op has multiple grouped upstream op that
      // can be
      // fused, try to support it later
      //            std::vector<NodeInfo> select_subgraph;
      if (grouped_candidate.size() == 1) {
        auto select_subgraph = grouped_candidate[0];
        int fuse_op_num = select_subgraph.size();
        for (int i = 0; i < fuse_op_num; ++i) {
          select_subgraph.push_back(NodeInfo(
              select_subgraph[i].node_->outlinks.front(), false, -1, i));
        }
        select_subgraph.push_back(NodeInfo(node, false, -1));
        int size = select_subgraph.size();
        VLOG(3) << "pattern found with " << size << " nodes";
        VLOG(3) << "pattern info with" << DebugPatternInfo(select_subgraph);
        patterns.push_back(select_subgraph);
      }
    }
  }
  return patterns;
}

size_t FpgaConcatFuser::operator()(SSAGraph* graph) {
  std::vector<std::vector<NodeInfo>> patterns = PatternMatch(graph);
  size_t num_patterns = patterns.size();
  if (num_patterns > 0) {
    InsertNewNode(graph, patterns);
    DeleteInterNodes(graph, patterns);
  }
  return num_patterns;
}

void FpgaConcatFuser::ExtractInputsOutputs(const std::vector<NodeInfo>& pattern,
                                           std::set<Node*>* input_var_nodes,
                                           std::set<Node*>* weight_var_nodes,
                                           std::set<Node*>* output_var_nodes) {
  // concat op to get output
  auto& concatnode = pattern[pattern.size() - 1].node_;
  for (auto* var_node : concatnode->outlinks)
    output_var_nodes->insert(var_node);
  for (auto& subgraph_nodeinfo : pattern) {
    Node* node = subgraph_nodeinfo.node_;
    if (node->IsStmt()) {
      bool wd_enable = subgraph_nodeinfo.wd_enable_;
      // if branch for conv
      if (wd_enable) {
        for (auto* var_node : node->inlinks) {
          if (var_node->AsArg().is_weight) {
            weight_var_nodes->insert(var_node);
          } else {
            input_var_nodes->insert(var_node);
          }
        }
      }
    }
  }
}

// in order to be consistent with graph, concat op with op to be concat are
// wrapped into sub graph
void FpgaConcatFuser::InsertNewNode(
    SSAGraph* graph, const std::vector<std::vector<NodeInfo>>& patterns) {
  // each pattern with a subgraph_op
  for (auto& subgraph_nodeinfos : patterns) {
    cpp::OpDesc subgraph_op_desc;
    subgraph_op_desc.SetType("subgraph");
    auto sub_program_desc = std::make_shared<cpp::ProgramDesc>();
    auto sub_block_desc = sub_program_desc->AddBlock<cpp::BlockDesc>();
    sub_block_desc->ClearOps();
    sub_block_desc->ClearVars();
    // subgraph opdescs
    NodeInfo concat_info = subgraph_nodeinfos[subgraph_nodeinfos.size() - 1];
    auto concat_opdesc = concat_info.node_->AsStmt().op_info();
    // concat op has only one output
    auto out_arg_name = concat_opdesc->Output("Out");
    for (auto& nodeinfo : subgraph_nodeinfos) {
      Node* node = nodeinfo.node_;
      bool wd_enable = nodeinfo.wd_enable_;
      if (node->IsStmt() && wd_enable) {
        auto sub_opdesc = node->AsStmt().mutable_op_info();
        std::string op_type = sub_opdesc->Type();
        // set new type such as conv2d --> fpga_conv2d
        sub_opdesc->SetType(FPGA_SUBOP_MAP[op_type]);
        // set attr to support jump write
        sub_opdesc->SetAttr<bool>("wd_enable", true);
        sub_opdesc->SetAttr<int>("wd_offset", nodeinfo.wd_offset_);
        sub_opdesc->SetAttr<int>("fuse_idx", nodeinfo.fuse_idx_);
        sub_opdesc->SetAttr<int>("original_out_channel",
                                 nodeinfo.original_out_channel_);
        sub_opdesc->SetAttr<int>("start_idx", nodeinfo.start_idx_);
        sub_opdesc->SetAttr<int>("end_idx", nodeinfo.end_idx_);
        // set the output of each conv to the output of concat
        // TODO(chengruichang) "Output" is a common attr name?
        sub_opdesc->SetOutput("Output", out_arg_name);
        auto sub_op_desc = sub_block_desc->AddOp<cpp::OpDesc>();
        // set this attr in order to pick the right kernel
        std::stringstream kerneltype;
        // TODO(chengruichang) is this the best way to support other kernel
        kerneltype << op_type << "/"
                   << "def"
                   << "/"
                   << "7"
                   << "/"
                   << "5"
                   << "/"
                   << "3";
        sub_opdesc->SetAttr<std::string>(kKernelTypeAttr, kerneltype.str());
        *sub_op_desc = *sub_opdesc;
      }
    }

    subgraph_op_desc.SetAttr<int32_t>("sub_block", 0);

    // prepare for inputs and outputs of the subgraph
    std::set<Node*> idata_var_nodes;
    std::set<Node*> weight_var_nodes;
    std::set<Node*> odata_var_nodes;
    ExtractInputsOutputs(subgraph_nodeinfos,
                         &idata_var_nodes,
                         &weight_var_nodes,
                         &odata_var_nodes);
    std::set<Node*> input_var_nodes(idata_var_nodes.begin(),
                                    idata_var_nodes.end());
    input_var_nodes.insert(weight_var_nodes.begin(), weight_var_nodes.end());
    std::set<Node*> output_var_nodes(odata_var_nodes.begin(),
                                     odata_var_nodes.end());

    // Set input and output name mapping which stores the real inputs and
    // outputs
    std::vector<std::string> idata_var_names;
    std::vector<std::string> odata_var_names;
    for (auto& var_node : idata_var_nodes) {
      idata_var_names.push_back(var_node->AsArg().name);
    }
    for (auto& var_node : odata_var_nodes) {
      odata_var_names.push_back(var_node->AsArg().name);
    }
    subgraph_op_desc.SetAttr<std::vector<std::string>>("input_data_names",
                                                       idata_var_names);
    subgraph_op_desc.SetAttr<std::vector<std::string>>("output_data_names",
                                                       odata_var_names);

    std::vector<std::string> input_var_names;
    std::vector<std::string> output_var_names;
    for (auto& var_node : input_var_nodes) {
      input_var_names.push_back(var_node->AsArg().name);
    }
    for (auto& var_node : output_var_nodes) {
      output_var_names.push_back(var_node->AsArg().name);
    }
    subgraph_op_desc.SetInput("Inputs", input_var_names);
    subgraph_op_desc.SetOutput("Outputs", output_var_names);

    // construct sub graph op
    auto subgraph_op = LiteOpRegistry::Global().Create("subgraph");
    static_cast<operators::SubgraphOp*>(subgraph_op.get())
        ->SetProgramDesc(sub_program_desc);
    auto any_op = subgraph_nodeinfos[0].node_->AsStmt().op();
    subgraph_op->Attach(subgraph_op_desc, any_op->scope());

    // Create and add a new subgraph node into the graph
    auto subgraph_op_node =
        graph->GraphCreateInstructNode(subgraph_op, any_op->valid_places());
    for (auto& var_node : input_var_nodes) {
      IR_NODE_LINK_TO(var_node, subgraph_op_node);
    }
    for (auto& var_node : output_var_nodes) {
      IR_OP_VAR_LINK(subgraph_op_node, var_node);
    }
  }
}

void FpgaConcatFuser::DeleteInterNodes(
    SSAGraph* graph, const std::vector<std::vector<NodeInfo>>& patterns) {
  std::set<const Node*> nodes2rm;
  for (auto each_pattern : patterns) {
    for (auto node_info : each_pattern) {
      Node* node = node_info.node_;
      nodes2rm.insert(node);
    }
  }
  VLOG(3) << nodes2rm.size() << " pattern nodes deleted";
  GraphSafeRemoveNodes(graph, nodes2rm);
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
