/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "framework/program/program-optimize/program_optimize.h"
#include <algorithm>
#include <utility>
#include "framework/program/program-optimize/fusion_op_register.h"

namespace paddle_mobile {

namespace framework {

std::shared_ptr<ProgramDesc> ProgramOptimize::FusionOptimize(
    std::shared_ptr<ProgramDesc> ori_des, bool add_split) {
  std::shared_ptr<ProgramDesc> optimize_program =
      std::make_shared<ProgramDesc>(*ori_des);
  current_block_ = optimize_program->Blocks().size();

  for (int i = 0; i < optimize_program->Blocks().size(); ++i) {
    std::unordered_map<std::string, std::shared_ptr<Node>> output_nodes;
    std::unordered_map<
        std::string,
        std::vector<
            std::pair<std::shared_ptr<Node>,
                      std::unordered_map<std::string, std::shared_ptr<Node>>>>>
        type_map;
    std::vector<std::shared_ptr<Node>> nodes;
    std::shared_ptr<Node> begin_node;

    auto block = optimize_program->Block(i);
    for (int j = 0; j < block->Ops().size(); ++j) {
      auto op = block->Ops()[j];
      std::shared_ptr<Node> node = std::make_shared<Node>(op);
      if (j == 0) {
        begin_node = node;
      }

      const std::string op_type = op->Type();
      nodes.push_back(node);
      type_map[op_type].push_back({node, output_nodes});
      const VariableNameMap &op_inputs = op->GetInputs();
      const VariableNameMap &op_outpus = op->GetOutputs();

      for (const auto &input : op_inputs) {
        for (const auto &input_name : input.second) {
          if (output_nodes.find(input_name) != output_nodes.end()) {
            auto input_node = output_nodes[input_name];
            *input_node > node;
          }
        }
      }

      for (const auto &output : op_outpus) {
        for (const auto &output_name : output.second) {
          output_nodes[output_name] = node;
        }
      }
    }

    for (auto &registed : FusionOpRegister::Instance()->Matchers()) {
      std::string fusion_type = registed->Type();
      std::shared_ptr<FusionOpMatcher> matcher = registed;

      auto match_vector = type_map[matcher->BeginType()];

      for (auto &match_node_pair : match_vector) {
        auto match_node = match_node_pair.first;

        auto node_has = match_node_pair.second;

        auto depth = matcher->BeginNode().Depth();
        auto sub_node = match_node->To(depth);
        //  DLOG << " sub node: " << *sub_node;
        if (*sub_node == matcher->BeginNode()) {
          bool can_folder = true;

          auto relationship_map = sub_node->Relationship();

          for (auto to_check : matcher->NeedCheck()) {
            auto nodes = (*sub_node)[to_check.first];
            for (auto node : nodes) {
              auto inputs_to_check =
                  node->OpDescOfNode()->Input(to_check.second);

              for (auto input_to_check : inputs_to_check) {
                if (node_has.find(input_to_check) == node_has.end()) {
                  if (relationship_map.find(input_to_check) ==
                      relationship_map.end()) {
                    can_folder = false;
                  } else {
                  }
                }
              }
            }
          }

          if (!can_folder) {
            continue;
          }

          std::vector<std::shared_ptr<Node>> removed_nodes;
          matcher->FolderNodes(match_node.get(), &removed_nodes);
          for (int k = removed_nodes.size() - 1; k >= 0; --k) {
            auto removed_node = removed_nodes[k];
            auto removed_ite =
                std::find(nodes.begin(), nodes.end(), removed_node);
            if (removed_ite != nodes.end()) {
              nodes.erase(removed_ite);
            }
          }
        }
      }
    }

    std::vector<std::shared_ptr<framework::OpDesc>> op_descs;
    if (add_split) {
      GenerateOps(&op_descs, begin_node.get(), add_split);
    } else {
      for (int m = 0; m < nodes.size(); ++m) {
        auto &node = nodes[m];
        op_descs.push_back(node->op_desc_);
      }
    }
    block->ops_ = op_descs;
  }

  for (int m = 0; m < new_blocks_.size(); ++m) {
    std::shared_ptr<BlockDesc> new_block = new_blocks_[m];
    new_block->index_ = m + ori_des->blocks_.size();
    optimize_program->blocks_.push_back(new_block);
  }
  return optimize_program;
}

void ProgramOptimize::GenerateOps(
    std::vector<std::shared_ptr<framework::OpDesc>> *op_desc, Node *input_node,
    Node *current_node) {
  if (current_node->inputs_.size() > 1 &&
      input_node != current_node->inputs_.back()) {
    DLOG << " current type " << current_node->Type();

    DLOG << " inputs size of current node > 0 ";

    for (int i = 0; i < current_node->inputs_.size(); ++i) {
      DLOG << " input i: " << current_node->inputs_[i]->Type();
    }

    return;
  } else if (current_node->inputs_.size() > 1 &&
             input_node == current_node->inputs_.back()) {
    op_desc->push_back(current_node->op_desc_);
  } else {
    op_desc->push_back(current_node->op_desc_);
  }

  for (int i = 0; i < current_node->outputs_.size(); ++i) {
    auto &output = current_node->outputs_[i];
    GenerateOps(op_desc, current_node, output.get());
  }
}

void ProgramOptimize::GenerateOps(
    std::vector<std::shared_ptr<framework::OpDesc>> *op_desc, Node *input_node,
    Node *current_node, bool adding_thread, int thread_num,
    std::shared_ptr<BlockDesc> new_block) {
  if (current_node->outputs_.size() > 1) {
    adding_thread = false;
  }

  bool can_add_split = false;
  const auto current_desc = current_node->OpDescOfNode();
  const VariableNameMap &current_op_inputs = current_desc->GetInputs();
  const VariableNameMap &current_op_outputs = current_desc->GetOutputs();
  // 如果当前节点有多个输出 并且 只有当前节点对应的 op_desc_ 输出数为 1 时支持
  if (current_node->outputs_.size() > 1 && current_op_outputs.size() == 1) {
    can_add_split = true;

    // 遍历当前节点的 output 节点
    for (const auto &output : current_node->outputs_) {
      // 不支持 output 有多个 output 的情况
      if (output->outputs_.size() > 1) {
        DLOG << "don't support multi output of output";
        can_add_split = false;
        break;
      }

      //与节点关联的 OpDesc
      std::shared_ptr<framework::OpDesc> &op_desc = output->op_desc_;
      //获取这个 op 的 inputs key 和 outputs key
      const VariableNameMap &op_inputs = op_desc->GetInputs();
      const VariableNameMap &op_outputs = op_desc->GetOutputs();

      //判断现在 是否存在这个 op
      //判断这个 output 和 input key 的 size 等于 1
      if (op_outputs.size() == 1 && op_inputs.size() == 1) {
        auto inputs_of_output = op_inputs.begin()->second;
        auto outputs_of_output = op_outputs.begin()->second;

        // 判断一下, 如果输入和输出没有同名, 是支持的
        for (int i = 0; i < inputs_of_output.size(); ++i) {
          std::string input_of_output = inputs_of_output[i];
          for (int j = 0; j < outputs_of_output.size(); ++j) {
            std::string output_of_output = outputs_of_output[j];
            if (input_of_output == output_of_output) {
              DLOG << "output的 output 包含 input" << input_of_output;
              can_add_split = false;
              break;
            }
          }
        }
      } else {  // 如果模型中包含没有的 op, 则不支持添加 split
        DLOG << "找不到 这个 op 类型: " << output->op_desc_->Type();
        can_add_split = false;
      }
    }
  }

  if (current_node->inputs_.size() > 1 &&
      input_node != current_node->inputs_.back()) {
    return;
  } else if (current_node->inputs_.size() > 1 &&
             input_node == current_node->inputs_.back()) {
    new_block.reset();
    adding_thread = false;
    op_desc->push_back(current_node->op_desc_);
  } else {
    if (new_block.get() && adding_thread) {
      new_block->ops_.push_back(current_node->op_desc_);
    } else {
      op_desc->push_back(current_node->op_desc_);
    }
  }
  if (adding_thread) {
    Attribute attr;
    attr.Set<int>(thread_num);
    current_node->op_desc_->attrs_["thread"] = attr;
  }

  if (can_add_split) {
    new_block = std::make_shared<BlockDesc>();
    new_block->multi_thread_ = true;
    new_block->index_ = current_block_;
    new_blocks_.push_back(new_block);

    adding_thread = true;
    std::shared_ptr<OpDesc> split_op_desc = std::make_shared<OpDesc>();
    split_op_desc->type_ = G_OP_TYPE_SPLIT;
    auto outputs = current_node->op_desc_->Output(
        op_input_output_key[current_node->op_desc_->Type()].second[0]);
    split_op_desc->inputs_ = {
        {op_input_output_key[G_OP_TYPE_SPLIT].first[0], outputs}};
    auto &split_outputs =
        split_op_desc->outputs_[op_input_output_key[G_OP_TYPE_SPLIT].second[0]];
    for (const auto &output : current_node->outputs_) {
      split_outputs.push_back(outputs[0]);
    }

    Attribute attr;
    attr.Set<int>(current_block_);
    split_op_desc->attrs_["block_id"] = attr;

    op_desc->push_back(split_op_desc);
    current_block_++;
  }

  for (int i = 0; i < current_node->outputs_.size(); ++i) {
    auto &output = current_node->outputs_[i];
    if (can_add_split) {
      GenerateOps(op_desc, current_node, output.get(), adding_thread, i,
                  new_block);
    } else {
      GenerateOps(op_desc, current_node, output.get(), adding_thread,
                  thread_num, new_block);
    }
  }
}

void ProgramOptimize::GenerateOps(
    std::vector<std::shared_ptr<framework::OpDesc>> *op_descs, Node *begin_node,
    bool can_add_split) {
  if (can_add_split) {
    this->GenerateOps(op_descs, begin_node, begin_node, false, -1, nullptr);
  } else {
    this->GenerateOps(op_descs, begin_node, begin_node);
  }
}

}  // namespace framework
}  // namespace paddle_mobile
