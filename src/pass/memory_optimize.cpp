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

#include "pass/memory_optimize.h"
#include "framework/lod_tensor.h"

namespace paddle_mobile {
namespace pass {

void MemoryOptPass::InitBlockVars(const framework::BlockDesc *block) {
  block_vars_.clear();
  for (const auto var : block->Vars()) {
    block_vars_[var->Name()] = var.get();
  }
}

bool MemoryOptPass::IsPersistable(const std::string name) {
  const auto it = block_vars_.find(name);
  if (it != block_vars_.end()) {
    return it->second->Persistable();
  }
  return false;
}

VarNode *MemoryOptPass::CreateNode(const std::string name) {
  auto it = created_nodes_.find(name);
  if (it != created_nodes_.end()) {
    ++(it->second->count);
    return it->second;
  }
  VarNode *var = new VarNode;
  var->name = name;
  var->count = 1;
  var->visited = false;
  created_nodes_[name] = var;
  return var;
}

void MemoryOptPass::operator()(const framework::ProgramDesc *program,
                               framework::Scope *scope) {
  const auto &blocks = program->Blocks();
  for (const auto &block : blocks) {
    // access all variables in block, and stored in map
    InitBlockVars(block.get());

    visited_nodes_.clear();
    reused_nodes_.clear();
    // collect all not persistable variables, and accumulate
    // it's reference count
    std::stack<VarNode *> empty_var_nodes;
    analysis_nodes_.swap(empty_var_nodes);

    for (const auto &op : block->Ops()) {
      DLOG << "op_desc->Type(): " << op->Type();
      const auto &outputs_map = op->GetOutputs();
      for (const auto &outputs : outputs_map) {
        for (const auto &output : outputs.second) {
          if (!IsPersistable(output)) {
            DLOG << "output: " << output;
            VarNode *node = CreateNode(output);
            analysis_nodes_.push(node);
          }
        }
      }
      const auto &inputs_map = op->GetInputs();
      for (const auto &inputs : inputs_map) {
        for (const auto &input : inputs.second) {
          if (!IsPersistable(input)) {
            DLOG << "input: " << input;
            VarNode *node = CreateNode(input);
            analysis_nodes_.push(node);
          }
        }
      }
    }

    // apply optimize
    while (!analysis_nodes_.empty()) {
      auto *node = analysis_nodes_.top();
      analysis_nodes_.pop();
      // only not visited node can reuse memory between other nodes
      // with 0 count which indicate they will not be used any more
      if (!node->visited) {
        bool reused = false;
        // find out a possable reuse list
        for (auto &list : reused_nodes_) {
          if (list.back()->count == 0) {
            list.push_back(node);
            reused = true;
            break;
          }
        }
        // create new list if can't find a reused list
        if (!reused) {
          std::vector<VarNode *> list;
          list.push_back(node);
          reused_nodes_.push_back(std::move(list));
        }
      }
      node->visited = true;
      node->count -= 1;
    }
  }
  // shared data within all variables in the same reused list
  for (const auto &list : reused_nodes_) {
    DLOG << "\n";
    DLOG << "share data within these variables";
    std::string name = list[0]->name;
    auto *reused_var = scope->Var(name);
    auto *reuse_tensor =
        reused_var->template GetMutable<framework::LoDTensor>();
    reuse_tensor->mutable_data<float>();
    for (const auto &node : list) {
      DLOG << node->name;
      auto *var = scope->Var(node->name);
      auto *tensor = var->template GetMutable<framework::LoDTensor>();
      tensor->ShareDataWith(*reuse_tensor);
    }
  }
}

}  // namespace pass
}  // namespace paddle_mobile
