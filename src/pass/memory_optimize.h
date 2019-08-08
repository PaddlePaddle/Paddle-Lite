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

#pragma once

#include <stack>
#include <string>
#include <unordered_map>
#include <vector>
#include "framework/program/program.h"
#include "pass/pass_base.h"

namespace paddle_mobile {
namespace pass {

typedef struct {
  std::string name;  // variable name
  int count;         // reference count
  bool visited;
} VarNode;

// MemoryOptPass will analyze the program, and reuse memory between
// variables as much as possible
class MemoryOptPass : public PassBase {
 public:
  MemoryOptPass() {}
  virtual ~MemoryOptPass() {
    for (auto &it : created_nodes_) {
      delete it.second;
    }
  }

  void operator()(const framework::ProgramDesc *program,
                  framework::Scope *scope,
                  MemoryOptimizationLevel memory_optimization_level);

  void AppendBlockVars(const framework::BlockDesc *block);

  bool IsPersistable(const std::string name);

  VarNode *CreateNode(const std::string name);

  void AdjustMemory();

 private:
  std::stack<VarNode *> analysis_nodes_;
  std::vector<std::vector<VarNode *>> reused_nodes_;
  std::unordered_map<std::string, VarNode *> created_nodes_;
  std::unordered_map<std::string, framework::VarDesc *> block_vars_;
  std::vector<framework::Variable *> memoryDeputies_;
};

}  // namespace pass
}  // namespace paddle_mobile
