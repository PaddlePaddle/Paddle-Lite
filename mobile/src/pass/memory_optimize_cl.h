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
#ifdef PADDLE_MOBILE_CL

#pragma once

#include <stack>
#include <string>
#include <unordered_map>
#include <vector>
#include "framework/cl/cl_image_converter.h"
#include "framework/lod_tensor.h"
#include "framework/program/program.h"
#include "pass/pass_base.h"

// use for opencl
namespace paddle_mobile {
namespace pass {

typedef struct {
  std::string name;  // variable name
  int count;         // reference count
  bool visited;
} ClVarNode;

// MemoryOptPass will analyze the program, and reuse memory between
// variables as much as possible
class MemoryOptPassCl : public PassBase {
 public:
  MemoryOptPassCl() {}
  virtual ~MemoryOptPassCl() {
    for (auto &it : created_nodes_) {
      delete it.second;
    }
    delete normal_converter;
  }

  void operator()(const framework::ProgramDesc *program,
                  framework::Scope *scope,
                  MemoryOptimizationLevel memory_optimization_level,
                  framework::DDim dims = {});

  void AppendBlockVars(const framework::BlockDesc *block);

  bool IsPersistable(const std::string name);

  ClVarNode *CreateNode(const std::string name);

  void ShareData(framework::Scope *scope,
                 MemoryOptimizationLevel memory_optimization_level,
                 framework::DDim dims) const;

 private:
  std::stack<ClVarNode *> analysis_nodes_;
  std::vector<std::vector<ClVarNode *>> reused_nodes_;
  std::unordered_map<std::string, ClVarNode *> created_nodes_;
  std::unordered_map<std::string, framework::VarDesc *> block_vars_;
  paddle_mobile::framework::CLImageConverterNormal *normal_converter =
      new paddle_mobile::framework::CLImageConverterNormal();
};

}  // namespace pass
}  // namespace paddle_mobile
#endif
