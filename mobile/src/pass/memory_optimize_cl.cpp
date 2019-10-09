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
#include "pass/memory_optimize_cl.h"
#include <algorithm>
#include "framework/cl/cl_image.h"
#include "framework/lod_tensor.h"
namespace paddle_mobile {
namespace pass {

void MemoryOptPassCl::AppendBlockVars(const framework::BlockDesc *block) {
  // block_vars_.clear();
  for (const auto var : block->Vars()) {
    block_vars_[var->Name()] = var.get();
  }
}

bool MemoryOptPassCl::IsPersistable(const std::string name) {
  const auto it = block_vars_.find(name);
  if (it != block_vars_.end()) {
    return it->second->Persistable();
  }
  return false;
}

ClVarNode *MemoryOptPassCl::CreateNode(const std::string name) {
  auto it = created_nodes_.find(name);
  if (it != created_nodes_.end()) {
    ++(it->second->count);
    return it->second;
  }
  ClVarNode *var = new ClVarNode;
  var->name = name;
  var->count = 1;
  var->visited = false;
  created_nodes_[name] = var;
  return var;
}

void MemoryOptPassCl::operator()(
    const framework::ProgramDesc *program, framework::Scope *scope,
    MemoryOptimizationLevel memory_optimization_level,
    framework::DDim target_dims) {
  const auto &blocks = program->Blocks();
  for (const auto &block : blocks) {
    // access all variables in each block
    AppendBlockVars(block.get());
    reused_nodes_.clear();
    // collect all not persistable variables, and accumulate
    // it's reference count
    std::stack<ClVarNode *> empty_var_nodes;
    analysis_nodes_.swap(empty_var_nodes);

    std::vector<std::string> exclude_var_names;
    for (const auto &op : block->Ops()) {
      for (const auto &inputs : op->GetInputs()) {
        for (const auto &input : inputs.second) {
          if (!IsPersistable(input)) {
            if (memory_optimization_level == MemoryOptimizationWithoutFeeds) {
              if (op->Type() == "feed") {
                exclude_var_names.push_back(input);
              }
            }
          }
        }
      }
    }

    std::vector<ClVarNode *> fetch_var_nodes;
    for (const auto &op : block->Ops()) {
      DLOG << "op_desc->Type(): " << op->Type();
      for (const auto &outputs : op->GetOutputs()) {
        for (const auto &output : outputs.second) {
          // not a persistable and not a exclude one ,then add it to
          // analysis_nodes
          if (!IsPersistable(output) &&
              std::find(exclude_var_names.begin(), exclude_var_names.end(),
                        output) == exclude_var_names.end()) {
            DLOG << "output: " << output;
            ClVarNode *node = CreateNode(output);
            analysis_nodes_.push(node);
          }
        }
      }
      for (const auto &inputs : op->GetInputs()) {
        for (const auto &input : inputs.second) {
          // not a persistable and not a exclude one ,then add it to
          // analysis_nodes
          if (!IsPersistable(input) &&
              std::find(exclude_var_names.begin(), exclude_var_names.end(),
                        input) == exclude_var_names.end()) {
            DLOG << "input: " << input;
            ClVarNode *node = CreateNode(input);
            analysis_nodes_.push(node);
            if (op->Type() == "fetch") {
              fetch_var_nodes.push_back(node);
            }
          }
        }
      }
      for (const auto &outputs : op->GetOutputs()) {
        for (const auto &output : outputs.second) {
          if (!IsPersistable(output) &&
              std::find(exclude_var_names.begin(), exclude_var_names.end(),
                        output) == exclude_var_names.end()) {
            DLOG << "output: " << output;
            ClVarNode *node = CreateNode(output);
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
          // reference count = 0 and not in fetch list
          if (list.back()->count == 0 &&
              std::find(fetch_var_nodes.begin(), fetch_var_nodes.end(),
                        list.back()) == fetch_var_nodes.end()) {
            list.push_back(node);
            reused = true;
            break;
          }
        }
        // create new list if can't find a reused list
        if (!reused) {
          std::vector<ClVarNode *> list;
          list.push_back(node);
          reused_nodes_.push_back(std::move(list));
        }
      }
      node->visited = true;
      node->count -= 1;
    }
    // shared data within all variables in the same reused list
    ShareData(scope, memory_optimization_level, target_dims);
  }
}

void MemoryOptPassCl::ShareData(
    framework::Scope *scope, MemoryOptimizationLevel memory_optimization_level,
    framework::DDim target_dims)
    const {  // shared data within all variables in the same reused list
  cl_context context = scope->GetCLScpoe()->Context();
  cl_command_queue command_queue = scope->GetCLScpoe()->CommandQueue();

  for (const auto &list : reused_nodes_) {
    DLOG << "\n";
    DLOG << "gpu . share memory within these variables";
    int64_t x_based_max_numl = -1;
    int64_t y_based_max_numl = -1;
    int64_t x_based_max_x = -1;
    int64_t x_based_max_y = -1;
    int64_t y_based_max_x = -1;
    int64_t y_based_max_y = -1;

    framework::CLImage *x_based_reuse_tensor = nullptr;
    framework::CLImage *y_based_reuse_tensor = nullptr;
    for (const auto &node : list) {
      auto *var = scope->Var(node->name);
      auto *tensor = var->template GetMutable<framework::CLImage>();
      const int64_t numl = tensor->numel();
      auto origin_tensor_dims = tensor->dims();

      // for super ,hack origin dims
      if (target_dims.size() == 4) {
        PADDLE_MOBILE_ENFORCE(origin_tensor_dims.size() == 4,
                              "tensor dims must be equal to 4");
        origin_tensor_dims = {origin_tensor_dims[0], origin_tensor_dims[1],
                              target_dims[2], target_dims[3]};
        tensor->Resize(origin_tensor_dims);
      }

      const framework::DDim &image_dims =
          normal_converter->InitImageDimInfoWith(origin_tensor_dims);
      int64_t image_dims_x = image_dims[0];
      int64_t image_dims_y = image_dims[1];
      // classify memory into two parts
      if (image_dims_x > image_dims_y) {
        // choose a biggest tensor for reuse
        if (x_based_max_numl < numl) {
          x_based_max_numl = numl;
          x_based_reuse_tensor = tensor;
        }
        x_based_max_x = std::max(x_based_max_x, image_dims_x);
        x_based_max_y = std::max(x_based_max_y, image_dims_y);
      } else {
        // choose a biggest tensor for reuse
        if (y_based_max_numl < numl) {
          y_based_max_numl = numl;
          y_based_reuse_tensor = tensor;
        }
        y_based_max_x = std::max(y_based_max_x, image_dims_x);
        y_based_max_y = std::max(y_based_max_y, image_dims_y);
      }
    }

    PADDLE_MOBILE_ENFORCE(
        x_based_reuse_tensor != nullptr || y_based_reuse_tensor != nullptr,
        "x_based_reuse_tensor and y_based_reuse_tensor can not be null at same "
        "time");

    // init x based shared cl mem
    if (x_based_reuse_tensor != nullptr) {
      const framework::DDim &x_reuse_dims = x_based_reuse_tensor->dims();
      x_based_reuse_tensor->InitFakeSizeImage(
          context, command_queue, x_reuse_dims, {x_based_max_x, x_based_max_y});
    }

    // init y based shared cl mem
    if (y_based_reuse_tensor != nullptr) {
      const framework::DDim &y_reuse_dims = y_based_reuse_tensor->dims();
      y_based_reuse_tensor->InitFakeSizeImage(
          context, command_queue, y_reuse_dims, {y_based_max_x, y_based_max_y});
    }
    // share mem
    for (const auto &node : list) {
      auto *var = scope->Var(node->name);
      auto *tensor = var->template GetMutable<framework::CLImage>();
      auto need_dims = tensor->dims();

      // for super ,hack origin dims
      if (target_dims.size() == 4) {
        need_dims = {need_dims[0], need_dims[1], target_dims[2],
                     target_dims[3]};
      }

      const framework::DDim &need_image_dims =
          normal_converter->InitImageDimInfoWith(need_dims);
      int64_t image_dims_x = need_image_dims[0];
      int64_t image_dims_y = need_image_dims[1];

      if (image_dims_x > image_dims_y) {
        PADDLE_MOBILE_ENFORCE(x_based_reuse_tensor != nullptr,
                              "x_based_reuse_tensor not null here");
        tensor->InitWithExistMem(context, command_queue, need_dims,
                                 *x_based_reuse_tensor);
      } else {
        PADDLE_MOBILE_ENFORCE(y_based_reuse_tensor != nullptr,
                              "y_based_reuse_tensor not null here");
        tensor->InitWithExistMem(context, command_queue, need_dims,
                                 *y_based_reuse_tensor);
      }
    }
  }
}

}  // namespace pass
}  // namespace paddle_mobile
#endif
