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

#ifdef PADDLE_EXECUTOR_MULTITHREAD
#include <string>
#include <unordered_map>
#include <vector>
#include "framework/operator.h"

namespace paddle_mobile {

class depCore {
 public:
  template <typename Dtype>
  void analysisDep(
      const std::vector<std::shared_ptr<framework::OperatorBase<Dtype>>>& ops) {
    std::unordered_map<std::string, int> vars;
    size_t nop = ops.size();
    deps.resize(nop);
    next.resize(nop);
    for (size_t i = 0; i < nop; i++) {
      const auto& op = ops[i];
      for (const auto& kv : op->Inputs()) {
        for (const auto& v : kv.second) {
          if (vars.find(v) == vars.end()) {
            continue;
          }
          int di = vars[v];
          if (di == i) {
            continue;
          }
          if (std::find(deps[i].begin(), deps[i].end(), di) != deps[i].end()) {
            continue;
          }
          deps[i].push_back(di);
          next[di].push_back(i);
        }
      }
      for (const auto& kv : op->Outputs()) {
        for (const auto& v : kv.second) {
          vars[v] = i;
        }
      }
    }
  }
  const std::vector<int>& getNext(int i) { return next[i]; }
  const std::vector<int>& getDeps(int i) { return deps[i]; }
  std::vector<std::vector<int>> deps;
  std::vector<std::vector<int>> next;
};

}  // namespace paddle_mobile

#endif
