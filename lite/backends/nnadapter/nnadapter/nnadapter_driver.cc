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

#include "nnadapter_driver.h"  // NOLINT
#include <map>
#include <utility>

namespace nnadapter {
namespace driver {

std::vector<Operation*> sortOperationsInTopologicalOrder(Graph* graph) {
  std::vector<Operation*> operations;  // Operations in topological order
  std::vector<Operation*> queue;
  // Use to find all of adjacent operations according to a given operand.
  std::multimap<Operand*, Operation*> map;
  // The counters of variable inputs for all of operations.
  std::map<Operation*, uint32_t> counts;
  for (auto& operation : graph->operations) {
    uint32_t count = 0;
    for (auto operand : operation.inputs) {
      auto lifetime = operand->type.lifetime;
      if (lifetime == NNADAPTER_TEMPORARY_VARIABLE ||
          lifetime == NNADAPTER_OUTPUT) {
        count++;
        map.insert(std::pair<Operand*, Operation*>(operand, &operation));
      }
    }
    if (count == 0) {
      // The operation which only depends the model inputs and constants
      queue.push_back(&operation);
    }
    counts[&operation] = count;
  }
  while (queue.size() > 0) {
    auto operation = queue.back();
    queue.pop_back();
    operations.push_back(operation);
    for (auto operand : operation->outputs) {
      auto range = map.equal_range(operand);
      for (auto i = range.first; i != range.second; i++) {
        uint32_t& count = counts[i->second];
        if (--count == 0) {
          queue.push_back(i->second);
        }
      }
    }
  }
  return operations;
}

}  // namespace driver
}  // namespace nnadapter
