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

#include "lite/core/mir/pass_utils.h"
#include <set>
#include <string>
#include <unordered_map>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {

using lite_api::Place;

void ExpandPlaces(std::set<Place>* places, const Place& place) {
  for (const auto& target : lite_api::ExpandValidTargets(place.target)) {
    for (const auto& precision :
         lite_api::ExpandValidPrecisions(place.precision)) {
      for (const auto& layout : lite_api::ExpandValidLayouts(place.layout)) {
        places->insert(Place(target, precision, layout));
      }
    }
  }
}

bool KernelRegistered(const std::string name, const Place& place) {
  std::set<Place> places;
  ExpandPlaces(&places, place);
  for (const auto& p : places) {
    if (!KernelRegistry::Global()
             .Create(name, p.target, p.precision, p.layout)
             .empty()) {
      return true;
    }
  }
  return false;
}

bool PassMatchesTarget(const mir::Pass& pass, TargetType target) {
  const auto& targets = pass.Targets();
  if (targets.find(TARGET(kAny)) != targets.end()) return true;
  return (targets.find(target) != targets.end());
}

bool PassMatchesKernels(const mir::Pass& pass) {
  const auto& kernels = pass.GetBoundKernels();
  for (const auto& kernel : kernels) {
    for (const auto& place : kernel.second) {
      if (!KernelRegistered(kernel.first, place)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace lite
}  // namespace paddle
