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
#include <map>
#include <set>
#include <string>
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

bool PassMatchesTarget(const mir::Pass& pass,
                       const std::set<TargetType>& targets) {
  // Whether the pass is suitable for targets ? The condition is the
  // intersection of targets and pass's bound targets is not empty, besides the
  // intersection of targets and pass's excluded targets is empty. The formula
  // is as follows: matched = !empty(targets ^ pass.bound_targets) &&
  // empty(targets ^ pass.excluded_targets), where ^ is intersection operation.
  const auto& bound_targets = pass.BoundTargets();
  bool matched = bound_targets.find(TARGET(kAny)) != bound_targets.end();
  std::set<TargetType> inter_bound_targets;
  std::set_intersection(
      bound_targets.begin(),
      bound_targets.end(),
      targets.begin(),
      targets.end(),
      std::inserter(inter_bound_targets, inter_bound_targets.begin()));
  matched |= !inter_bound_targets.empty();
  const auto& excluded_targets = pass.ExcludedTargets();
  matched &= excluded_targets.find(TARGET(kAny)) == excluded_targets.end();
  std::set<TargetType> inter_excluded_targets;
  std::set_intersection(
      excluded_targets.begin(),
      excluded_targets.end(),
      targets.begin(),
      targets.end(),
      std::inserter(inter_excluded_targets, inter_excluded_targets.begin()));
  matched &= inter_excluded_targets.empty();
  return matched;
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
