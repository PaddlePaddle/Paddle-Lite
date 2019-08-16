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

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "framework/operator.h"
#include "framework/program/program-optimize/node.h"

namespace paddle_mobile {
namespace framework {

class FusionOpRegister {
 public:
  static FusionOpRegister* Instance() {
    static FusionOpRegister* regist = nullptr;
    if (regist == nullptr) {
      regist = new FusionOpRegister();
    }
    return regist;
  }

  void regist(FusionOpMatcher* matcher) {
    if (matchers_.find(matcher->Type()) != matchers_.end()) {
      return;
    }

    std::shared_ptr<FusionOpMatcher> shared_matcher(matcher);
    matchers_[matcher->Type()] = shared_matcher;
  }

  const std::vector<std::shared_ptr<FusionOpMatcher>> Matchers() {
    std::vector<std::shared_ptr<FusionOpMatcher>> matchers;
    for (const auto& match : matchers_) {
      matchers.push_back(match.second);
    }
    std::sort(matchers.begin(), matchers.end(),
              [](std::shared_ptr<FusionOpMatcher> first,
                 std::shared_ptr<FusionOpMatcher> second) {
                return first->BeginNode().Depth() > second->BeginNode().Depth();
              });
    return matchers;
  }

 private:
  std::map<std::string, std::shared_ptr<FusionOpMatcher>> matchers_;
  FusionOpRegister() {}
};

class FusionOpRegistrar {
 public:
  explicit FusionOpRegistrar(FusionOpMatcher* matcher) {
    FusionOpRegister::Instance()->regist(matcher);
  }
  void Touch() {}
};

}  // namespace framework
}  // namespace paddle_mobile

#define REGISTER_FUSION_MATCHER(op_type, matcher)          \
  static paddle_mobile::framework::FusionOpRegistrar       \
      __fusion_matcher_registrar_##op_type(new matcher()); \
  int TouchFusionMatcherRegistrar_##op_type() {            \
    __fusion_matcher_registrar_##op_type.Touch();          \
    return 0;                                              \
  }
