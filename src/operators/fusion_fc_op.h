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

#include <string>

#include "framework/operator.h"
#include "framework/program/program-optimize/fusion_op_register.h"

namespace paddle_mobile {
namespace operators {

class FusionFcMatcher : public framework::FusionOpMatcher {
 public:
  FusionFcMatcher() {
    node_ = framework::Node("mul");
    node_ > std::make_shared<framework::Node>("elementwise_add");
  }

  void FolderNodes(framework::Node &node) {
    std::vector<std::shared_ptr<framework::OpDesc>> origin_descs =
        node.OpDescs(node_.Depth());
    node.Folder(node_.Depth(), Type(), {{"elementwise_add", {"Y", "Z"}}});
  }

  std::string Type() { return "fc"; }
};

class FusionFcOp {
 public:
 private:
};

static framework::FusionOpRegistrar fc_registrar(new FusionFcMatcher());

}  // namespace operators
}  // namespace paddle_mobile
