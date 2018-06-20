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

#ifdef CONVADDRELU_OP

#pragma once

#include "framework/operator.h"
#include "framework/program/program-optimize/fusion_op_register.h"

namespace paddle_mobile {
namespace operators {

class FushionConvAddReluOpMatcher : public framework::FusionOpMatcher {
 public:
  FushionConvAddReluOpMatcher() {
    node_ = framework::Node(G_OP_TYPE_CONV);
    node_ > std::make_shared<framework::Node>(G_OP_TYPE_ELEMENTWISE_ADD) >
        std::make_shared<framework::Node>(G_OP_TYPE_RELU);
  }

  void FolderNodes(
      framework::Node *node,
      std::vector<std::shared_ptr<framework::Node>> *removed_nodes) {
    std::vector<std::shared_ptr<framework::OpDesc>> origin_descs =
        node->OpDescs(node_.Depth());
    node->Folder(node_.Depth(), Type(),
                 {{G_OP_TYPE_ELEMENTWISE_ADD, {"Y", "Z"}}}, removed_nodes);
  }
  std::string Type() { return G_OP_TYPE_FUSION_CONV_ADD_RELU; }
};

class ConvAddReluOp {
 public:
 private:
};

#ifdef PADDLE_MOBILE_CPU
// static framework::FusionOpRegistrar fusion_conv_add_relu_registrar(
//        new FushionConvAddReluOpMatcher());
#endif
#ifdef PADDLE_MOBILE_MALI_GPU
#endif
#ifdef PADDLE_MOBILE_FPGA
#endif

}  // namespace operators
}  // namespace paddle_mobile

#endif
