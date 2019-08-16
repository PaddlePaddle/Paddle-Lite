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

#ifdef FUSION_CONVBNRELU_OP

#pragma once

#include <string>
#include <vector>
#include "framework/operator.h"
#include "framework/program/program-optimize/fusion_op_register.h"
#include "operators/kernel/conv_bn_relu_kernel.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {
using std::string;
using std::vector;
class FusionConvBNReluMatcher : public framework::FusionOpMatcher {
 public:
  FusionConvBNReluMatcher() {
    node_ = framework::Node(G_OP_TYPE_CONV);
    node_ > std::make_shared<framework::Node>(G_OP_TYPE_BATCHNORM) >
        std::make_shared<framework::Node>(G_OP_TYPE_RELU);
  }

  void FolderNodes(
      framework::Node *node,
      std::vector<std::shared_ptr<framework::Node>> *removed_nodes) {
    node->Folder(node_.Depth(), Type(),
                 {{G_OP_TYPE_BATCHNORM,
                   {{"Scale", "Scale"},
                    {"Mean", "Mean"},
                    {"Bias", "Bias"},
                    {"Variance", "Variance"}}}},
                 removed_nodes);
  }

  std::string Type() { return G_OP_TYPE_FUSION_CONV_BN_RELU; }
};

template <typename DeviceType, typename T>
class FusionConvBNReluOp : public framework::OperatorWithKernel<
                               DeviceType, FusionConvBNReluParam<DeviceType>,
                               operators::ConvBNReluKernel<DeviceType, T>> {
 public:
  FusionConvBNReluOp(const string &type, const VariableNameMap &inputs,
                     const VariableNameMap &outputs,
                     const framework::AttributeMap &attrs,
                     framework::Scope *scope)
      : framework::OperatorWithKernel<
            DeviceType, FusionConvBNReluParam<DeviceType>,
            operators::ConvBNReluKernel<DeviceType, T>>(type, inputs, outputs,
                                                        attrs, scope) {}
  void InferShape() const override;

 protected:
};

}  // namespace operators
}  // namespace paddle_mobile

#endif
