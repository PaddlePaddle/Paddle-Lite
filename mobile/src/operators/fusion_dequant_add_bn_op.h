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

#ifdef FUSION_DEQUANT_ADD_BN_OP

#pragma once

#include <string>
#include <vector>
#include "framework/operator.h"
#include "framework/program/program-optimize/fusion_op_register.h"
#include "operators/kernel/dequant_bn_kernel.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

class FusionDequantAddBNMatcher : public framework::FusionOpMatcher {
 public:
  FusionDequantAddBNMatcher() {
    node_ = framework::Node(G_OP_TYPE_DEQUANTIZE);
    node_ > std::make_shared<framework::Node>(G_OP_TYPE_ELEMENTWISE_ADD) >
        std::make_shared<framework::Node>(G_OP_TYPE_BATCHNORM);
  }

  void FolderNodes(
      framework::Node *node,
      std::vector<std::shared_ptr<framework::Node>> *removed_nodes) {
    node->Folder(node_.Depth(), Type(),
                 {{G_OP_TYPE_ELEMENTWISE_ADD, {{"Y", "Y"}}},
                  {G_OP_TYPE_BATCHNORM,
                   {{"Scale", "BNScale"},
                    {"Mean", "BNMean"},
                    {"Bias", "BNBias"},
                    {"Variance", "BNVariance"},
                    {"Y", "Out"}}}},
                 removed_nodes);
  }

  std::string Type() { return G_OP_TYPE_FUSION_DEQUANT_ADD_BN; }
};

template <typename DeviceType, typename T>
class FusionDequantAddBNOp
    : public framework::OperatorWithKernel<
          DeviceType, FusionDequantAddBNParam<DeviceType>,
          operators::FusionDequantAddBNKernel<DeviceType, T>> {
 public:
  FusionDequantAddBNOp(const std::string &type, const VariableNameMap &inputs,
                       const VariableNameMap &outputs,
                       const framework::AttributeMap &attrs,
                       framework::Scope *scope)
      : framework::OperatorWithKernel<
            DeviceType, FusionDequantAddBNParam<DeviceType>,
            operators::FusionDequantAddBNKernel<DeviceType, T>>(
            type, inputs, outputs, attrs, scope) {}
  // inference output shape
  void InferShape() const override;
};

}  // namespace operators
}  // namespace paddle_mobile

#endif
