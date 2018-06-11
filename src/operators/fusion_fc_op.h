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

#ifdef FUSION_FC_OP

#pragma once

#include <string>
#include <vector>

#include "framework/operator.h"
#include "framework/program/program-optimize/fusion_op_register.h"
#include "operators/kernel/fushion_fc_kernel.h"

namespace paddle_mobile {
namespace operators {
using std::string;
using std::vector;
class FusionFcMatcher : public framework::FusionOpMatcher {
 public:
  FusionFcMatcher() {
    node_ = framework::Node(G_OP_TYPE_MUL);
    node_ > std::make_shared<framework::Node>(G_OP_TYPE_ELEMENTWISE_ADD);
  }

  void FolderNodes(
      framework::Node *node,
      std::vector<std::shared_ptr<framework::Node>> *removed_nodes) {
    vector<std::shared_ptr<framework::OpDesc>> origin_descs =
        node->OpDescs(node_.Depth());
    node->Folder(node_.Depth(), Type(),
                 {{G_OP_TYPE_ELEMENTWISE_ADD, {"Y", "Z"}}}, removed_nodes);
  }

  std::string Type() { return G_OP_TYPE_FC; }
};

template <typename DeviceType, typename T>
class FushionFcOp : public framework::OperatorWithKernel<DeviceType> {
 public:
  FushionFcOp(const string &type, const VariableNameMap &inputs,
              const VariableNameMap &outputs,
              const framework::AttributeMap &attrs,
              std::shared_ptr<framework::Scope> scope)
      : framework::OperatorWithKernel<DeviceType>(type, inputs, outputs, attrs,
                                                  scope),
        param_(inputs, outputs, attrs, *scope) {}

  void RunImpl() const {
    operators::FushionFcKernel<DeviceType, T> kernel;
    kernel.Compute(param_);
  }

  using framework::OperatorWithKernel<DeviceType>::OperatorWithKernel;
  void InferShape() const override;

 protected:
  FushionFcParam param_;
};

static framework::FusionOpRegistrar fc_registrar(new FusionFcMatcher());

}  // namespace operators
}  // namespace paddle_mobile

#endif
