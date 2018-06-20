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
#define FUSION_CONVADD_OP
#ifdef FUSION_CONVADD_OP

#pragma once

#include <string>
#include <vector>
#include "framework/operator.h"
#include "framework/program/program-optimize/fusion_op_register.h"
#include "op_param.h"
#include "operators/kernel/conv_add_kernel.h"

namespace paddle_mobile {
namespace operators {
using std::string;
using std::vector;
class FusionConvAddMatcher : public framework::FusionOpMatcher {
 public:
  FusionConvAddMatcher() {
    node_ = framework::Node(G_OP_TYPE_CONV);
    node_ > std::make_shared<framework::Node>(G_OP_TYPE_ELEMENTWISE_ADD);
  }

  void FolderNodes(
      framework::Node *node,
      std::vector<std::shared_ptr<framework::Node>> *removed_nodes) {
    vector<std::shared_ptr<framework::OpDesc>> origin_descs =
        node->OpDescs(node_.Depth());
    node->Folder(node_.Depth(), Type(),
                 {{G_OP_TYPE_ELEMENTWISE_ADD, {"Y", "Y"}}}, removed_nodes);
  }

  std::string Type() { return G_OP_TYPE_CONV_ADD; }
};

template <typename DeviceType, typename T>
class FushionConvAddOp : public framework::OperatorWithKernel<
                             DeviceType, FushionConvAddParam,
                             operators::ConvAddKernel<DeviceType, T>> {
 public:
  FushionConvAddOp(const string &type, const VariableNameMap &inputs,
                   const VariableNameMap &outputs,
                   const framework::AttributeMap &attrs,
                   std::shared_ptr<framework::Scope> scope)
      : framework::OperatorWithKernel<DeviceType, FushionConvAddParam,
                                      operators::ConvAddKernel<DeviceType, T>>(
            type, inputs, outputs, attrs, scope) {}

  using framework::OperatorWithKernel<
      DeviceType, FushionConvAddParam,
      operators::ConvAddKernel<DeviceType, T>>::OperatorWithKernel;
  void InferShape() const override;

 protected:
};

inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                          int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  return output_size;
}

#ifdef PADDLE_MOBILE_CPU
static framework::FusionOpRegistrar convadd_registrar(
    new FusionConvAddMatcher());
#endif
#ifdef PADDLE_MOBILE_MALI_GPU
static framework::FusionOpRegistrar convadd_registrar(
    new FusionConvAddMatcher());
#endif
#ifdef PADDLE_MOBILE_FPGA
#endif

}  // namespace operators
}  // namespace paddle_mobile

#endif
