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

#ifdef FLATTEN_OP

#pragma once

#include <string>
#include <vector>

#include "framework/operator.h"
#include "operators/kernel/flatten_kernel.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

inline std::vector<int32_t> GetOutputShape(const int axis,
                                           const framework::DDim &in_dims) {
  int64_t outer = 1, inner = 1;
  for (int i = 0; i < in_dims.size(); ++i) {
    if (i < axis) {
      outer *= in_dims[i];
    } else {
      inner *= in_dims[i];
    }
  }
  std::vector<int32_t> out_shape(2);
  out_shape[0] = static_cast<int>(outer);
  out_shape[1] = static_cast<int>(inner);
  return out_shape;
}

template <typename DeviceType, typename T>
class FlattenOp : public framework::OperatorWithKernel<
                      DeviceType, FlattenParam<DeviceType>,
                      operators::FlattenKernel<DeviceType, T>> {
 public:
  FlattenOp(const std::string &type, const VariableNameMap &inputs,
            const VariableNameMap &outputs,
            const framework::AttributeMap &attrs, framework::Scope *scope)
      : framework::OperatorWithKernel<DeviceType, FlattenParam<DeviceType>,
                                      operators::FlattenKernel<DeviceType, T>>(
            type, inputs, outputs, attrs, scope) {}
  void InferShape() const override;
};

template <typename DeviceType, typename T>
class Flatten2Op : public FlattenOp<DeviceType, T> {
 public:
  Flatten2Op(const std::string &type, const VariableNameMap &inputs,
             const VariableNameMap &outputs,
             const framework::AttributeMap &attrs, framework::Scope *scope)
      : FlattenOp<DeviceType, T>(type, inputs, outputs, attrs, scope) {}
};

}  // namespace operators
}  // namespace paddle_mobile

#endif
