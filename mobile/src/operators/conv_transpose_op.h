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

#ifdef CONV_TRANSPOSE_OP

#pragma once

#include <string>
#include <vector>

#include "framework/operator.h"
#include "operators/kernel/conv_transpose_kernel.h"

namespace paddle_mobile {
namespace operators {
template <typename DeviceType, typename T>
class ConvOpTranspose : public framework::OperatorWithKernel<
                            DeviceType, ConvTransposeParam<DeviceType>,
                            operators::ConvTransposeKernel<DeviceType, T>> {
 public:
  ConvOpTranspose(const std::string &type, const VariableNameMap &inputs,
                  const VariableNameMap &outputs,
                  const framework::AttributeMap &attrs, framework::Scope *scope)
      : framework::OperatorWithKernel<
            DeviceType, ConvTransposeParam<DeviceType>,
            operators::ConvTransposeKernel<DeviceType, T>>(
            type, inputs, outputs, attrs, scope) {}

  void InferShape() const {
    auto input = this->param_.Input();
    auto in_dims = input->dims();

    auto filter = this->param_.Filter();
    auto filter_dims = filter->dims();

    std::vector<int> strides = this->param_.Strides();
    std::vector<int> paddings = this->param_.Paddings();
    std::vector<int> dilations = this->param_.Dilations();
    std::vector<int> output_size = this->param_.OutputSize();

    int groups = this->param_.Groups();

    PADDLE_MOBILE_ENFORCE(
        in_dims.size() == 4 || in_dims.size() == 5,
        "ConvTransposeOp intput should be 4-D or 5-D tensor.");
    PADDLE_MOBILE_ENFORCE(
        in_dims.size() == filter_dims.size(),
        "ConvTransposeOp input dimension and filter dimension "
        "should be the same.");
    PADDLE_MOBILE_ENFORCE(
        in_dims.size() - strides.size() == 2U,
        "ConvTransposeOp input dimension and strides dimension should "
        "be consistent.");
    PADDLE_MOBILE_ENFORCE(paddings.size() == strides.size(),
                          "ConvTransposeOp paddings dimension and strides "
                          "dimension should be the same.");
    PADDLE_MOBILE_ENFORCE(paddings.size() == dilations.size(),
                          "ConvTransposeOp paddings dimension and dilations "
                          "dimension should be the same.");
    PADDLE_MOBILE_ENFORCE(
        in_dims[1] == filter_dims[0],
        "In ConvTransposeOp, The number of input channels should "
        "be equal to the number of filter's channels.");

    std::vector<int64_t> output_shape({in_dims[0], filter_dims[1] * groups});
    if (output_size.size() == 2) {
      output_shape.push_back(output_size[0]);
      output_shape.push_back(output_size[1]);
    } else {
      for (size_t i = 0; i < strides.size(); ++i) {
        auto filter_extent = dilations[i] * (filter_dims[i + 2] - 1) + 1;
        output_shape.push_back((in_dims[i + 2] - 1) * strides[i] -
                               2 * paddings[i] + filter_extent);
      }
    }

    this->param_.Output()->Resize(framework::make_ddim(output_shape));
  }

 private:
};

}  // namespace operators
}  // namespace paddle_mobile

#endif
