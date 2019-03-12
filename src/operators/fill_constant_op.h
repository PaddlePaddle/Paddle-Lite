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

#ifdef FILL_CONSTANT_OP

#pragma once

#include <string>
#include "framework/data_type.h"
#include "framework/operator.h"
#include "framework/selected_rows.h"
#include "operators/math/math_function.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
class FillConstantOp : public framework::OperatorBase<DeviceType> {
 public:
  FillConstantOp(const std::string &type, const VariableNameMap &inputs,
                 const VariableNameMap &outputs,
                 const framework::AttributeMap attrs, framework::Scope *scope)
      : framework::OperatorBase<DeviceType>(type, inputs, outputs, attrs,
                                            scope),
        param_(inputs, outputs, attrs, scope.get()) {}
  void RunImpl() {
    auto data_type =
        static_cast<_PaddleMobile__Framework__Proto__VarType__Type>(
            param_.DataDtype());
    framework::Tensor *tensor = nullptr;
    auto value = param_.Value();
    auto *outvar = param_.OutVar();

    if (outvar->template IsType<framework::LoDTensor>()) {
      tensor = outvar->template GetMutable<framework::LoDTensor>();
    } else if (outvar->template IsType<framework::SelectedRows>()) {
      tensor = outvar->template GetMutable<framework::SelectedRows>()
                   ->mutable_value();
    } else {
      PADDLE_MOBILE_THROW_EXCEPTION(
          "fill constant op's output only"
          "supports SelectedRows and LoDTensor");
    }
    tensor->Resize(framework::make_ddim(param_.Shape()));
    tensor->mutable_data(framework::ToTypeIndex(data_type));

    math::SetConstant(tensor, value);
  }

  void Init() {}

  void InferShape() const {
    PADDLE_MOBILE_ENFORCE(
        param_.Out() != nullptr,
        "Output (Out) of fill_constant op should not be null.");
    framework::DDim ddim = framework::make_ddim(param_.Shape());
    param_.Out()->Resize(ddim);
  }

 protected:
  FillConstantParam<DeviceType> param_;
};

}  // namespace operators
}  // namespace paddle_mobile

#endif
