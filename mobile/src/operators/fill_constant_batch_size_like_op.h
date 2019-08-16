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

#ifdef FILL_CONSTANT_BATCH_SIZE_LIKE_OP

#pragma once

#include <algorithm>
#include <string>
#include "framework/data_type.h"
#include "framework/operator.h"
#include "framework/selected_rows.h"
#include "operators/math/math_function.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
class FillConstantBatchSizeLikeOp : public framework::OperatorBase<DeviceType> {
 public:
  FillConstantBatchSizeLikeOp(const std::string &type,
                              const VariableNameMap &inputs,
                              const VariableNameMap &outputs,
                              const framework::AttributeMap attrs,
                              framework::Scope *scope)
      : framework::OperatorBase<DeviceType>(type, inputs, outputs, attrs,
                                            scope),
        param_(inputs, outputs, attrs, scope) {}
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
          "fill constant batch size like op's output only"
          "supports SelectedRows and LoDTensor");
    }
    auto shape = param_.Shape();
    std::vector<int64_t> shape_int64(shape.size(), 0);
    std::transform(shape.begin(), shape.end(), shape_int64.begin(),
                   [](int a) { return static_cast<int64_t>(a); });
    auto ddim = framework::make_ddim(shape_int64);
    ddim[param_.OutputDimIdx()] = param_.Input()->dims()[param_.InputDimIdx()];
    tensor->Resize(ddim);
    tensor->mutable_data(framework::ToTypeIndex(data_type));

    math::SetConstant(tensor, value);
  }

  void Init() {}

  void InferShape() const {
    PADDLE_MOBILE_ENFORCE(
        param_.Out() != nullptr,
        "Output (Out) of fill_constant_batch_size_like op should not be null.");

    auto shape = param_.Shape();

    std::vector<int64_t> shape_int64(shape.size(), 0);
    std::transform(shape.begin(), shape.end(), shape_int64.begin(),
                   [](int a) { return static_cast<int64_t>(a); });
    DLOG << shape_int64;
    auto ddim = framework::make_ddim(shape_int64);
    ddim[param_.OutputDimIdx()] = param_.Input()->dims()[param_.InputDimIdx()];
    param_.Out()->Resize(ddim);
  }

 protected:
  FillConstantBatchSizeLikeParam<DeviceType> param_;
};

}  // namespace operators
}  // namespace paddle_mobile

#endif
