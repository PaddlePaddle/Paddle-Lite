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
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {
using std::string;
template <typename DeviceType, typename T>
class FeedOp : public framework::OperatorBase<DeviceType> {
 public:
  FeedOp(const string &type, const VariableNameMap &inputs,
         const VariableNameMap &outputs, const framework::AttributeMap attrs,
         std::shared_ptr<framework::Scope> scope)
      : framework::OperatorBase<DeviceType>(type, inputs, outputs, attrs,
                                            scope),
        param_(inputs, outputs, attrs, scope.get()) {}

  void InferShape() const {
    auto out_dims = param_.Out()->dims();
    out_dims[0] = param_.BatchSize();
    param_.Out()->Resize(out_dims);
  }

#ifdef PADDLE_MOBILE_FPGA

  void Init() {
    Tensor *output = param_.Out();
    output->mutable_data<half>();
  }

  void RunImpl() const {
    const Tensor *input = param_.InputX();
    auto input_ptr = input->data<float>();
    Tensor *output = param_.Out();
    auto output_ptr = output->mutable_data<half>();
    auto out_address = output->fpga_args().scale_pointer();
    fpga::BypassArgs args;
    args.convert_type = fpga::DATA_FP32_TO_FP16;
    args.layout_type = fpga::LAYOUT_CHW_TO_HWC;
    args.image.address = (void *)input_ptr;
    args.image.channels = input->dims()[1];
    args.image.height = input->dims()[2];
    args.image.width = input->dims()[3];
    args.image.pad_height = 0;
    args.image.pad_width = 0;
    args.output.address = output_ptr;
    args.output.scale_address = out_address;
    fpga::PerformBypass(args);
  }

#else
  void Init() {}
  void RunImpl() const { param_.Out()->ShareDataWith(*param_.InputX()); }
#endif

 protected:
  FeedParam<DeviceType> param_;
};

}  // namespace operators
}  // namespace paddle_mobile

#ifdef PADDLE_MOBILE_CPU
USE_OP_CPU(feed);
#endif
#ifdef PADDLE_MOBILE_MALI_GPU
USE_OP_MALI_GPU(feed);
#endif
#ifdef PADDLE_MOBILE_FPGA
USE_OP_FPGA(feed);
#endif
