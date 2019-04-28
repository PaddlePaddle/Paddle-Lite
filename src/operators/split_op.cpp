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

#ifdef SPLIT_OP
#include "operators/split_op.h"
#include <vector>

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
void SplitOp<DeviceType, T>::InferShape() const {
  PADDLE_MOBILE_ENFORCE(this->param_.InputX() != nullptr,
                        "Input(X) of SplitOp should not be null.");
  //  std::string str;
  //  str.size()
  const auto &outs = this->param_.Outs();
  PADDLE_MOBILE_ENFORCE(outs.size() >= 1UL,
                        "Outputs(Out) of SplitOp should not be empty.");

  auto in_dims = this->param_.InputX()->dims();
  size_t axis = static_cast<size_t>(this->param_.Axis());
  size_t num = static_cast<size_t>(this->param_.Num());

  const auto &sections = this->param_.Sections();

  const size_t outs_number = outs.size();
  std::vector<framework::DDim> outs_dims;
  outs_dims.reserve(outs_number);

  if (num > 0) {
    int64_t in_axis_dim = in_dims[axis];
    PADDLE_MOBILE_ENFORCE(in_axis_dim % num == 0,
                          "tensor split does not result"
                          " in an equal division");
    size_t out_axis_dim = in_axis_dim / num;
    for (size_t i = 0; i < outs_number; ++i) {
      auto dim = in_dims;
      dim[axis] = out_axis_dim;
      outs_dims.push_back(dim);
    }
  } else if (sections.size() > 0) {
    PADDLE_MOBILE_ENFORCE(sections.size() == outs_number,
                          "tensor split sections size"
                          "should be equal to output size.");
    for (size_t i = 0; i < outs_number; ++i) {
      auto dim = in_dims;
      dim[axis] = sections[i];
      outs_dims.push_back(dim);
    }
  }

  PADDLE_MOBILE_ENFORCE(outs_dims.size() == outs.size(),
                        "length==dims.size()  must be true!");
  for (int j = 0; j < outs_dims.size(); ++j) {
    outs[j]->Resize(outs_dims[j]);
  }

  //  todo lod impl
  //  if (axis != 0) {
  //    // Only pass LoD when not spliting along the first dim.
  //    for (size_t i = 0; i < outs_number; ++i) {
  //      ctx->ShareLoD("X", "Out", 0, i);
  //    }
  //  }
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(split, ops::SplitOp);
#endif
#if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
REGISTER_OPERATOR_FPGA(split, ops::SplitOp);
#endif

#endif  // SPLIT_OP
