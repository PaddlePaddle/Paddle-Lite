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

#ifdef INCREMENT_OP

#include "operators/increment_op.h"
#include "framework/op_proto_maker.h"
#include "framework/op_registry.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void IncrementOp<Dtype, T>::InferShape() const {
  auto input = this->param_.InputX();
  auto out = this->param_.Out();
  PADDLE_MOBILE_ENFORCE(input->numel() == 1, "input's numel should be 1");
  out->Resize(input->dims());
  if (std::is_same<DeviceType<kCPU>, Dtype>::value) {
    out->set_lod(input->lod());
  }
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(increment, ops::IncrementOp);
#endif

#ifdef PADDLE_MOBILE_FPGA
#endif

#ifdef PADDLE_MOBILE_CL
#endif

#endif
