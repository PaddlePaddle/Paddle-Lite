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

#ifdef CRF_OP

#include <vector>

#include "common/enforce.h"
#include "operators/crf_op.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void CrfOp<Dtype, T>::InferShape() const {
  PADDLE_MOBILE_ENFORCE(this->param_.InputEmission(),
                        "Input(Emission) should be not null.");
  PADDLE_MOBILE_ENFORCE(this->param_.InputTransition(),
                        "Input(Transition) should be not null.");
  PADDLE_MOBILE_ENFORCE(this->param_.outputVBP(),
                        "Input(ViterbiPath) should be not null.");

  auto emission_dims = this->param_.InputEmission()->dims();
  PADDLE_MOBILE_ENFORCE(emission_dims.size() == 2U,
                        "The Input(Emission) should be a 2-D tensor.");
  PADDLE_MOBILE_ENFORCE(emission_dims[0],
                        "An empty mini-batch is not allowed.");

  this->param_.outputVBP()->Resize(
      {this->param_.InputEmission()->dims()[0], 1});
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(crf_decoding, ops::CrfOp);
#endif

#ifdef PADDLE_MOBILE_FPGA
#endif

#endif
