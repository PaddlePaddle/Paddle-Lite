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

#ifdef LOOKUP_OP

#include <vector>

#include "common/enforce.h"
#include "operators/lookup_op.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void LookupOp<Dtype, T>::InferShape() const {
  PADDLE_MOBILE_ENFORCE(this->param_.InputW() != nullptr,
                        "Input(W) of LookupTableOp should not be null.");
  auto *ids_t = this->param_.InputIds();

  PADDLE_MOBILE_ENFORCE(ids_t != nullptr,
                        "Input(Ids) of LookupTableOp should not be null.");
  PADDLE_MOBILE_ENFORCE(this->param_.Out() != nullptr,
                        "Output(Out) of LookupTableOp should not be null.");
  //    this->param__.InputW()->

  auto table_dims = this->param_.InputW()->dims();
  auto ids_dims = ids_t->dims();

  int ids_rank = ids_dims.size();

  PADDLE_MOBILE_ENFORCE(table_dims.size() == 2,
                        "table_dims.size()==2 check failed");

  PADDLE_MOBILE_ENFORCE(ids_dims[ids_rank - 1] == 1,
                        "The last dimension of the 'Ids' tensor must be 1.");

  auto output_dims =
      framework::vectorize(framework::slice_ddim(ids_dims, 0, ids_rank - 1));
  output_dims.push_back(table_dims[1]);

  this->param_.Out()->Resize(framework::make_ddim(output_dims));
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(lookup_table, ops::LookupOp);
#endif

#ifdef PADDLE_MOBILE_FPGA
#endif

#endif
