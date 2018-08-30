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

#include "operators/kernel/lookup_kernel.h"
#include "operators/kernel/central-arm-func/lookup_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool LookupKernel<CPU, float>::Init(LookupParam<CPU> *param) {
  return true;
}

template <>
void LookupKernel<CPU, float>::Compute(const LookupParam<CPU> &param) const {
  PADDLE_MOBILE_ENFORCE(param.InputW() != nullptr,
                        "Input(W) of LookupTableOp should not be null.");
  auto *ids_t = param.InputIds();

  PADDLE_MOBILE_ENFORCE(ids_t != nullptr,
                        "Input(Ids) of LookupTableOp should not be null.");
  PADDLE_MOBILE_ENFORCE(param.Out() != nullptr,
                        "Output(Out) of LookupTableOp should not be null.");
  //    param_.InputW()->

  auto table_dims = param.InputW()->dims();
  auto ids_dims = ids_t->dims();

  int ids_rank = ids_dims.size();

  PADDLE_MOBILE_ENFORCE(table_dims.size() == 2,
                        "table_dims.size()==2 check failed");

  PADDLE_MOBILE_ENFORCE(ids_dims[ids_rank - 1] == 1,
                        "The last dimension of the 'Ids' tensor must be 1.");

  auto output_dims =
      framework::vectorize(framework::slice_ddim(ids_dims, 0, ids_rank - 1));
  output_dims.push_back(table_dims[1]);

  param.Out()->Resize(framework::make_ddim(output_dims));
  LookupCompute<float>(param);
  param.Out()->set_lod(param.InputIds()->lod());
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
