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

#ifdef LOD_RESET_OP

#include "operators/kernel/kernels.h"

namespace paddle_mobile {
namespace operators {

template <>
bool LodResetKernel<CPU, float>::Init(LodResetParam<CPU> *param) {
  return true;
}

template <>
void LodResetKernel<CPU, float>::Compute(const LodResetParam<CPU> &param) {
  const auto *input = param.input_x_;
  const auto *lod_t = param.input_y_;
  bool append = param.append;
  auto *output = param.output_;

  output->ShareDataWith(*input);

  std::vector<int> level0;
  if (lod_t) {
    if (lod_t->lod().size() > 0) {
      output->set_lod(lod_t->lod());
      return;  // early return, since lod already set
    } else {
      auto *lod = lod_t->data<int>();
      level0 = std::vector<int>(lod, lod + lod_t->numel());
    }
  } else {
    level0 = param.target_lod_;
  }

  // cast level0 to size_t
  std::vector<size_t> ulevel0(level0.size(), 0);
  std::transform(level0.begin(), level0.end(), ulevel0.begin(),
                 [](int a) { return static_cast<size_t>(a); });

  if (append) {
    auto *out_lod = output->mutable_lod();
    out_lod->push_back(ulevel0);
  } else {
    framework::LoD target_lod;
    target_lod.push_back(ulevel0);
    output->set_lod(target_lod);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // LOD_RESET_OP
