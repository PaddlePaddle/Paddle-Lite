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

#include "operators/kernel/feed_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool FeedKernel<GPU_MALI, float>::Init(FeedParam<GPU_MALI> *param) {
  return true;
}

template <>
void FeedKernel<GPU_MALI, float>::Compute(const FeedParam<GPU_MALI> &param) {
  param.Out()->ShareDataWith(*(param.InputX()));
  param.Out()->set_lod(param.InputX()->lod());
}

template class FeedKernel<GPU_MALI, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
