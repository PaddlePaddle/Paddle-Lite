/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/loongarch/math/softmax.h"
#include "lite/backends/loongarch/math/softmax_impl.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

template class SoftmaxFunctor<lite::TargetType::kLoongArch, float, true>;
// note: these implemetaions have not been called yet
// template class SoftmaxFunctor<lite::TargetType::kLoongArch, float, false>;
// template class SoftmaxFunctor<lite::TargetType::kLoongArch, double, true>;
// template class SoftmaxFunctor<lite::TargetType::kLoongArch, double, false>;
// template class SoftmaxGradFunctor<lite::TargetType::kLoongArch, float>;
// template class SoftmaxGradFunctor<lite::TargetType::kLoongArch, double>;

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
