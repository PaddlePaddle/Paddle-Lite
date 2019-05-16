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

#include <common/types.h>
#include <string>

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
struct DtypeTensorTrait {
  // This is the type we obtained in variable.
  typedef framework::LoDTensor gtype;
  // This type will be the parent class type
  // or the same type.
  typedef framework::Tensor rtype;
};

#ifdef PADDLE_MOBILE_CL
template <>
struct DtypeTensorTrait<GPU_CL> {
  // This is the type we obtained in variable.
  typedef framework::CLImage gtype;
  // This type will be the parent class type
  // or the same type.
  typedef framework::CLImage rtype;
};
#endif

}  // namespace framework
}  // namespace paddle_mobile
