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

#include "io/paddle_test_inference_api.h"
#include "io/paddle_mobile.h"

namespace paddle_mobile {

template <typename Device, typename T>
double PaddleTester<Device, T>::CaculatePredictTime(std::string *cl_path) {
  PaddleMobile<Device, T> paddle_mobile;
#ifdef PADDLE_MOBILE_CL
  if (cl_path) {
    paddle_mobile.SetCLPath(*cl_path);
  }

#endif
  return paddle_mobile.GetPredictTime();
}
template class PaddleTester<CPU, float>;
template class PaddleTester<FPGA, float>;

template class PaddleTester<GPU_CL, float>;

}  // namespace paddle_mobile
