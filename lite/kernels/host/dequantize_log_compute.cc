// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/host/dequantize_log_compute.h"
#include <set>
#include <string>
#include <vector>
#include "lite/backends/host/math/reduce.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void DequantizeLogCompute<T>::Run() {
  std::cout << "DequantizeLogCompute::Run" << std::endl;
  auto& param = Param<operators::QuantizeLogParam>();
  auto x = param.X;
  auto dict = param.Dict;
  auto output = param.Out;
  const float* dict_data = dict->template data<float>();
  const T* input_data = x->template data<T>();
  float* output_data = output->template mutable_data<float>();
  int ind = x->numel();
  for (size_t i = 0; i < (unsigned)ind; i++) {
    if (input_data[i] < 0) {
      output_data[i] = -dict_data[input_data[i] + 128];
    } else {
      output_data[i] = dict_data[input_data[i]];
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(dequantize_log,
                     kHost,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::host::DequantizeLogCompute<int8_t>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .BindInput("Dict",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindPaddleOpVersion("dequantize_log", 1)
    .Finalize();
