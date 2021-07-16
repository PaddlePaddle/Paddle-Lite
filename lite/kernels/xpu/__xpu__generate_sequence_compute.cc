// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/__xpu__generate_sequence_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
std::vector<T> XPUGenerateSequenceCompute::generate_sequence() {
  auto& param = this->template Param<param_t>();
  const lite::Tensor* x = param.input;
  lite::Tensor* y = param.output;
  auto x_dims = x->dims();

  int axis = (param.axis < 0 ? param.axis + x_dims.size() : param.axis);
  T value = static_cast<T>(param.value);

  std::vector<T> y_data_host(y->numel(), 0);

  if (param.flatten || x_dims.size() == 1) {
    int64_t x_size = x->numel();
    y_data_host[0] = 0;
    for (int64_t i = 1; i < x_size; i++) {
      y_data_host[i] = y_data_host[i - 1] * value;
    }
  } else {
    int64_t pre = x_dims.count(0, axis);
    int64_t count = x_dims[axis];
    int64_t post = x_dims.count(axis + 1, x_dims.size());

    for (int64_t i = 0; i < pre; i++) {
      for (int64_t j = 0; j < post; j++) {
        int64_t step = i * count * post + j;
        T* dst = y_data_host.data() + step;
        dst[0] = 0;
        for (int64_t k = 1; k < count; k++) {
          dst[k * post] = (dst[(k - 1) * post] + 1) * value;
        }
      }
    }
  }
  return y_data_host;
}

void XPUGenerateSequenceCompute::Run() {
  auto& param = this->template Param<param_t>();
  lite::Tensor* y = param.output;
  int dtype = param.dtype;

  switch (dtype) {
    case 2: {
      std::vector<int32_t> y_data_host = generate_sequence<int32_t>();
      int32_t* y_data = y->template mutable_data<int32_t>(TARGET(kXPU));
      size_t mem_size = y_data_host.size() * sizeof(int32_t);
      TargetWrapperXPU::MemcpySync(
          y_data, y_data_host.data(), mem_size, IoDirection::HtoD);

      break;
    }
    case 3: {
      std::vector<int64_t> y_data_host = generate_sequence<int64_t>();
      int64_t* y_data = y->template mutable_data<int64_t>(TARGET(kXPU));
      size_t mem_size = y_data_host.size() * sizeof(int64_t);
      TargetWrapperXPU::MemcpySync(
          y_data, y_data_host.data(), mem_size, IoDirection::HtoD);

      break;
    }
    case 5: {
      std::vector<float> y_data_host = generate_sequence<float>();
      float* y_data = y->template mutable_data<float>(TARGET(kXPU));
      size_t mem_size = y_data_host.size() * sizeof(float);
      TargetWrapperXPU::MemcpySync(
          y_data, y_data_host.data(), mem_size, IoDirection::HtoD);
      break;
    }
    default: {
      LOG(FATAL) << "Attribute dtype in XPUGenerateSequence op "
                    "must be 2[int32] or 3[int64] or 5[fp32] for xpu: "
                 << dtype;
      break;
    }
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__generate_sequence,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUGenerateSequenceCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
