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

#include "lite/kernels/xpu/sequence_mask_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T>
void SequenceMaskCompute<T>::Run() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();

  int max_len = param.maxlen;
  auto* max_len_tensor = param.MaxLenTensor;
  if (max_len_tensor != nullptr) {
    max_len = max_len_tensor->template data<int>()[0];
    CHECK_GT(max_len, 0)
        << "Input(MaxLenTensor)'s value should be greater than "
           "0. But received maxlen: "
        << max_len;
  }

  auto* x = param.X;
  auto* x_cpu_ptr = x->template data<T>();
  int x_size = static_cast<int>(x->numel());
  if (max_len < 0) {
    max_len =
        static_cast<int>(*std::max_element(x_cpu_ptr, x_cpu_ptr + x_size));
  }
  XPUScratchPadGuard x_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(x_size * sizeof(T));
  auto* x_xpu_ptr = reinterpret_cast<T*>(x_xpu_guard_->addr_);
  XPU_CALL(xpu_memcpy(x_xpu_ptr,
                      x_cpu_ptr,
                      x_size * sizeof(T),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  auto* y = param.Y;
  auto y_shape = x->dims().Vectorize();
  y_shape.push_back(static_cast<int64_t>(max_len));
  y->Resize(y_shape);
  y->set_lod(x->lod());

  int out_type = param.out_dtype;
  switch (lite::core::FluidType(out_type)) {
    case lite::core::FluidType::FP32: {
      int ret = xdnn::sequence_mask<int64_t, float>(
          ctx.GetRawContext(),
          x_xpu_ptr,
          y->template mutable_data<float>(TARGET(kXPU)),
          x_size,
          max_len);
      CHECK_EQ(ret, 0) << "call xdnn::sequence_mask failed!";
      break;
    }
    case lite::core::FluidType::INT32: {
      LOG(FATAL) << "XPU unsupported out data type: " << out_type;
      break;
    }
    case lite::core::FluidType::INT64: {
      LOG(FATAL) << "XPU unsupported out data type: " << out_type;
      break;
    }
    default:
      LOG(FATAL) << "unsupported out data type: " << out_type;
      break;
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_mask,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SequenceMaskCompute<int64_t>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("MaxLenTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
