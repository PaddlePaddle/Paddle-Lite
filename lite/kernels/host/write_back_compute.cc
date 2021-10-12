// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/write_back_compute.h"
#include "lite/core/target_wrapper.h"
#ifdef LITE_WITH_XPU
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void WriteBackCompute::RunImplement(const lite::Tensor* x,
                                    lite::Tensor* y,
                                    bool is_tensor_array_copy) {
  auto x_target = x->target();
  auto y_target = y->target();
  auto is_host = [](TargetType x) -> bool {
    return x == TARGET(kHost) || x == TARGET(kX86) || x == TARGET(kARM);
  };

  if (is_host(x_target) && is_host(y_target)) {
    if (is_tensor_array_copy)
      y->ShareDataWith(*x);
    else
      y->CopyDataFrom(*x);
  } else if (x_target == TARGET(kXPU) || y_target == TARGET(kXPU)) {
#ifdef LITE_WITH_XPU
    y->set_precision(x->precision());
    y->Resize(x->dims());
    y->set_lod(x->lod());
    if (is_host(x_target)) {
      auto mem_size = x->memory_size();
      VLOG(4) << "host to xpu, copy size " << mem_size;
      auto* data = y->mutable_data(TARGET(kXPU), mem_size);
      if (mem_size > 0) {
        TargetWrapperXPU::MemcpySync(
            data, x->raw_data(), mem_size, IoDirection::HtoD);
      }
    } else if (is_host(y_target)) {
      auto mem_size = x->memory_size();
      VLOG(4) << "xpu to host, copy size " << mem_size;
      auto* data = y->mutable_data(TARGET(kHost), mem_size);
      if (mem_size > 0) {
        TargetWrapperXPU::MemcpySync(
            data, x->raw_data(), mem_size, IoDirection::DtoH);
      }
    } else {
      auto mem_size = x->memory_size();
      VLOG(4) << "xpu to xpu, copy size " << mem_size;
      if (mem_size > 0) {
        int r = xdnn::copy<int8_t>(
            TargetWrapperXPU::GetRawContext(),
            reinterpret_cast<const int8_t*>(x->raw_data()),
            reinterpret_cast<int8_t*>(y->mutable_data(TARGET(kXPU), mem_size)),
            mem_size);
        CHECK_EQ(r, 0);
      }
    }
#endif
  } else {
    LOG(ERROR) << "Not support copy x_target("
               << lite_api::TargetToStr(x_target) << ") to y_target("
               << lite_api::TargetToStr(y_target) << ").";
  }
}

void WriteBackCompute::Run() {
  auto& param = this->template Param<operators::WriteBackParam>();
  if (!param.tensor_array_copy) {
    auto* x = param.x;
    auto* y = param.y;
    RunImplement(x, y, false);
  } else {
    auto size = param.array_y->size();
    for (size_t i = size; i > 0; i--) {
      auto& y = param.array_y->at(size - 1);
      auto& x = param.array_x->at(size - 1);
      if (x.raw_data()) continue;
      RunImplement(&y, &x, true);
    }
    param.array_y->resize(param.array_x->size());
    for (size_t i = 0; i < param.array_x->size(); i++) {
      auto& x = param.array_x->at(i);
      auto& y = param.array_y->at(i);
      if (y.raw_data()) continue;
      RunImplement(&x, &y, true);
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(write_back,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::WriteBackCompute,
                     write_back)
    .BindInput("Src_LoDTensor",
               {LiteType::GetTensorTy(TARGET(kAny),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Dst_LoDTensor",
               {LiteType::GetTensorTy(TARGET(kAny),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Src_LoDTensorArray",
               {LiteType::GetTensorListTy(TARGET(kAny),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindInput("Dst_LoDTensorArray",
               {LiteType::GetTensorListTy(TARGET(kAny),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindInput("Dep_LoDTensor",
               {LiteType::GetTensorTy(TARGET(kAny),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Dep_LoDTensorArray",
               {LiteType::GetTensorListTy(TARGET(kAny),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .Finalize();
