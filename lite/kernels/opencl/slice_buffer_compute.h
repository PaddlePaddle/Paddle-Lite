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
#pragma once

#include <string>
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

template <typename T, PrecisionType PType>
class SliceCompute
    : public KernelLite<TARGET(kOpenCL), PType, DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::SliceParam;

  void PrepareForRun() override;

  void ReInitWhenNeeded() override;

  void Run() override;

  ~SliceCompute() {
    TargetWrapperCL::Free(src_step_buf_);
    TargetWrapperCL::Free(dst_step_buf_);
    TargetWrapperCL::Free(real_starts_buf_);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event = this->event_;  // `event_` defined in `kernel.h`, valid after
                                  // kernel::Run
  }
#endif

 protected:
  param_t* slice_param_{nullptr};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  std::string kernel_func_name_{"slice"};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  cl::Kernel kernel_;
  cl::NDRange gws_;
  cl::Buffer* src_step_buf_{nullptr};
  cl::Buffer* dst_step_buf_{nullptr};
  cl::Buffer* real_starts_buf_{nullptr};
  int32_t out_num_{0};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
