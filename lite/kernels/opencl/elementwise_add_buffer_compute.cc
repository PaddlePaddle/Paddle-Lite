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

#include "lite/kernels/opencl/elementwise_add_buffer_compute.h"
#include <memory>
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/op_registry.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

void ElementwiseAddCompute::PrepareForRun() {
  ele_param_ = param_.get_mutable<param_t>();
}

void ElementwiseAddCompute::ReInitWhenNeeded() {
  auto x_dims = ele_param_->X->dims();
  if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
      first_epoch_for_reinit_) {
    last_x_dims_ = x_dims;
    first_epoch_for_reinit_ = false;
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "buffer/elementwise_add_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    UpdateParams();
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());

    global_work_size_ = cl::NDRange{channels_, batch_};

#ifdef LITE_WITH_LOG
    VLOG(4) << TargetToStr(ele_param_->X->target());
    VLOG(4) << TargetToStr(ele_param_->Y->target());
    VLOG(4) << TargetToStr(ele_param_->Out->target());
#endif
  }
}

void ElementwiseAddCompute::Run() {
  auto* x_buf = GET_BUFFER_GPU(ele_param_->X);
  auto* y_buf = GET_BUFFER_GPU(ele_param_->Y);
  auto* out_buf = MUTABLE_BUFFER_GPU(ele_param_->Out);

  int arg_idx = 0;
  cl_int status = kernel_.setArg(arg_idx, *x_buf);
  CL_CHECK_FATAL(status);
  status = kernel_.setArg(++arg_idx, *y_buf);
  CL_CHECK_FATAL(status);
  status = kernel_.setArg(++arg_idx, *out_buf);
  CL_CHECK_FATAL(status);
  status = kernel_.setArg(++arg_idx, (const int)batch_);
  CL_CHECK_FATAL(status);
  status = kernel_.setArg(++arg_idx, (const int)channels_);
  CL_CHECK_FATAL(status);
  status = kernel_.setArg(++arg_idx, (const int)num_);
  CL_CHECK_FATAL(status);

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  status = EnqueueNDRangeKernel(context,
                                kernel_,
                                cl::NullRange,
                                global_work_size_,
                                cl::NullRange,
                                nullptr,
                                event_);
  CL_CHECK_FATAL(status);

#ifdef LITE_WITH_PROFILE
  event_.wait();
  auto queue_start_nanos =
      event_.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
  auto submit_start_nanos =
      event_.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
  auto run_start_nanos = event_.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  auto run_stop_nanos = event_.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  double time_ms = (submit_start_nanos - queue_start_nanos) / 1000000.0;
  VLOG(4) << "GetQueuedToSubmitTime: " << time_ms << std::endl;

  time_ms = (run_start_nanos - submit_start_nanos) / 1000000.0;
  VLOG(4) << "GetSubmitToStartTime: " << time_ms << std::endl;

  time_ms = (run_stop_nanos - run_start_nanos) / 1000000.0;
  VLOG(4) << "GetStartToEndTime: " << time_ms << std::endl;
#endif
}

void ElementwiseAddCompute::UpdateParams() {
  auto axis = ele_param_->axis;
  const auto& x_dims = ele_param_->X->dims();
  const auto& y_dims = ele_param_->Y->dims();
  const auto& out_dims = ele_param_->Out->dims();
  if (axis < 0) {
    axis = static_cast<int>(x_dims.size() - y_dims.size());
  }
  if (y_dims[0] == 1 && y_dims[1] == 1 && y_dims[2] == 1 &&
      y_dims[3] == x_dims[3]) {
    axis = 3;
  }
  for (int i = 0; i < axis; ++i) {
    batch_ *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    channels_ *= y_dims[i];
  }
  for (int i = static_cast<int>(y_dims.size() + axis); i < x_dims.size(); ++i) {
    num_ *= x_dims[i];
  }
#ifdef LITE_WITH_LOG
  VLOG(4) << "axis: " << axis;
  VLOG(4) << "batch: " << batch_;
  VLOG(4) << "channels: " << channels_;
  VLOG(4) << "num: " << num_;
#endif
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;

REGISTER_LITE_KERNEL(
    elementwise_add, kOpenCL, kFP16, kNCHW, ocl::ElementwiseAddCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
