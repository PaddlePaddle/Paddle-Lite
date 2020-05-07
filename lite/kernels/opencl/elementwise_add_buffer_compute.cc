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
  auto& context = ctx_->As<OpenCLContext>();
  context.cl_context()->AddKernel(kernel_func_name_,
                                  "buffer/elementwise_add_kernel.cl",
                                  build_options_,
                                  time_stamp_);
  ele_param_ = param_.get_mutable<param_t>();
  UpdateParams();
}

void ElementwiseAddCompute::Run() {
  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  auto* x_buf = ele_param_->X->template data<float, cl::Buffer>();
  auto* y_buf = ele_param_->Y->template data<float, cl::Buffer>();
  auto* out_buf = ele_param_->Out->template mutable_data<float, cl::Buffer>(
      TARGET(kOpenCL));
  STL::stringstream kernel_key;
  kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
  auto kernel = context.cl_context()->GetKernel(kernel_key.str());
#ifdef LITE_WITH_LOG
  VLOG(4) << TargetToStr(ele_param_->X->target());
  VLOG(4) << TargetToStr(ele_param_->Y->target());
  VLOG(4) << TargetToStr(ele_param_->Out->target());
#endif
  int arg_idx = 0;
  cl_int status = kernel.setArg(arg_idx, *x_buf);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *y_buf);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *out_buf);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, (const int)batch_);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, (const int)channels_);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, (const int)num_);
  CL_CHECK_FATAL(status);

  auto global_work_size = cl::NDRange{channels_, batch_};

  status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel, cl::NullRange, global_work_size, cl::NullRange, nullptr, nullptr);
  CL_CHECK_FATAL(status);
}

void ElementwiseAddCompute::UpdateParams() {
  auto axis = ele_param_->axis;
  const auto& x_dims = ele_param_->X->dims();
  const auto& y_dims = ele_param_->Y->dims();
  const auto& out_dims = ele_param_->Out->dims();
  if (axis < 0) {
    axis = static_cast<int>(x_dims.size() - y_dims.size());
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
    elementwise_add, kOpenCL, kFloat, kNCHW, ocl::ElementwiseAddCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
