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

#include "lite/kernels/opencl/elementwise_mul_compute.h"
#include <memory>
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/op_registry.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

void ElementwiseMulFloatImageCompute::PrepareForRun() {
  ele_param_ = param_.get_mutable<param_t>();
  auto* y = ele_param_->Y;
  auto y_dims = y->dims();
  if (y_dims == ele_param_->X->dims()) {
    kernel_func_name_ = "elementwise_mul";
  } else if (y_dims.size() == 1 || y_dims.size() == 4) {
    kernel_func_name_ = "channel_mul";
  } else if (y_dims.size() == 2) {
    kernel_func_name_ = "channel_mul_d2";
  } else {
    LOG(FATAL) << "ElementwiseMul not supported y_dims.size():"
               << y_dims.size();
  }
  VLOG(4) << "kernel_func_name_:" << kernel_func_name_;
  VLOG(4) << "y_dims:" << y_dims;
  VLOG(4) << "y_dims.size():" << y_dims.size();

  auto& context = ctx_->As<OpenCLContext>();
  context.cl_context()->AddKernel(
      kernel_func_name_, "image/elementwise_mul_kernel.cl", build_options_);

  //  UpdateParams();
}

void ElementwiseMulFloatImageCompute::Run() {
  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);

  auto* x = ele_param_->X;
  auto* y = ele_param_->Y;
  auto* out = ele_param_->Out;

  VLOG(4) << "x->target():" << TargetToStr(x->target());
  VLOG(4) << "y->target():" << TargetToStr(y->target());
  VLOG(4) << "out->target():" << TargetToStr(out->target());
  VLOG(4) << "x->dims():" << x->dims();
  VLOG(4) << "y->dims():" << y->dims();
  VLOG(4) << "out->dims():" << out->dims();

  paddle::lite::CLImageConverterDefault default_convertor;
  auto x_img_shape = default_convertor.InitImageDimInfoWith(x->dims());  // w, h
  auto x_img_width = x_img_shape[0];
  auto x_img_height = x_img_shape[1];
  auto out_img_shape =
      default_convertor.InitImageDimInfoWith(out->dims());  // w, h
  auto y_img_shape = default_convertor.InitImageDimInfoWith(y->dims());

  auto* x_img = x->data<float, cl::Image2D>();
  auto* y_img = y->data<float, cl::Image2D>();
  auto* out_img =
      out->mutable_data<float, cl::Image2D>(out_img_shape[0], out_img_shape[1]);

  VLOG(4) << "x_img_shape[w,h]:" << x_img_width << " " << x_img_height;
  VLOG(4) << "y_img_shape[w,h]:" << y_img_shape[0] << " " << y_img_shape[1];
  VLOG(4) << "out_img_shape[w,h]:" << out_img_shape[0] << " "
          << out_img_shape[1];

  STL::stringstream kernel_key;
  kernel_key << kernel_func_name_ << build_options_;
  auto kernel = context.cl_context()->GetKernel(kernel_key.str());

  int arg_idx = 0;
  auto y_dims = y->dims();
  if (y_dims == ele_param_->X->dims()) {
    // elementwise_mul
    cl_int status = kernel.setArg(arg_idx, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *y_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_img);
    CL_CHECK_FATAL(status);
  } else if (y_dims.size() == 1 || y_dims.size() == 2 || y_dims.size() == 4) {
    if (y_dims.size() == 1) {
      // channel_mul
      VLOG(4) << "channel_mul";
    } else if (y_dims.size() == 2) {
      // channel_mul_d2
      VLOG(4) << "channel_mul_d2";
    } else if (y_dims.size() == 4) {
      // channel_mul_d4
      VLOG(4) << "channel_mul_d4";
    }

    auto tensor_w = x->dims()[x->dims().size() - 1];
    cl_int status = kernel.setArg(arg_idx, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *y_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, tensor_w);
    CL_CHECK_FATAL(status);
  } else {
    LOG(FATAL) << "ElementwiseMul not supported y_dims.size():"
               << y_dims.size();
  }

  auto global_work_size = cl::NDRange{static_cast<cl::size_type>(x_img_width),
                                      static_cast<cl::size_type>(x_img_height)};
  auto status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_work_size,
      cl::NullRange,
      nullptr,
      event_.get());
  CL_CHECK_FATAL(status);
  context.cl_wait_list()->emplace(out_img, event_);
}

void ElementwiseMulFloatImageCompute::UpdateParams() {
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
  VLOG(4) << "axis: " << axis;
  VLOG(4) << "batch: " << batch_;
  VLOG(4) << "channels: " << channels_;
  VLOG(4) << "num: " << num_;
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;
REGISTER_LITE_KERNEL(elementwise_mul,
                     kOpenCL,
                     kFloat,
                     kImageDefault,
                     ocl::ElementwiseMulFloatImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
