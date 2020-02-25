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

#include <memory>
#include <string>
#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/logging.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ElementwiseMulImageCompute
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ElementwiseParam;

  std::string doc() const override {
    return "ElementwiseMul using cl::Image2D(ImageDefault/RGBA), kFP32";
  }

  void PrepareForRun() override {
    ele_param_ = param_.get_mutable<param_t>();
    auto* y = ele_param_->Y;
    auto* x = ele_param_->X;
    auto y_dims = y->dims();
    auto x_dims = x->dims();
    if (y_dims == x_dims) {
      kernel_func_name_ = "elementwise_mul";
    } else if (y_dims.size() == 1) {
      kernel_func_name_ = "channel_mul_d1";
    } else if (y_dims.size() == 2) {
      if (x_dims[0] == y_dims[0] && x_dims[1] == y_dims[1]) {
        kernel_func_name_ = "channel_mul_d2_nc";
      } else {
        kernel_func_name_ = "channel_mul_d2_hw";
      }
    } else if (y_dims.size() == 4) {
      kernel_func_name_ = "channel_mul_d4";
    } else {
      LOG(FATAL) << "ElementwiseMul not supported y_dims.size():"
                 << y_dims.size()
                 << ", x_dims.size():" << ele_param_->X->dims().size();
    }
    VLOG(4) << "kernel_func_name_:" << kernel_func_name_;
    VLOG(4) << "y_dims:" << y_dims;
    VLOG(4) << "y_dims.size():" << y_dims.size();

    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/elementwise_mul_kernel.cl", build_options_);
  }

  void Run() override {
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
    auto x_img_shape =
        default_convertor.InitImageDimInfoWith(x->dims());  // w, h
    auto x_img_width = x_img_shape[0];
    auto x_img_height = x_img_shape[1];
    auto out_img_shape =
        default_convertor.InitImageDimInfoWith(out->dims());  // w, h
    auto y_img_shape = default_convertor.InitImageDimInfoWith(y->dims());

    auto* x_img = x->data<half_t, cl::Image2D>();
    auto* y_img = y->data<half_t, cl::Image2D>();
    auto* out_img = out->mutable_data<half_t, cl::Image2D>(out_img_shape[0],
                                                           out_img_shape[1]);

    VLOG(4) << "x_img_shape[w,h]:" << x_img_width << " " << x_img_height;
    VLOG(4) << "y_img_shape[w,h]:" << y_img_shape[0] << " " << y_img_shape[1];
    VLOG(4) << "out_img_shape[w,h]:" << out_img_shape[0] << " "
            << out_img_shape[1];

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    auto y_dims = y->dims();
    auto x_dims = x->dims();
    if (y_dims == x_dims) {
      // kernel: elementwise_mul(channel_mul_d4)
      cl_int status = kernel.setArg(arg_idx, *x_img);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *y_img);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *out_img);
      CL_CHECK_FATAL(status);
    } else if (y_dims.size() == 1 || y_dims.size() == 4) {
      auto tensor_w = x_dims[x_dims.size() - 1];
      VLOG(4) << "tensor_w:" << tensor_w;
      // kernel: channel_mul_d1 / channel_mul_d4
      cl_int status = kernel.setArg(arg_idx, *x_img);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *y_img);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *out_img);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, static_cast<const int>(tensor_w));
      CL_CHECK_FATAL(status);
    } else if (y_dims.size() == 2) {
      if (x_dims[0] == y_dims[0] && x_dims[1] == y_dims[1]) {
        auto tensor_w = x_dims[x_dims.size() - 1];
        VLOG(4) << "tensor_w:" << tensor_w;
        // kernel: channel_mul_d2_nc
        cl_int status = kernel.setArg(arg_idx, *x_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, *y_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, *out_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, static_cast<const int>(tensor_w));
        CL_CHECK_FATAL(status);
      } else {
        auto y_tensor_h = y->dims()[0];
        auto y_tensor_w = y->dims()[1];
        VLOG(4) << "y_tensor_w:" << y_tensor_w << " y_tensor_h:" << y_tensor_h;
        // kernel: channel_mul_d2_hw
        cl_int status = kernel.setArg(arg_idx, *x_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, *y_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, *out_img);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, static_cast<const int>(y_tensor_w));
        CL_CHECK_FATAL(status);
        status = kernel.setArg(++arg_idx, static_cast<const int>(y_tensor_h));
        CL_CHECK_FATAL(status);
      }
    } else {
      LOG(FATAL) << "ElementwiseMul not supported y_dims.size():"
                 << y_dims.size();
    }

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(x_img_width),
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

    VLOG(4) << "global_work_size:[2D]:" << x_img_width << " " << x_img_height;
  }

 protected:
  param_t* ele_param_{nullptr};
  std::string kernel_func_name_{"elementwise_mul"};
  std::string build_options_{"-DCL_DTYPE_half"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;
REGISTER_LITE_KERNEL(elementwise_mul,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::ElementwiseMulImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
