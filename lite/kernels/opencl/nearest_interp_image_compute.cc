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

#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class NearestInterpComputeImageDefault
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::InterpolateParam;

  std::string doc() const override {
    return "NearestInterp using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/nearest_interp_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.X->dims();
    const auto& y_dims = param.Out->dims();
    auto* x_img =
        param.X->data<half_t,
                      cl::Image2D>();  // use half_t represents half float
    auto out_image_shape = InitImageDimInfoWith(y_dims);
    auto* out_img = param.Out->mutable_data<half_t, cl::Image2D>(  // use half_t
        // represents half float
        out_image_shape["width"],
        out_image_shape["height"]);

    float scale_h = y_dims[2] / x_dims[2];
    float scale_w = y_dims[3] / x_dims[3];
    int in_dims_h = x_dims[2];
    int out_dims_h = y_dims[2];
    int in_dims_w = x_dims[3];
    int out_dims_w = y_dims[3];

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const float>(scale_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const float>(scale_w));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims_w));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims_w));
    CL_CHECK_FATAL(status);

#ifdef LITE_WITH_LOG
    VLOG(4) << TargetToStr(param.X->target());
    VLOG(4) << TargetToStr(param.Out->target());
    VLOG(4) << "out_image_shape(w,h):" << out_image_shape["width"] << " "
            << out_image_shape["height"];
    VLOG(4) << "x_dims[" << x_dims.size() << "D]:" << x_dims[0] << " "
            << x_dims[1] << " " << x_dims[2] << " " << x_dims[3];
    VLOG(4) << "y_dims[" << y_dims.size() << "D]:" << y_dims[0] << " "
            << y_dims[1] << " " << y_dims[2] << " " << y_dims[3];
#endif

    const std::vector<size_t>& default_work_size =
        DefaultWorkSize(y_dims,
                        DDim(std::vector<DDim::value_type>{
                            static_cast<int64_t>(out_image_shape["width"]),
                            static_cast<int64_t>(out_image_shape["height"])}));
    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(default_work_size.data()[0]),
                    static_cast<cl::size_type>(default_work_size.data()[1]),
                    static_cast<cl::size_type>(default_work_size.data()[2])};

    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        nullptr);
    CL_CHECK_FATAL(status);
  }

 private:
  std::string kernel_func_name_{"nearest_interp"};
  std::string build_options_{" -DCL_DTYPE_half"};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    nearest_interp,
    kOpenCL,
    kFP16,
    kImageDefault,
    paddle::lite::kernels::opencl::NearestInterpComputeImageDefault,
    ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
