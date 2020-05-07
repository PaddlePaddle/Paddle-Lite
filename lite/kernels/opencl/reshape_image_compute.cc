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
#include "lite/utils/logging.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

// reshape operator
class ReshapeComputeFloatImage : public KernelLite<TARGET(kOpenCL),
                                                   PRECISION(kFP16),
                                                   DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ReshapeParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/reshape_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const Tensor* const x = param.x;

    const auto x_dims = x->dims();
    const std::map<std::string, size_t>& input_image_shape =
        InitImageDimInfoWith(x_dims);

    const int64_t& input_image_width = input_image_shape.at("width");
    const int64_t& input_image_height = input_image_shape.at("height");

    const cl::Image2D* const x_image = x->data<half_t, cl::Image2D>();

    const std::vector<int>& shape_vct = param.shape_vct;
    Tensor* const output = param.output;
    const DDimLite& out_dims = output->dims();
    VLOG(4) << "out_dims= " << out_dims;

    const std::map<std::string, size_t>& out_image_shape =
        InitImageDimInfoWith(out_dims);
    cl::Image2D* const out_image = output->mutable_data<half_t, cl::Image2D>(
        out_image_shape.at("width"), out_image_shape.at("height"));
#ifdef LITE_WITH_LOG
    VLOG(4) << "out_dims=   " << out_dims;
#endif
    const std::vector<size_t>& default_work_size = DefaultWorkSize(
        out_dims,
        DDim(std::vector<DDim::value_type>{
            static_cast<int64_t>(out_image_shape.at("width")),
            static_cast<int64_t>(out_image_shape.at("height"))}));

    int x_v_dims[4] = {1, 1, 1, 1};
    int out_v_dims[4] = {1, 1, 1, 1};
    // 1 1000 1 1
    for (int i = 0; i < x_dims.size(); i++) {
      x_v_dims[4 - x_dims.size() + i] = x_dims[i];
    }
    // 1 1 1 1000
    for (int i = 0; i < out_dims.size(); i++) {
      out_v_dims[4 - out_dims.size() + i] = out_dims[i];
    }

    int out_C = out_v_dims[1];
    int out_H = out_v_dims[2];
    int out_W = out_v_dims[3];
    int in_W = x_v_dims[3];
    int in_H = x_v_dims[2];
    int in_Stride0 = in_W;
    int in_Stride1 = x_v_dims[2] * x_v_dims[3];
    int in_Stride2 = x_v_dims[1] * x_v_dims[2] * x_v_dims[3];
    int out_Stride0 = out_W;
    int out_Stride1 = out_H * out_W;
    int out_Stride2 = out_C * out_H * out_W;

#ifdef LITE_WITH_LOG
    VLOG(4) << "out_C=" << out_C;
    VLOG(4) << "out_H=" << out_H;
    VLOG(4) << "out_W=" << out_W;
    VLOG(4) << "in_W=" << in_W;
    VLOG(4) << "default_work_size= " << default_work_size[0] << ", "
            << default_work_size[1] << ", " << default_work_size[2];
    VLOG(4) << "in_Stride0=" << in_Stride0;
    VLOG(4) << "in_Stride1=" << in_Stride1;
    VLOG(4) << "out_Stride0=" << out_Stride0;
    VLOG(4) << "out_Stride1=" << out_Stride1;
#endif

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

#ifdef LITE_WITH_LOG
    VLOG(4) << TargetToStr(x->target());
    VLOG(4) << TargetToStr(param.output->target());
#endif

    int arg_idx = 0;
    cl_int status;
    status = kernel.setArg(arg_idx, *x_image);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_image);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, out_C);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, out_H);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, out_W);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, in_W);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, in_H);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, in_Stride0);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, in_Stride1);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, in_Stride2);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, out_Stride0);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, out_Stride1);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, out_Stride2);
    CL_CHECK_FATAL(status);

    auto global_work_size =
        cl::NDRange{static_cast<size_t>(default_work_size.data()[0]),
                    static_cast<size_t>(default_work_size.data()[1]),
                    static_cast<size_t>(default_work_size.data()[2])};

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
  std::string kernel_func_name_{"reshape"};
  std::string build_options_{"-DCL_DTYPE_half"};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reshape,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ReshapeComputeFloatImage,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("ShapeTensor", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Shape", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape2,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ReshapeComputeFloatImage,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("ShapeTensor", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Shape", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ReshapeComputeFloatImage,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Shape", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten2,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ReshapeComputeFloatImage,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Shape", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
