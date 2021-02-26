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

#include <vector>
#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"
#include "lite/utils/string.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class SliceComputeImage2D : public KernelLite<TARGET(kOpenCL),
                                              PRECISION(kFP16),
                                              DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::SliceParam;

  std::string doc() const override { return "Slice using cl::Image2D, kFP16"; }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/slice_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

  void Run() override {
    const auto& param = *param_.get_mutable<param_t>();
    const auto& in_dims = param.X->dims();
    auto* x_img = GET_DATA_GPU(param.X);
    auto& out_dims = param.Out->dims();

    std::vector<int> axes = param.axes;
    std::vector<int32_t> starts = param.starts;
    std::vector<int32_t> ends = param.ends;

    if (axes.size() > 1 || axes[0] != 1) {
      LOG(FATAL) << "opencl slice_image only support channel slice ";
    }

    int axis = axes[0];
    int start = starts[0];
    int end = ends[0];
    int dim_w = in_dims[axis + 2];

    auto out_image_shape = InitImageDimInfoWith(out_dims);
    auto* out_img = MUTABLE_DATA_GPU(param.Out,
                                     out_image_shape["width"],
                                     out_image_shape["height"],
                                     nullptr);

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    cl_int status;
    int arg_idx = 0;
    status = kernel.setArg(arg_idx, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, start);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, end);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, dim_w);
    CL_CHECK_FATAL(status);

    const std::vector<size_t>& default_work_size = DefaultGlobalWorkSize(
        out_dims,
        DDim(std::vector<DDim::value_type>{
            static_cast<int64_t>(out_image_shape["width"]),
            static_cast<int64_t>(out_image_shape["height"])}));
    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(default_work_size.data()[0]),
                    static_cast<cl::size_type>(default_work_size.data()[1]),
                    static_cast<cl::size_type>(default_work_size.data()[2])};

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  std::string kernel_func_name_{"slice"};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(slice,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::SliceComputeImage2D,
                     image2d)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
