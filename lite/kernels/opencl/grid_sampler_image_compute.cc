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
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {
class GridSamplerImageCompute : public KernelLite<TARGET(kOpenCL),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::GridSamplerParam;

  std::string doc() const override {
    return "GridSampler using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/grid_sampler_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
    VLOG(4) << "kernel_key: " << kernel_key.str();
  }

  void ReInitWhenNeeded() override {
    grid_param_ = param_.get_mutable<param_t>();
    auto x_dims = grid_param_->x->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute image shape
      paddle::lite::CLImageConverterDefault default_convertor;
      out_img_shape_ =
          default_convertor.InitImageDimInfoWith(grid_param_->out->dims());

      // compute global work size
      GetGlobalWorkSize();
    }
  }

  void GetGlobalWorkSize() {
    auto default_work_size =
        DefaultGlobalWorkSize(grid_param_->out->dims(),
                              DDim(std::vector<DDim::value_type>{
                                  static_cast<int64_t>(out_img_shape_[0]),
                                  static_cast<int64_t>(out_img_shape_[1])}));
    global_work_size_ =
        cl::NDRange{static_cast<cl::size_type>(default_work_size[0]),
                    static_cast<cl::size_type>(default_work_size[1]),
                    static_cast<cl::size_type>(default_work_size[2] / 4)};
#ifdef LITE_WITH_LOG
    VLOG(4) << "default_work_size: " << default_work_size[0] << ", "
            << default_work_size[1] << ", " << default_work_size[2];
    VLOG(4) << "global_work_size_:[2D]:" << global_work_size_[0] << " "
            << global_work_size_[1] << " " << global_work_size_[2];
#endif
  }

  void Run() override {
    auto* x = grid_param_->x;
    auto* grid = grid_param_->grid;
    auto* out = grid_param_->out;

    auto out_dims = out->dims();
    int out_height = out_dims[2];
    int out_width = out_dims[3];

    auto* x_img = x->data<half_t, cl::Image2D>();
    auto* grid_img = x->data<half_t, cl::Image2D>();
    auto* out_img = out->mutable_data<half_t, cl::Image2D>(out_img_shape_[0],
                                                           out_img_shape_[1]);

#ifdef LITE_WITH_LOG
    auto in_dims = x->dims();
    VLOG(4) << "x->target():" << TargetToStr(x->target());
    VLOG(4) << "out->target():" << TargetToStr(out->target());
    VLOG(4) << "x->dims():" << in_dims;
    VLOG(4) << "out->dims():" << out_dims;
    // VLOG(4) << "x_image: " << x_img;
    // VLOG(4) << "grid_img: " << grid_img;
    // VLOG(4) << "out_image" << out_img;
    VLOG(4) << "out_img_shape_[w,h]:" << out_img_shape_[0] << " "
            << out_img_shape_[1];
#endif

    cl_int status;
    auto kernel = kernel_;
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *grid_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, out_height);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, out_width);
    CL_CHECK_FATAL(status);

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size_,
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

 protected:
  param_t* grid_param_{nullptr};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  DDim out_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  std::string kernel_func_name_{"grid_sampler"};
  cl::Kernel kernel_;
  cl::NDRange global_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  std::string build_options_{"-DCL_DTYPE_half"};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;
REGISTER_LITE_KERNEL(grid_sampler,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::GridSamplerImageCompute,
                     ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Grid",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
