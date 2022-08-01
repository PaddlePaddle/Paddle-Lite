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
#include "lite/utils/log/logging.h"
#include "lite/utils/replace_stl/stream.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {
class LrnImageCompute : public KernelLite<TARGET(kOpenCL),
                                          PRECISION(kFP16),
                                          DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::LrnParam;

  std::string doc() const override {
    return "Lrn using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    lrn_param_ = param_.get_mutable<param_t>();

    auto& context = ctx_->As<OpenCLContext>();
    n_ = lrn_param_->n;
    k_ = lrn_param_->k;
    alpha_ = lrn_param_->alpha;
    beta_ = lrn_param_->beta;
    norm_region_ = lrn_param_->norm_region;
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/lrn_kernel.cl", build_options_, time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
  }

  void Run() override {
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    auto* x = lrn_param_->X;
    auto* out = lrn_param_->Out;
    if (norm_region_ != "AcrossChannels") {
      LOG(FATAL) << "This norm_region_: " << norm_region_ << "doesn't support";
      return;
    }
    auto out_dims = out->dims();
    auto in_dims = x->dims();

#ifdef LITE_WITH_LOG
    VLOG(4) << "x->target(): " << TargetToStr(x->target());
    VLOG(4) << "out->target(): " << TargetToStr(out->target());
    VLOG(4) << "in->dims(): " << in_dims;
    VLOG(4) << "out->dims(): " << out_dims;
    VLOG(4) << "lrn param: ";
    VLOG(4) << "n: " << n_;
    VLOG(4) << "k: " << k_;
    VLOG(4) << "alpha: " << alpha_;
    VLOG(4) << "beta: " << beta_;
    VLOG(4) << "norm_region: " << norm_region_;
#endif

    auto out_image_shape = InitImageDimInfoWith(out_dims);
    auto* x_img = GET_DATA_GPU(x);
    // VLOG(4) << "x_image: " << x_img;

    auto* out_img = MUTABLE_DATA_GPU(
        out, out_image_shape["width"], out_image_shape["height"], nullptr);

#ifdef LITE_WITH_LOG
    // VLOG(4) << "out_image" << out_img;
    VLOG(4) << "out_image_shape[w,h]:" << out_image_shape["width"] << " "
            << out_image_shape["height"];
#endif

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    int out_channel = out_dims[1];
    int out_width = out_dims[3];
    auto default_work_size = DefaultGlobalWorkSize(
        out_dims,
        DDim(std::vector<DDim::value_type>{
            static_cast<int64_t>(out_image_shape["width"]),
            static_cast<int64_t>(out_image_shape["height"])}));
#ifdef LITE_WITH_LOG
    VLOG(4) << "default_work_size: " << default_work_size[0] << ", "
            << default_work_size[1] << ", " << default_work_size[2];
#endif
    cl_int status = kernel.setArg(arg_idx++, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, out_channel);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, out_width);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, n_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, k_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, alpha_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, beta_);
    CL_CHECK_FATAL(status);

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(default_work_size[0]),
                    static_cast<cl::size_type>(default_work_size[1]),
                    static_cast<cl::size_type>(default_work_size[2])};

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
#ifdef LITE_WITH_LOG
    VLOG(4) << "global_work_size:[2D]:" << global_work_size[0] << " "
            << global_work_size[1] << " " << global_work_size[2];
#endif
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 protected:
  param_t* lrn_param_{nullptr};
  int n_{5};
  float alpha_{1e-4f};
  float beta_{0.75};
  float k_{1.};
  std::string norm_region_{"AcrossChannels"};
  std::string kernel_func_name_{"lrn"};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;
REGISTER_LITE_KERNEL(
    lrn, kOpenCL, kFP16, kImageDefault, ocl::LrnImageCompute, ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindOutput("MidOut", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
