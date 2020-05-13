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
class BoxCoderComputeImage : public KernelLite<TARGET(kOpenCL),
                                               PRECISION(kFP16),
                                               DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::BoxCoderParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    boxcoder_param_ = param_.get_mutable<param_t>();
    if (boxcoder_param_->code_type == "decode_center_size" &&
        boxcoder_param_->box_normalized == true) {
      kernel_func_name_ = "decode_center_size";
    } else {
      LOG(FATAL) << "This code_type " << boxcoder_param_->code_type
                 << " doesn't support";
    }
    CHECK(context.cl_context() != nullptr);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/box_coder_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

  void Run() override {
    boxcoder_param_ = param_.get_mutable<param_t>();
    const auto& out_dims = boxcoder_param_->proposals->dims();
    auto image_shape = InitImageDimInfoWith(out_dims);

    auto* out_buf =
        boxcoder_param_->proposals->mutable_data<half_t, cl::Image2D>(
            image_shape["width"], image_shape["height"]);

#ifdef LITE_WITH_LOG
    VLOG(4) << "boxcoder input shape:  ";

#endif
    const auto* input_priorbox = boxcoder_param_->prior_box;
    const auto* input_priorboxvar = boxcoder_param_->prior_box_var;
    const auto* input_targetbox = boxcoder_param_->target_box;
    const auto& code_type = boxcoder_param_->code_type;
    if (code_type == "decode_center_size") {
      auto* prior_box_image = input_priorbox->data<half_t, cl::Image2D>();
      auto* prior_box_var_image =
          input_priorboxvar->data<half_t, cl::Image2D>();
      auto* target_box_image = input_targetbox->data<half_t, cl::Image2D>();

      int new_dims[4] = {1, 1, 1, 1};
      for (int i = 0; i < out_dims.size(); i++) {
        new_dims[4 - out_dims.size() + i] = out_dims[i];
      }
      auto& context = ctx_->As<OpenCLContext>();
      CHECK(context.cl_context() != nullptr);
      STL::stringstream kernel_key;
      kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
      auto kernel = context.cl_context()->GetKernel(kernel_key.str());

      auto default_work_size =
          DefaultWorkSize(out_dims,
                          DDim(std::vector<DDim::value_type>{
                              static_cast<int64_t>(image_shape["width"]),
                              static_cast<int64_t>(image_shape["height"])}));

      int out_C = new_dims[1];
      int out_H = new_dims[2];
#ifdef LITE_WITH_LOG
      VLOG(4) << TargetToStr(boxcoder_param_->proposals->target());
      VLOG(4) << "output shape: " << out_dims[0] << ", " << out_dims[1] << ", "
              << out_dims[2] << ", " << out_dims[3];
      VLOG(4) << "image_shape(w,h):" << image_shape["width"] << " "
              << image_shape["height"];
      VLOG(4) << "out_C = " << out_C;
      VLOG(4) << "out_H = " << out_H;
      VLOG(4) << "default_work_size = " << default_work_size[0] << ", "
              << default_work_size[1] << ", " << default_work_size[2];
#endif
      int arg_idx = 0;
      cl_int status = kernel.setArg(arg_idx++, *prior_box_image);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, *prior_box_var_image);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, *target_box_image);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, *out_buf);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, out_C);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, out_H);
      CL_CHECK_FATAL(status);
      auto global_work_size =
          cl::NDRange{static_cast<cl::size_type>(default_work_size[0]),
                      static_cast<cl::size_type>(default_work_size[2])};

      status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
          kernel,
          cl::NullRange,
          global_work_size,
          cl::NullRange,
          nullptr,
          nullptr);
      CL_CHECK_FATAL(status);

#ifdef LITE_WITH_LOG
      VLOG(4) << "global_work_size:[2D]:" << global_work_size[0] << " "
              << global_work_size[1];
#endif
    }
  }
  std::string doc() { return "Boxcoder using cl::Image, kFP16"; }

  param_t* boxcoder_param_{nullptr};
  std::string kernel_func_name_{};
  std::string build_options_{" -DCL_DTYPE_half"};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
typedef paddle::lite::kernels::opencl::BoxCoderComputeImage BoxCoder_image;

REGISTER_LITE_KERNEL(
    box_coder, kOpenCL, kFP16, kImageDefault, BoxCoder_image, ImageDefault)
    .BindInput("PriorBox",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("PriorBoxVar",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("TargetBox",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("OutputBox",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
