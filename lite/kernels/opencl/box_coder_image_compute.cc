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
class BoxCoderComputeImage : public KernelLite<TARGET(kOpenCL),
                                               PRECISION(kFP16),
                                               DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::BoxCoderParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    boxcoder_param_ = param_.get_mutable<param_t>();
    if (boxcoder_param_->code_type == "decode_center_size" &&
        boxcoder_param_->axis == 0) {
      kernel_func_name_ = "decode_center_size_axis0";
    } else if (boxcoder_param_->code_type == "decode_center_size" &&
               boxcoder_param_->axis == 1) {
      kernel_func_name_ = "decode_center_size_axis1";
    } else if (boxcoder_param_->code_type == "encode_center_size") {
      kernel_func_name_ = "encode_center_size";
    } else {
      LOG(FATAL) << "This code_type " << boxcoder_param_->code_type
                 << " doesn't support";
    }

    if (boxcoder_param_->prior_box_var != nullptr) {
      build_options_ += " -DPRIOR_BOX_VAR";
    }

    if (boxcoder_param_->prior_box->persistable() &&
        boxcoder_param_->prior_box_var->persistable()) {
      // ssd_boxes_calc_offline_pass was applied.
      // prior_box & prior_box_var are as const weights now.
      // So we need to copy prior_box & prior_box_var from cpu to gpu.
      CLImageConverterNormal converter;
      priorbox_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
      priorboxvar_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
      auto priorbox_cpu_image = std::unique_ptr<Tensor>(new Tensor);
      auto priorboxvar_cpu_image = std::unique_ptr<Tensor>(new Tensor);

      const auto* priorbox_cpu = boxcoder_param_->prior_box->data<float>();
      const auto& priorbox_dims = boxcoder_param_->prior_box->dims();
      auto image_shape = converter.InitImageDimInfoWith(priorbox_dims);
      priorbox_cpu_image->Resize({1, image_shape[0], image_shape[1], 4});
      auto* priorbox_image_data = MUTABLE_DATA_CPU(priorbox_cpu_image);
      converter.NCHWToImage(
          const_cast<float*>(priorbox_cpu), priorbox_image_data, priorbox_dims);
      MUTABLE_DATA_GPU(priorbox_gpu_image_,
                       image_shape[0],
                       image_shape[1],
                       priorbox_image_data);

      if (boxcoder_param_->prior_box_var != nullptr) {
        const auto* priorboxvar_cpu =
            boxcoder_param_->prior_box_var->data<float>();
        const auto& priorboxvar_dims = boxcoder_param_->prior_box_var->dims();
        image_shape = converter.InitImageDimInfoWith(priorboxvar_dims);
        priorboxvar_cpu_image->Resize({1, image_shape[0], image_shape[1], 4});
        auto* priorboxvar_image_data = MUTABLE_DATA_CPU(priorboxvar_cpu_image);
        converter.NCHWToImage(const_cast<float*>(priorboxvar_cpu),
                              priorboxvar_image_data,
                              priorboxvar_dims);
        MUTABLE_DATA_GPU(priorboxvar_gpu_image_,
                         image_shape[0],
                         image_shape[1],
                         priorboxvar_image_data);

        priorbox_image_ = DATA_GPU(priorbox_gpu_image_);
        priorboxvar_image_ = DATA_GPU(priorboxvar_gpu_image_);
      }

    } else {
      priorbox_image_ = GET_DATA_GPU(boxcoder_param_->prior_box);
      if (boxcoder_param_->prior_box_var != nullptr) {
        priorboxvar_image_ = GET_DATA_GPU(boxcoder_param_->prior_box_var);
      }
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

    auto* out_buf = MUTABLE_DATA_GPU(boxcoder_param_->proposals,
                                     image_shape["width"],
                                     image_shape["height"],
                                     nullptr);

    const auto* input_targetbox = boxcoder_param_->target_box;
    const auto& code_type = boxcoder_param_->code_type;

    auto* target_box_image = GET_DATA_GPU(input_targetbox);

    int new_dims[4] = {1, 1, 1, 1};
    for (int i = 0; i < out_dims.size(); i++) {
      new_dims[4 - out_dims.size() + i] = out_dims[i];
    }
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    auto default_work_size = DefaultGlobalWorkSize(
        out_dims,
        DDim(std::vector<DDim::value_type>{
            static_cast<int64_t>(image_shape["width"]),
            static_cast<int64_t>(image_shape["height"])}));

    int out_C = new_dims[1];
    int out_H = new_dims[2];
    int normalized = static_cast<int>(boxcoder_param_->box_normalized);
#ifdef LITE_WITH_LOG
    VLOG(4) << TargetToStr(boxcoder_param_->proposals->target());
    VLOG(4) << "input[PriorBox] shape: " << boxcoder_param_->prior_box->dims();
    VLOG(4) << "input[TargetBox] shape: "
            << boxcoder_param_->target_box->dims();
    VLOG(4) << "output[OutputBox] shape: " << out_dims;
    VLOG(4) << "image_shape(w,h):" << image_shape["width"] << " "
            << image_shape["height"];
    VLOG(4) << "out_C = " << out_C;
    VLOG(4) << "out_H = " << out_H;
    VLOG(4) << "default_work_size = " << default_work_size[0] << ", "
            << default_work_size[1] << ", " << default_work_size[2];
#endif
    std::vector<float> variance = boxcoder_param_->variance;
    float variance_4[4] = {1, 1, 1, 1};
    if ((!(variance.empty()))) {
      const float* variance_data = variance.data();
      variance_4[0] = static_cast<float>(variance_data[0]);
      variance_4[1] = static_cast<float>(variance_data[1]);
      variance_4[2] = static_cast<float>(variance_data[2]);
      variance_4[3] = static_cast<float>(variance_data[3]);
    }

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx++, *priorbox_image_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *target_box_image);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, out_C);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, out_H);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, normalized);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, variance_4);
    CL_CHECK_FATAL(status);
    if (boxcoder_param_->prior_box_var != nullptr) {
      status = kernel.setArg(arg_idx++, *priorboxvar_image_);
      CL_CHECK_FATAL(status);
    }

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(default_work_size[0]),
                    static_cast<cl::size_type>(default_work_size[2])};

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }
  std::string doc() { return "Boxcoder using cl::Image, kFP16"; }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  param_t* boxcoder_param_{nullptr};
  std::string kernel_func_name_{};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  std::unique_ptr<Tensor> priorbox_gpu_image_{nullptr};
  std::unique_ptr<Tensor> priorboxvar_gpu_image_{nullptr};
  const cl::Image2D* priorbox_image_{nullptr};
  const cl::Image2D* priorboxvar_image_{nullptr};
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
