// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
class LayerNormImageCompute : public KernelLite<TARGET(kOpenCL),
                                                PRECISION(kFP16),
                                                DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::LayerNormParam;

  std::string doc() const override {
    return "LayerNorm using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    layer_norm_param_ = param_.get_mutable<param_t>();
    auto& context = ctx_->As<OpenCLContext>();

    auto x_dims = layer_norm_param_->X->dims();
    begin_norm_axis = layer_norm_param_->begin_norm_axis;
    VLOG(4) << "begin_norm_axis: " << begin_norm_axis;
    if (x_dims.size() == 4) {
      if (begin_norm_axis == 1) {
        kernel_func_name_ = "layer_norm_batch";
      } else if (begin_norm_axis == 2) {
        kernel_func_name_ = "layer_norm_chann";
      } else if (begin_norm_axis == 3) {
        kernel_func_name_ = "layer_norm_width";
      } else {
        LOG(FATAL) << "unsupported norm axis.";
      }
    } else if (x_dims.size() == 3) {
      if (begin_norm_axis == 1) {
        kernel_func_name_ = "layer_norm_chann";
      } else if (begin_norm_axis == 2) {
        kernel_func_name_ = "layer_norm_width";
      } else {
        LOG(FATAL) << "unsupported norm axis.";
      }
    } else {
      LOG(FATAL) << "unsupported input dim.";
    }
    auto* scale = layer_norm_param_->Scale;
    auto* bias = layer_norm_param_->Bias;
    build_options_ += (scale != nullptr) ? " -DSCALE " : "";
    build_options_ += (bias != nullptr) ? " -DBIAS " : "";
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/layer_norm_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
    if (scale != nullptr) {
      VLOG(4) << "scale_dims: " << scale->dims();
      auto* scale_cpu = scale->data<float>();
      scale_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
      auto* scale_gpu_data =
          scale_gpu_t_->mutable_data(TARGET(kOpenCL), scale->memory_size());
      TargetWrapperCL::MemcpySync(
          scale_gpu_data, scale_cpu, scale->memory_size(), IoDirection::HtoD);
      has_scale_ = true;
    }
    if (bias != nullptr) {
      VLOG(4) << "bias_dims: " << bias->dims();
      auto* bias_cpu = bias->data<float>();
      bias_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
      auto* bias_gpu_data =
          bias_gpu_t_->mutable_data(TARGET(kOpenCL), bias->memory_size());
      TargetWrapperCL::MemcpySync(
          bias_gpu_data, bias_cpu, bias->memory_size(), IoDirection::HtoD);
      has_bias_ = true;
    }
  }

  void Run() override {
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    auto* x = layer_norm_param_->X;
    auto* scale = layer_norm_param_->Scale;
    auto* bias = layer_norm_param_->Bias;

    auto x_dims = x->dims();
    auto* x_img = GET_DATA_GPU(x);

    auto* y = layer_norm_param_->Y;
    auto* mean = layer_norm_param_->Mean;
    auto* var = layer_norm_param_->Variance;

    auto y_dims = y->dims();
    auto y_image_shape = InitImageDimInfoWith(y_dims);
    auto* y_img = MUTABLE_DATA_GPU(
        y, y_image_shape["width"], y_image_shape["height"], nullptr);

    VLOG(4) << "x_dims: " << x_dims;
    VLOG(4) << "y_dims: " << y_dims;
    VLOG(4) << "mean_dims: " << mean->dims();
    VLOG(4) << "var_dims: " << var->dims();

    auto matrix_dim = x_dims.Flatten2D(begin_norm_axis);
    int batch_size = matrix_dim[0];
    int feature_size = matrix_dim[1];
    epsilon = layer_norm_param_->epsilon;
    int height = x_dims[x_dims.size() - 2];
    int width = x_dims[x_dims.size() - 1];

    int arg_idx = 0;
    auto default_work_size = DefaultGlobalWorkSize(
        y_dims,
        DDim(std::vector<DDim::value_type>{
            static_cast<int64_t>(y_image_shape["width"]),
            static_cast<int64_t>(y_image_shape["height"])}));
#ifdef LITE_WITH_LOG
    VLOG(4) << "default_work_size: " << default_work_size[0] << ", "
            << default_work_size[1] << ", " << default_work_size[2];
#endif
    auto kernel = kernel_;
    cl_int status = kernel.setArg(arg_idx++, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *y_img);
    CL_CHECK_FATAL(status);
    if (has_scale_) {
      auto* scale_buffer_p_ = GET_BUFFER_GPU(scale_gpu_t_);
      status = kernel.setArg(arg_idx++, *scale_buffer_p_);
      CL_CHECK_FATAL(status);
    }
    if (has_bias_) {
      auto* bias_buffer_p_ = GET_BUFFER_GPU(bias_gpu_t_);
      status = kernel.setArg(arg_idx++, *bias_buffer_p_);
      CL_CHECK_FATAL(status);
    }
    status = kernel.setArg(arg_idx++, batch_size);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, feature_size);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, height);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, static_cast<int>(y_image_shape["width"]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, width);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, epsilon);
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
  param_t* layer_norm_param_{nullptr};
  cl::Kernel kernel_;
  float epsilon{1e-5f};
  int begin_norm_axis{1};
  std::string kernel_func_name_{""};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  bool has_scale_{false};
  bool has_bias_{false};
  const cl::Buffer* scale_buffer_p_{nullptr};
  const cl::Buffer* bias_buffer_p_{nullptr};
  std::unique_ptr<Tensor> bias_gpu_t_{nullptr};
  std::unique_ptr<Tensor> scale_gpu_t_{nullptr};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;
REGISTER_LITE_KERNEL(layer_norm,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::LayerNormImageCompute,
                     ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Y",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindOutput("Mean", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Variance", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
