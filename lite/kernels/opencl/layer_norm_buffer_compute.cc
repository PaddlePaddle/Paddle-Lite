// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
class LayerNormBufferCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::LayerNormParam;

  std::string doc() const override {
    return "LayerNorm using cl::Buffer, kFP16";
  }

  void PrepareForRun() override {
    layer_norm_param_ = param_.get_mutable<param_t>();
    auto* scale = layer_norm_param_->Scale;
    auto* bias = layer_norm_param_->Bias;
    bool fp16_flag =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16);
    if (scale != nullptr) {
      auto* scale_cpu = scale->data<float>();
      scale_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
      auto* scale_gpu_data =
          scale_gpu_t_->mutable_data(TARGET(kOpenCL), scale->memory_size());
      TargetWrapperCL::MemcpySync(
          scale_gpu_data, scale_cpu, scale->memory_size(), IoDirection::HtoD);
      has_scale_ = true;
    }
    if (bias != nullptr) {
      auto* bias_cpu = bias->data<float>();
      bias_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
      auto* bias_gpu_data =
          bias_gpu_t_->mutable_data(TARGET(kOpenCL), bias->memory_size());
      TargetWrapperCL::MemcpySync(
          bias_gpu_data, bias_cpu, bias->memory_size(), IoDirection::HtoD);
      has_bias_ = true;
    }
  }

  void SetGlobalLocal() {
    // compute global/local work size
    auto device_info = CLRuntime::Global()->GetDeviceInfo();
    int max_work_item_size1 = device_info["CL_DEVICE_MAX_WORK_ITEM_SIZES_1"];
    int lws0 = 1;
    int lws1 =
        std::min(max_work_item_size1, std::min(256, static_cast<int>(w_)));
    int lws2 = 1;
    global_work_size_ = cl::NDRange{static_cast<cl::size_type>(c_),
                                    static_cast<cl::size_type>(lws1),
                                    static_cast<cl::size_type>(b_ * h_)};
    local_work_size_ = cl::NDRange{static_cast<cl::size_type>(lws0),
                                   static_cast<cl::size_type>(lws1),
                                   static_cast<cl::size_type>(lws2)};
  }

  void ReInitWhenNeeded() override {
    auto x_dims = layer_norm_param_->X->dims();
    auto y_dims = layer_norm_param_->Y->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;
      if (y_dims.size() == 2) {
        w_ = y_dims[1];
        h_ = y_dims[0];
      } else if (y_dims.size() == 3) {
        c_ = y_dims[0];
        w_ = y_dims[2];
        h_ = y_dims[1];
      } else if (y_dims.size() == 4) {
        c_ = y_dims[1];
        w_ = y_dims[3];
        b_ = y_dims[0];
        h_ = y_dims[2];
      }
      // auto x_dims = layer_norm_param_->X->dims();
      begin_norm_axis = layer_norm_param_->begin_norm_axis;
      if (x_dims.size() == 4) {
        if (begin_norm_axis == 1) {
          kernel_func_name_ = "layer_norm_buffer_batch";
        } else if (begin_norm_axis == 2) {
          kernel_func_name_ = "layer_norm_buffer_chann";
        } else if (begin_norm_axis == 3) {
          kernel_func_name_ = "layer_norm_buffer_width";
        } else {
          LOG(FATAL) << "unsupported norm axis.";
        }
      } else if (x_dims.size() == 3) {
        if (begin_norm_axis == 1) {
          kernel_func_name_ = "layer_norm_buffer_chann";
        } else if (begin_norm_axis == 2) {
          kernel_func_name_ = "layer_norm_buffer_width";
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
      // build_options_ += "-DLOCAL_MEM_128";
      auto& context = ctx_->As<OpenCLContext>();
      context.cl_context()->AddKernel(kernel_func_name_,
                                      "buffer/layer_norm_kernel.cl",
                                      build_options_,
                                      time_stamp_);
      VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
      STL::stringstream kernel_key;
      kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
      kernel_ = context.cl_context()->GetKernel(kernel_key.str());
      SetGlobalLocal();
    }
  }

  void Run() override {
    auto* x = layer_norm_param_->X;
    auto x_dims = x->dims();
    auto* x_buf = GET_BUFFER_GPU(x);
    auto* y_buf = MUTABLE_BUFFER_GPU(layer_norm_param_->Y);

    auto matrix_dim = x_dims.Flatten2D(begin_norm_axis);
    int batch_size = matrix_dim[0];
    int feature_size = matrix_dim[1];
    epsilon = layer_norm_param_->epsilon;
    int height = x_dims[x_dims.size() - 2];
    int width = x_dims[x_dims.size() - 1];

    int arg_idx = 0;
    auto kernel = kernel_;
    cl_int status = kernel.setArg(arg_idx++, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *y_buf);
    CL_CHECK_FATAL(status);
    if (has_scale_) {
      auto* scale_buffer_p_ = scale_gpu_t_->data<float, cl::Buffer>();
      status = kernel.setArg(arg_idx++, *scale_buffer_p_);
      CL_CHECK_FATAL(status);
    }
    if (has_bias_) {
      auto* bias_buffer_p_ = bias_gpu_t_->data<float, cl::Buffer>();
      status = kernel.setArg(arg_idx++, *bias_buffer_p_);
      CL_CHECK_FATAL(status);
    }
    status = kernel.setArg(arg_idx++, static_cast<const int>(batch_size));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, static_cast<const int>(feature_size));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, static_cast<const int>(height));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, static_cast<const int>(c_));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, static_cast<const int>(width));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, static_cast<const float>(epsilon));
    CL_CHECK_FATAL(status);

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size_,
                                  local_work_size_,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
#ifdef LITE_WITH_LOG
    VLOG(4) << "global_work_size:[2D]:" << global_work_size_[0] << " "
            << global_work_size_[1] << " " << global_work_size_[2];
#endif
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
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
  cl::NDRange global_work_size_;
  cl::NDRange local_work_size_;
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  int c_{1};
  int w_{1};
  int b_{1};
  int h_{1};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;
REGISTER_LITE_KERNEL(
    layer_norm, kOpenCL, kFP16, kNCHW, ocl::LayerNormBufferCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Mean", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Variance", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
