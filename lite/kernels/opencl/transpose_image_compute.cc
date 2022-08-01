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

#include "lite/backends/opencl/cl_half.h"
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

#undef LITE_WITH_LOG

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class TransposeComputeFloatImage
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::TransposeParam;

  std::vector<int> CalStrides(const DDim& dims) {
    int dsize = dims.size();
    std::vector<int> strides(dsize, 1);
    for (int i = dsize - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
  }

  std::vector<int> CalIndex(const std::vector<int>& strides, int offset) {
    int dsize = strides.size();
    std::vector<int> index(dsize, 0);
    for (int i = 0; i < dsize; i++) {
      index[i] = offset / strides[i];
      offset %= strides[i];
    }
    return index;
  }

  std::vector<int> TransIndex(const std::vector<int>& in_index,
                              const std::vector<int>& axis) {
    std::vector<int> out_index(in_index.size(), 0);
    for (int i = 0; i < axis.size(); i++) {
      out_index[i] = in_index[axis[i]];
    }
    return out_index;
  }

  int CalOffset(const std::vector<int>& strides,
                const std::vector<int>& index) {
    int offset = 0;
    for (int i = 0; i < index.size(); i++) {
      offset += strides[i] * index[i];
    }
    return offset;
  }

  void PrepareForRun() override {
    transpose_param_ = param_.get_mutable<param_t>();
    axis_ = transpose_param_->axis;
    auto x = transpose_param_->x;
    x_tensor_dims_ = x->dims();
    auto output = transpose_param_->output;
    output_tensor_dims_ = output->dims();
    auto output_image_shape = InitImageDimInfoWith(output_tensor_dims_);
    output_image_h_ = output_image_shape.at("height");
    output_image_w_ = output_image_shape.at("width");
    VLOG(4) << "x_tensor_dims_: " << x_tensor_dims_;

    if (axis_.size() == 3) {
      VLOG(4) << "Extend CHW to 1CHW";
      axis_.insert(axis_.begin(), 0);  // extend batch dim is 1
      for (int i = 1; i < axis_.size(); ++i) {
        axis_[i]++;
      }
    }
    if (axis_.size() == 4) {
      std::vector<int> tmp = axis_;
      sort(tmp.begin(), tmp.end());
      if (tmp == std::vector<int>({0, 1, 2, 3}) &&
          axis_ != std::vector<int>({0, 1, 2, 3})) {
        if (axis_[1] == 1) {
          kernel_func_name_ = "transpose4d_perm_without_chann";
          build_options_ += " -DNON_CHANNEL_PERMUTATION ";
          build_options_ += " -DPERM";
          for (int i = 0; i < axis_.size(); ++i) {
            build_options_ += std::to_string(axis_[i]);
          }
        } else {
          kernel_func_name_ = "transpose4d_perm_with_channel";
          if (axis_[0] == 0) {
            build_options_ += " -DN2N";
          } else if (axis_[0] == 1) {
            build_options_ += " -DC2N";
          } else if (axis_[0] == 2) {
            build_options_ += " -DH2N";
          } else if (axis_[0] == 3) {
            build_options_ += " -DW2N";
          }
          if (axis_[2] == 1) {
            build_options_ += " -DC2H";
          } else if (axis_[2] == 3) {
            build_options_ += " -DW2H";
          } else if (axis_[2] == 0) {
            build_options_ += " -DN2H";
          } else if (axis_[2] == 2) {
            build_options_ += " -DH2H";
          }
          if (axis_[1] == 2) {
            build_options_ += " -DH2C";
          } else if (axis_[1] == 3) {
            build_options_ += " -DW2C";
          } else if (axis_[1] == 0) {
            build_options_ += " -DN2C";
          } else if (axis_[1] == 1) {
            build_options_ += " -DC2C";
          }
          if (axis_[3] == 3) {
            build_options_ += " -DW2W";
          } else if (axis_[3] == 1) {
            build_options_ += " -DC2W";
          } else if (axis_[3] == 2) {
            build_options_ += " -DH2W";
          } else if (axis_[3] == 0) {
            build_options_ += " -DN2W";
          }
        }
        kernel_path_ = "image/transpose_kernel.cl";
      } else {
        LOG(FATAL) << "Unsupported axis permutation for current lite OpenCL "
                      "kernel! ";
      }
    } else if (axis_.size() == 2) {
      std::vector<int> tmp = axis_;
      sort(tmp.begin(), tmp.end());
      if (tmp == std::vector<int>({0, 1}) &&
          axis_ != std::vector<int>({0, 1})) {
        kernel_func_name_ = "transpose_2d";
        kernel_path_ = "image/transpose_kernel.cl";
      } else {
        LOG(FATAL) << "Unsupported axis permutation for current lite OpenCL "
                      "kernel! ";
      }
    } else if (axis_.size() > 4) {
      kernel_func_name_ = "transpose_general_buffer";
      kernel_path_ = "buffer/transpose_kernel.cl";
    } else {
      LOG(FATAL) << "Unsupported axis permutation for current lite OpenCL "
                    "kernel! ";
    }

    if (kernel_func_name_ == "transpose_general_buffer") {
      build_options_ = "-DCL_DTYPE_float";
      // create kernels of im2buf and buf2im
      auto im2buf_kernels = KernelRegistry::Global().Create(
          "layout", TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW));
      auto buf2im_kernels =
          KernelRegistry::Global().Create("layout",
                                          TARGET(kOpenCL),
                                          PRECISION(kAny),
                                          DATALAYOUT(kImageDefault));

      im2buf_kernel_ = std::move(im2buf_kernels.front());
      buf2im_kernel_ = std::move(buf2im_kernels.front());

      // calc output shape
      std::vector<int64_t> new_output_tensor_shape(x_tensor_dims_.size(), 0);
      for (size_t i = 0; i < x_tensor_dims_.size(); i++) {
        new_output_tensor_shape[i] = x_tensor_dims_[axis_[i]];
      }
      output->Resize(new_output_tensor_shape);
      output_tensor_dims_ = output->dims();
      // calc in/out index of transpose
      std::vector<int> x_tensor_strides = CalStrides(x_tensor_dims_);
      std::vector<int> output_tensor_strides = CalStrides(output_tensor_dims_);
      std::vector<int> output_tensor_idxs_vec(output->dims().production());
      for (size_t i = 0; i < x_tensor_dims_.production(); i++) {
        std::vector<int> x_tensor_index = CalIndex(x_tensor_strides, i);
        std::vector<int> out_tensor_index = TransIndex(x_tensor_index, axis_);
        output_tensor_idxs_vec[i] =
            CalOffset(output_tensor_strides, out_tensor_index);
      }

      // copy output_tensor_idxs_vec data to gpu
      output_tensor_idxs_t_ = std::unique_ptr<Tensor>(new Tensor);
      output_tensor_idxs_t_->Resize(output_tensor_dims_);
      output_tensor_idxs_data_ =
          output_tensor_idxs_t_->mutable_data<int, cl::Buffer>(TARGET(kOpenCL));
      TargetWrapperCL::MemcpySync(output_tensor_idxs_data_,
                                  output_tensor_idxs_vec.data(),
                                  output_tensor_idxs_t_->memory_size(),
                                  IoDirection::HtoD);
    }

    if (output_tensor_dims_.size() == 4) {
      output_tensor_n_ = output_tensor_dims_[0];
      output_tensor_c_ = output_tensor_dims_[1];
      output_tensor_h_ = output_tensor_dims_[2];
      output_tensor_w_ = output_tensor_dims_[3];
      x_tensor_w_ = x_tensor_dims_[3];
      x_tensor_h_ = x_tensor_dims_[2];
    } else if (output_tensor_dims_.size() == 3) {
      output_tensor_c_ = output_tensor_dims_[0];
      output_tensor_h_ = output_tensor_dims_[1];
      output_tensor_w_ = output_tensor_dims_[2];
      x_tensor_w_ = x_tensor_dims_[2];
      x_tensor_h_ = x_tensor_dims_[1];
    } else if (output_tensor_dims_.size() == 2) {
      output_tensor_h_ = output_tensor_dims_[0];
      output_tensor_w_ = output_tensor_dims_[1];
      x_tensor_w_ = x_tensor_dims_[1];
    }

    auto& context = ctx_->As<OpenCLContext>();
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(
        kernel_func_name_, kernel_path_, build_options_, time_stamp_);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  void GetGlobalWorkSize() {
    if (kernel_func_name_ == "transpose_general_buffer") {
      global_work_size_ =
          cl::NDRange{static_cast<cl::size_type>(output_tensor_h_),
                      static_cast<cl::size_type>(output_tensor_w_),
                      static_cast<cl::size_type>(output_tensor_c_)};
    } else {
      const std::vector<size_t>& ws =
          DefaultGlobalWorkSize(output_tensor_dims_,
                                DDim(std::vector<DDim::value_type>{
                                    static_cast<int64_t>(output_image_w_),
                                    static_cast<int64_t>(output_image_h_)}));
      global_work_size_ = cl::NDRange{static_cast<cl::size_type>(ws[0]),
                                      static_cast<cl::size_type>(ws[1]),
                                      static_cast<cl::size_type>(ws[2])};
    }
  }

  void Run() override {
    auto* x_image = GET_DATA_GPU(transpose_param_->x);
    auto* output_image = MUTABLE_DATA_GPU(
        transpose_param_->output, output_image_w_, output_image_h_, nullptr);

    auto& context = ctx_->As<OpenCLContext>();
    auto kernel = kernel_;
    cl_int status;
    if (kernel_func_name_ == "transpose_general_buffer") {
      // do image layout transform: image to buffer
      // create and set param, context to kernel im2buf
      operators::LayoutParam im2buf_param;
      std::shared_ptr<lite::Tensor> im2buf_out_t(new lite::Tensor);
      im2buf_out_t->Resize(x_tensor_dims_);
      auto im2buf_out_t_buffer_p =
          im2buf_out_t->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
      im2buf_param.x = transpose_param_->x;
      im2buf_param.y = im2buf_out_t.get();
      auto s = im2buf_kernel_->op_type();
      im2buf_kernel_->SetParam(im2buf_param);

      std::unique_ptr<KernelContext> im2buf_ctx(new KernelContext);
      context.CopySharedTo(&(im2buf_ctx->As<OpenCLContext>()));
      im2buf_kernel_->SetContext(std::move(im2buf_ctx));
      im2buf_kernel_->Launch();

      // create and set param, context to kernel buf2im
      std::shared_ptr<lite::Tensor> buf2im_in_t(new lite::Tensor);
      buf2im_in_t->Resize(transpose_param_->output->dims());
      auto buf2im_in_t_buffer_p =
          buf2im_in_t->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
      operators::LayoutParam buf2im_param;
      buf2im_param.x = buf2im_in_t.get();
      buf2im_param.y = transpose_param_->output;
      buf2im_kernel_->SetParam(buf2im_param);

      std::unique_ptr<KernelContext> buf2im_ctx(new KernelContext);
      context.CopySharedTo(&(buf2im_ctx->As<OpenCLContext>()));
      buf2im_kernel_->SetContext(std::move(buf2im_ctx));

      // set kernel args
      status = kernel.setArg(0, *im2buf_out_t_buffer_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(1, *buf2im_in_t_buffer_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(2, *output_tensor_idxs_data_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(3, output_tensor_c_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(4, output_tensor_h_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(5, output_tensor_w_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(6, output_tensor_h_ * output_tensor_w_);
      CL_CHECK_FATAL(status);

      GetGlobalWorkSize();
      auto& context = ctx_->As<OpenCLContext>();
      status = EnqueueNDRangeKernel(context,
                                    kernel,
                                    cl::NullRange,
                                    global_work_size_,
                                    cl::NullRange,
                                    nullptr,
                                    event_);
      CL_CHECK_FATAL(status);
      // run kernel: buffer->image
      buf2im_kernel_->Launch();
    } else {
      status = kernel.setArg(0, *x_image);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(1, *output_image);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(2, output_tensor_c_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(3, output_tensor_h_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(4, output_tensor_w_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(5, x_tensor_w_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(6, x_tensor_h_);
      CL_CHECK_FATAL(status);

      GetGlobalWorkSize();
      status = EnqueueNDRangeKernel(context,
                                    kernel,
                                    cl::NullRange,
                                    global_work_size_,
                                    cl::NullRange,
                                    nullptr,
                                    event_);
      CL_CHECK_FATAL(status);
    }
  }

 private:
  std::string kernel_func_name_{""};
  std::string kernel_path_{""};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};

  param_t* transpose_param_{nullptr};
  std::unique_ptr<Tensor> output_tensor_idxs_t_{nullptr};
  cl::Buffer* output_tensor_idxs_data_;

  std::vector<int> axis_;
  DDim x_tensor_dims_{};
  int x_tensor_w_{1};
  int x_tensor_h_{1};
  DDim output_tensor_dims_{};
  int output_tensor_n_{1};
  int output_tensor_c_{1};
  int output_tensor_h_{1};
  int output_tensor_w_{1};
  int output_image_h_{1};
  int output_image_w_{1};

  cl::NDRange global_work_size_;
  cl::Kernel kernel_;

  // transpose_general_buffer
  std::unique_ptr<KernelBase> im2buf_kernel_;
  std::unique_ptr<KernelBase> buf2im_kernel_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(transpose,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::TransposeComputeFloatImage,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(transpose2,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::TransposeComputeFloatImage,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

#define LITE_WITH_LOG
