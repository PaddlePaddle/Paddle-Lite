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
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

#undef LITE_WITH_LOG

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

// transpose operator
class TransposeComputeFloatImage
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::TransposeParam;

  void PrepareForRun() override {
    auto& param = *param_.get_mutable<param_t>();
    Tensor* const output = param.output;
    const DDimLite& out_dims = output->dims();
    if (out_dims.size() == 4) {
      kernel_func_name_ = "transpose_4d";
    } else {
      kernel_func_name_ = "transpose";
    }
    auto& context = ctx_->As<OpenCLContext>();
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/transpose_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const Tensor* const x = param.x;
    const auto x_dims = x->dims();
    const std::map<std::string, size_t>& input_image_shape =
        InitImageDimInfoWith(x_dims);
    const int64_t& input_image_width = input_image_shape.at("width");
    const int64_t& input_image_height = input_image_shape.at("height");
    const cl::Image2D* const x_image = x->data<half_t, cl::Image2D>();

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
    const std::vector<size_t>& default_work_size = DefaultGlobalWorkSize(
        out_dims,
        DDim(std::vector<DDim::value_type>{
            static_cast<int64_t>(out_image_shape.at("width")),
            static_cast<int64_t>(out_image_shape.at("height"))}));

    int out_C = 0, out_H = 0, out_W = 0, in_W = 0;
    if (param.output->dims().size() == 4) {
      out_C = out_dims[1];
      out_H = out_dims[2];
      out_W = out_dims[3];
      in_W = x_dims[3];
    } else if (param.output->dims().size() == 3) {
      out_C = out_dims[0];
      out_H = out_dims[1];
      out_W = out_dims[2];
      in_W = x_dims[2];
    } else if (param.output->dims().size() == 2) {
      out_C = 1;
      out_H = out_dims[0];
      out_W = out_dims[1];
      in_W = x_dims[1];
    }

#ifdef LITE_WITH_LOG
    VLOG(4) << "out_C=" << out_C;
    VLOG(4) << "out_H=" << out_H;
    VLOG(4) << "out_W=" << out_W;
    VLOG(4) << "in_W=" << in_W;
    VLOG(4) << "default_work_size= " << default_work_size[0] << ", "
            << default_work_size[1] << ", " << default_work_size[2];
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

    auto global_work_size =
        cl::NDRange{static_cast<size_t>(default_work_size.data()[0]),
                    static_cast<size_t>(default_work_size.data()[1]),
                    static_cast<size_t>(default_work_size.data()[2])};

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }

 private:
  std::string kernel_func_name_{"transpose"};
  std::string build_options_{"-DCL_DTYPE_half"};
  std::string time_stamp_{GetTimeStamp()};
};

// transpose2 operator
class Transpose2ComputeFloatImage
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::TransposeParam;

  void PrepareForRun() override {}

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {}
#endif

  bool IsShuffleChannel(const std::vector<int>& axis) {
    bool is_shuffle_channel = true;
    if (axis.size() > 2 && axis[0] == 0 && axis[1] == 2 && axis[2] == 1) {
      for (int i = 3; i < axis.size(); ++i) {
        if (axis[i] != i) {
          is_shuffle_channel = false;
          break;
        }
      }
    } else {
      return false;
    }
    return is_shuffle_channel;
  }

  template <typename Dtype>
  void DeviceTensorToHostTensor(const Tensor* device_tensor,
                                Tensor* host_tensor) {
    host_tensor->Resize(device_tensor->dims());
    Dtype* host_ptr = host_tensor->mutable_data<Dtype>();
    CLRuntime::Global()->command_queue().finish();
    CLImageConverterDefault default_converter;
    auto device_tensor_image_dim =
        default_converter.InitImageDimInfoWith(device_tensor->dims());
    half_t* image_data = new half_t[device_tensor_image_dim.production() * 4];
    TargetWrapperCL::ImgcpySync(image_data,
                                device_tensor->data<half_t, cl::Image2D>(),
                                device_tensor_image_dim[0],
                                device_tensor_image_dim[1],
                                0,
                                0,
                                IoDirection::DtoH);
    default_converter.ImageToNCHW(
        image_data, host_ptr, device_tensor_image_dim, host_tensor->dims());
    delete[] image_data;
  }

  template <typename Dtype>
  void HostTensorToDeviceTensor(const Tensor* host_tensor,
                                Tensor* device_tensor) {
    Dtype* host_ptr = const_cast<Dtype*>(host_tensor->data<Dtype>());
    CLImageConverterDefault default_converter;
    auto device_tensor_image_dim =
        default_converter.InitImageDimInfoWith(device_tensor->dims());
    device_tensor->mutable_data<half_t, cl::Image2D>(
        device_tensor_image_dim[0], device_tensor_image_dim[1]);
    half_t* image_data = new half_t[device_tensor->dims().production() * 4];
    default_converter.NCHWToImage(host_ptr, image_data, device_tensor->dims());

    TargetWrapperCL::ImgcpySync(
        device_tensor->mutable_data<half_t, cl::Image2D>(),
        image_data,
        device_tensor_image_dim[0],
        device_tensor_image_dim[1],
        0,
        0,
        IoDirection::HtoD);

    delete[] image_data;
  }

  template <typename Dtype>
  void ShuffleChannelCompute(const operators::TransposeParam& param) {
    const Tensor* input = param.x;
    Tensor* input_tensor = new Tensor();
    DeviceTensorToHostTensor<Dtype>(input, input_tensor);
    Dtype* input_ptr = input_tensor->mutable_data<Dtype>();

    Tensor* output = param.output;
    Tensor* output_tensor = new Tensor();
    output_tensor->Resize(output->dims());
    Dtype* output_ptr = output_tensor->mutable_data<Dtype>();

    // input and output's shape dimension must >= 2 && <= 6.
    const DDim& in_dim = input->dims();
    const DDim& out_dim = output->dims();
    size_t offset = 1;
    for (int i = 3; i < param.axis.size(); ++i) {
      offset *= in_dim[i];
    }
#pragma omp parallel for collapse(3)
    for (int batch = 0; batch < out_dim[0]; ++batch) {
      for (int c1 = 0; c1 < out_dim[1]; ++c1) {
        for (int c2 = 0; c2 < out_dim[2]; ++c2) {
          size_t out_offset =
              ((batch * out_dim[1] + c1) * out_dim[2] + c2) * offset;
          size_t in_offset =
              ((batch * in_dim[1] + c2) * in_dim[2] + c1) * offset;
          memcpy(output_ptr + out_offset,
                 input_ptr + in_offset,
                 offset * sizeof(Dtype));
        }
      }
    }
    HostTensorToDeviceTensor<Dtype>(output_tensor, output);
    delete input_tensor;
    delete output_tensor;
  }

  template <typename Dtype>
  void Transpose2Compute(const operators::TransposeParam& param) {
    const Tensor* input = param.x;
    Tensor* input_tensor = new Tensor();
    DeviceTensorToHostTensor<Dtype>(input, input_tensor);
    Dtype* input_ptr = input_tensor->mutable_data<Dtype>();

    Tensor* output = param.output;
    Tensor* output_tensor = new Tensor();
    output_tensor->Resize(output->dims());
    Dtype* output_ptr = output_tensor->mutable_data<Dtype>();

    // input and output's shape dimension must >= 2 && <= 6.
    const DDim& in_dim = input->dims();
    const DDim& out_dim = output->dims();

    // precompute inverted output dim and strides
    size_t rout_dim[6], strides[6];
    auto& axis = param.axis;
    int permute = axis.size();  // permute must >=2 && <= 6.
    for (int i = 0; i < permute; ++i) {
      int k = permute - 1 - i;
      strides[k] = 1;
      for (int j = axis[i] + 1; j < permute; ++j) {
        strides[k] *= in_dim[j];
      }
      rout_dim[k] = out_dim[i];
    }

    // unroll the first 2 dimensions
    int reamin_dim = 1;
    for (int i = 2; i < out_dim.size(); ++i) {
      reamin_dim *= out_dim[i];
    }

#pragma omp parallel for collapse(2)
    for (int batch = 0; batch < out_dim[0]; ++batch) {
      for (int j = 0; j < out_dim[1]; ++j) {
        size_t offset = batch * strides[permute - 1] + j * strides[permute - 2];
        Dtype* out_ptr = output_ptr + (batch * out_dim[1] + j) * reamin_dim;
        int indics[4] = {0, 0, 0, 0};
        for (int k = 0; k < reamin_dim; ++k) {
          out_ptr[k] = input_ptr[offset];
          indics[0] += 1;
          offset += strides[0];
          for (int p = 0; p < permute - 3; ++p) {
            if (indics[p] == rout_dim[p]) {
              indics[p + 1] += 1;
              indics[p] = 0;
              offset += strides[p + 1];
              offset -= rout_dim[p] * strides[p];
            } else {
              break;
            }
          }
        }
      }
    }
    HostTensorToDeviceTensor<Dtype>(output_tensor, output);
    delete input_tensor;
    delete output_tensor;
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const std::vector<int> axis = param.axis;

    bool shuffle_channel = IsShuffleChannel(axis);
    if (shuffle_channel) {
      ShuffleChannelCompute<float>(param);
    } else {
      Transpose2Compute<float>(param);
    }
  }
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
                     paddle::lite::kernels::opencl::Transpose2ComputeFloatImage,
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
