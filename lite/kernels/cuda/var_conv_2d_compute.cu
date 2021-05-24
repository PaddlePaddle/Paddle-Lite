/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <functional>
#include <memory>
#include <vector>

#include "lite/backends/cuda/math/gemm.h"
#include "lite/backends/cuda/math/type_trans.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/tensor.h"
#include "lite/kernels/cuda/var_conv_2d_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

inline int ConvOutputSize(int input_size,
                          int filter_size,
                          int dilation,
                          int pad_left,
                          int pad_right,
                          int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size =
      (input_size + (pad_left + pad_right) - dkernel) / stride + 1;

  return output_size;
}

// Eliminate the effects of pad, support batch > 1.
template <typename dtype>
__global__ void eliminate_pad_effect(dtype* src,
                                     const int64_t* offset,
                                     const int num_batch,
                                     const int batch_stride,
                                     const int num_channel,
                                     const int channel_stride,
                                     const int num_height,
                                     const int height_stride,
                                     const int num_width,
                                     const int width_stride,
                                     const int count) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int thread_num = blockDim.x * gridDim.x;
  for (tid = threadIdx.x + blockIdx.x * blockDim.x; tid < count;
       tid += thread_num) {
    int batch_id = tid / batch_stride;
    int width_id = tid % num_width;
    int cur_len = offset[batch_id + 1] - offset[batch_id];
    if (width_id >= cur_len) {
      src[tid] = 0.f;
    }
  }
}

template <typename dtype>
__global__ void ReorganizeOutput(const dtype* src,
                                 dtype* dst,
                                 const int dst_count,
                                 const int channel,
                                 const int src_height,
                                 const int src_width,
                                 const int dst_height,
                                 const int dst_width) {
  CUDA_KERNEL_LOOP(tid, dst_count) {
    int w_id = tid % dst_width;
    int h_id = tid / dst_width % dst_height;
    int c_id = tid / dst_height / dst_width % channel;
    if (h_id < src_height && w_id < src_width) {
      dst[tid] = src[c_id * src_height * src_width + h_id * src_width + w_id];
    } else {
      dst[tid] = 0.f;
    }
  }
}

template <typename T, PrecisionType PType>
void VarConv2DCompute<T, PType>::PrepareForRun() {
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();
  auto& param = this->template Param<param_t>();

  conv_param_.paddings.reset(new std::vector<int>);
  conv_param_.paddings->push_back(static_cast<int>(param.kernel_h / 2));
  conv_param_.paddings->push_back(static_cast<int>(param.kernel_h / 2));
  conv_param_.paddings->push_back(static_cast<int>(param.kernel_w / 2));
  conv_param_.paddings->push_back(static_cast<int>(param.kernel_w / 2));
  conv_param_.dilations.reset(new std::vector<int>);
  conv_param_.dilations->push_back(1);
  conv_param_.dilations->push_back(1);
  conv_param_.strides[0] = param.stride_h;
  conv_param_.strides[1] = param.stride_w;
  conv_param_.filter = const_cast<lite::Tensor*>(param.W);
  conv_param_.filter->Resize({param.output_channel,
                              param.input_channel,
                              param.kernel_h,
                              param.kernel_w});
  conv_param_.var_length = true;
  if (param.fuse_relu) {
    conv_param_.activation_param.has_active = true;
    conv_param_.activation_param.active_type = lite_api::ActivationType::kRelu;
  }

  auto lod = param.X->lod();
  if (lod.size() == 1) {
    conv_param_.x = const_cast<lite::Tensor*>(param.X);
    conv_param_.output = param.Out;

    std::vector<int64_t> output_shape(
        {conv_param_.x->dims()[0], param.output_channel});
    for (size_t i = 0; i < conv_param_.strides.size(); ++i) {
      output_shape.push_back(
          ConvOutputSize(conv_param_.x->dims()[i + 2],
                         conv_param_.filter->dims()[i + 2],
                         (*conv_param_.dilations.get())[i],
                         (*conv_param_.paddings.get())[i * 2],
                         (*conv_param_.paddings.get())[i * 2 + 1],
                         conv_param_.strides[i]));
    }
    conv_param_.output->Resize({output_shape});
    conv_impl_.reset(new lite::cuda::math::CudnnConv2D<T, PType>);
    conv_impl_->init(conv_param_, &context);
  } else if (lod.size() == 3) {
    auto height_lod = lod[0];
    auto width_lod = lod[1];
    int cur_h = height_lod[1] - height_lod[0];
    int cur_w = width_lod[1] - width_lod[0];
    int out_cur_h = ConvOutputSize(cur_h,
                                   conv_param_.filter->dims()[2],
                                   (*conv_param_.dilations.get())[0],
                                   (*conv_param_.paddings.get())[0],
                                   (*conv_param_.paddings.get())[1],
                                   conv_param_.strides[0]);
    int out_cur_w = ConvOutputSize(cur_w,
                                   conv_param_.filter->dims()[3],
                                   (*conv_param_.dilations.get())[1],
                                   (*conv_param_.paddings.get())[2],
                                   (*conv_param_.paddings.get())[3],
                                   conv_param_.strides[1]);
    std::vector<int64_t> output_shape({1, param.output_channel});
    output_shape.push_back(out_cur_h);
    output_shape.push_back(out_cur_w);

    auto* in = const_cast<lite::Tensor*>(param.X);
    lite::Tensor t =
        in->template Slice<T>(0, cur_h * cur_w * param.input_channel);
    t.Resize({1, param.input_channel, cur_h, cur_w});

    conv_param_.x = &t;
    conv_param_.output = param.Col;
    conv_param_.output->Resize({output_shape});
    conv_impl_.reset(new lite::cuda::math::CudnnConv2D<T, PType>);
    conv_impl_->init(conv_param_, &context);
  }
}

template <typename T, PrecisionType PType>
void VarConv2DCompute<T, PType>::Run() {
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();
  auto& param = this->template Param<param_t>();

  auto lod = param.X->lod();
  if (lod.size() == 1) {
    conv_param_.x = const_cast<lite::Tensor*>(param.X);
    param.Out->set_lod(param.X->lod());
    std::vector<int64_t> output_shape(
        {conv_param_.x->dims()[0], param.output_channel});
    for (size_t i = 0; i < conv_param_.strides.size(); ++i) {
      output_shape.push_back(
          ConvOutputSize(conv_param_.x->dims()[i + 2],
                         conv_param_.filter->dims()[i + 2],
                         (*conv_param_.dilations.get())[i],
                         (*conv_param_.paddings.get())[i * 2],
                         (*conv_param_.paddings.get())[i * 2 + 1],
                         conv_param_.strides[i]));
    }
    conv_param_.output = param.Out;
    conv_param_.output->Resize({output_shape});
    conv_impl_->create(conv_param_, &context);
    conv_impl_->run(conv_param_);

    // Avoid situations where cascading conv does not support multiple batch
    // calculations
    T* out_data = param.Out->template mutable_data<T>();
    const int batch_num = output_shape[1] * output_shape[2] * output_shape[3];
    std::vector<int64_t> lod(param.X->lod()[0].size(), 0);
    for (size_t i = 0; i < param.X->lod()[0].size(); ++i) {
      lod[i] = param.X->lod()[0][i];
    }
    int count = std::accumulate(
        output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    int width_stride = 1;
    int height_stride = output_shape[3];
    int channel_stride = output_shape[2] * output_shape[3];
    int batch_stride = output_shape[1] * output_shape[2] * output_shape[3];
    int threads = 512;
    int blocks = (count + threads - 1) / threads;

    offset_.Resize({static_cast<int64_t>(lod.size())});
    int64_t* d_offset = offset_.mutable_data<int64_t>(TARGET(kCUDA));
    TargetWrapperCuda::MemcpyAsync(d_offset,
                                   lod.data(),
                                   sizeof(int64_t) * lod.size(),
                                   IoDirection::HtoD,
                                   stream);

    eliminate_pad_effect<T><<<blocks, threads, 0, stream>>>(out_data,
                                                            d_offset,
                                                            output_shape[0],
                                                            batch_stride,
                                                            output_shape[1],
                                                            channel_stride,
                                                            output_shape[2],
                                                            height_stride,
                                                            output_shape[3],
                                                            width_stride,
                                                            count);
  } else if (lod.size() == 3) {
    auto height_lod = lod[0];
    auto width_lod = lod[1];
    int batch = height_lod.size() - 1;
    int max_h = 0;
    int max_w = 0;
    std::vector<uint64_t> out_lod(lod[0].size(), 0);
    for (size_t i = 0; i < batch; ++i) {
      int cur_h = height_lod[i + 1] - height_lod[i];
      int cur_w = width_lod[i + 1] - width_lod[i];
      max_h = cur_h > max_h ? cur_h : max_h;
      max_w = cur_w > max_w ? cur_w : max_w;
      out_lod[i + 1] = out_lod[i] + cur_h * cur_w * param.output_channel;
    }
    param.Out->set_lod({out_lod});

    int out_max_h = ConvOutputSize(max_h,
                                   conv_param_.filter->dims()[2],
                                   (*conv_param_.dilations.get())[0],
                                   (*conv_param_.paddings.get())[0],
                                   (*conv_param_.paddings.get())[1],
                                   conv_param_.strides[0]);
    int out_max_w = ConvOutputSize(max_w,
                                   conv_param_.filter->dims()[3],
                                   (*conv_param_.dilations.get())[1],
                                   (*conv_param_.paddings.get())[2],
                                   (*conv_param_.paddings.get())[3],
                                   conv_param_.strides[1]);

    std::vector<int64_t> output_shape({batch, param.output_channel});
    output_shape.push_back(out_max_h);
    output_shape.push_back(out_max_w);
    param.Out->Resize({output_shape});
    param.Col->Resize({1, param.output_channel, out_max_h, out_max_w});
    param.Col->template mutable_data<T>(TARGET(kCUDA));
    int64_t num = 0;
    for (size_t i = 0; i < batch; ++i) {
      int len_h = height_lod[i + 1] - height_lod[i];
      int len_w = width_lod[i + 1] - width_lod[i];

      std::vector<int64_t> output_shape({1, param.output_channel});
      int out_len_h = ConvOutputSize(len_h,
                                     conv_param_.filter->dims()[2],
                                     (*conv_param_.dilations.get())[0],
                                     (*conv_param_.paddings.get())[0],
                                     (*conv_param_.paddings.get())[1],
                                     conv_param_.strides[0]);
      int out_len_w = ConvOutputSize(len_w,
                                     conv_param_.filter->dims()[3],
                                     (*conv_param_.dilations.get())[1],
                                     (*conv_param_.paddings.get())[2],
                                     (*conv_param_.paddings.get())[3],
                                     conv_param_.strides[1]);
      output_shape.push_back(out_len_h);
      output_shape.push_back(out_len_w);

      auto* in = const_cast<lite::Tensor*>(param.X);
      lite::Tensor t = in->template Slice<T>(
          num, int64_t(num + len_h * len_w * param.input_channel));
      t.Resize({1, param.input_channel, len_h, len_w});
      conv_param_.x = &t;
      conv_param_.output = param.Col;
      conv_param_.output->Resize({output_shape});
      num += len_h * len_w * param.input_channel;
      conv_impl_->create(conv_param_, &context);
      conv_impl_->run(conv_param_);

      param.Out->Resize({batch, param.output_channel, out_max_h, out_max_w});
      ReorganizeOutput<<<CUDA_GET_BLOCKS(out_max_h * out_max_w *
                                         param.output_channel),
                         CUDA_NUM_THREADS,
                         0,
                         stream>>>(
          conv_param_.output->data<T>(),
          param.Out->template mutable_data<T>(TARGET(kCUDA)) +
              i * param.output_channel * out_max_h * out_max_w,
          out_max_h * out_max_w * param.output_channel,
          param.output_channel,
          output_shape[2],
          output_shape[3],
          out_max_h,
          out_max_w);
    }
  } else {
    LOG(FATAL) << "Not supported lod size in var_conv_2d. we only support 1 "
                  "and 3, but got "
               << lod.size();
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(ERROR) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using VarConvFp32 =
    paddle::lite::kernels::cuda::VarConv2DCompute<float, PRECISION(kFloat)>;
using VarConvFp16 =
    paddle::lite::kernels::cuda::VarConv2DCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(var_conv_2d, kCUDA, kFloat, kNCHW, VarConvFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("COLUMN", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("ROW", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Col", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(var_conv_2d, kCUDA, kFP16, kNCHW, VarConvFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("COLUMN",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("ROW", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Col", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
