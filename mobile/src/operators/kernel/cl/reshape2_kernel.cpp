/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef RESHAPE2_OP

#include "operators/kernel/reshape2_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool Reshape2Kernel<GPU_CL, float>::Init(Reshape2Param<GPU_CL> *param) {
  this->cl_helper_.AddKernel("reshape", "reshape.cl");
  return true;
}

inline framework::DDim ValidateShape(const std::vector<int> shape,
                                     const framework::DDim &in_dims) {
  const int64_t in_size = framework::product(in_dims);
  // only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int64_t unk_dim_val = -1;
  const int64_t copy_dim_val = 0;

  std::vector<int64_t> output_shape(shape.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      PADDLE_MOBILE_ENFORCE(
          unk_dim_idx == -1,
          "Only one input dimension of Attr(shape) can be unknown.");
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      PADDLE_MOBILE_ENFORCE(
          static_cast<int>(i) < in_dims.size(),
          "The index of dimension to copy from input shape must be less "
          "than the size of input shape.");
    } else {
      PADDLE_MOBILE_ENFORCE(
          shape[i] > 0,
          "Each input dimension of Attr(shape) must not be negtive except "
          "one unknown dimension.");
    }

    capacity *= (shape[i] ? shape[i] : in_dims[i]);
    output_shape[i] = (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);
  }

  if (unk_dim_idx != -1) {
    output_shape[unk_dim_idx] = -in_size / capacity;
    PADDLE_MOBILE_ENFORCE(output_shape[unk_dim_idx] * capacity == -in_size,
                          "Invalid shape is given.");
  } else {
    PADDLE_MOBILE_ENFORCE(capacity == in_size, "Invalid shape is given.");
  }
  return framework::make_ddim(output_shape);
}

template <>
void Reshape2Kernel<GPU_CL, float>::Compute(
    const Reshape2Param<GPU_CL> &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.Out());
  const auto *input = param.InputX();
  auto *output = param.Out();
  auto input_image = input->GetCLImage();
  auto output_image = output->GetCLImage();
  const auto &inputDim = input->dims();
  const auto &outputDim = output->dims();
  int input_dims[4] = {1, 1, 1, 1};
  int output_dims[4] = {1, 1, 1, 1};
  // 1 1000 1 1
  for (int i = 0; i < inputDim.size(); i++) {
    input_dims[4 - inputDim.size() + i] = inputDim[i];
  }

  // 1 1 1 1000
  for (int i = 0; i < outputDim.size(); i++) {
    output_dims[4 - outputDim.size() + i] = outputDim[i];
  }

  int out_C = output_dims[1];
  int out_H = output_dims[2];
  int out_W = output_dims[3];
  int in_W = input_dims[3];
  int in_H = input_dims[2];
  int in_Stride0 = in_W;
  int in_Stride1 = input_dims[2] * input_dims[3];
  int in_Stride2 = input_dims[1] * input_dims[2] * input_dims[3];
  int out_Stride0 = out_W;
  int out_Stride1 = out_H * out_W;
  int out_Stride2 = out_C * out_H * out_W;
  DLOG << "out_C=" << out_C;
  DLOG << "out_H=" << out_H;
  DLOG << "out_W=" << out_W;
  DLOG << "in_W=" << in_W;
  DLOG << "default_work_size=" << default_work_size;
  DLOG << "in_Stride0=" << in_Stride0;
  DLOG << "in_Stride1=" << in_Stride1;
  DLOG << "out_Stride0=" << out_Stride0;
  DLOG << "out_Stride1=" << out_Stride1;
  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(int), &out_C);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(int), &out_H);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(int), &out_W);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(int), &in_W);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(int), &in_H);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(int), &in_Stride0);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 8, sizeof(int), &in_Stride1);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 9, sizeof(int), &in_Stride2);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 10, sizeof(int), &out_Stride0);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 11, sizeof(int), &out_Stride1);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 12, sizeof(int), &out_Stride2);
  CL_CHECK_ERRORS(status);
  status = clEnqueueNDRangeKernel(
      this->cl_helper_.CLCommandQueue(), kernel, default_work_size.size(), NULL,
      default_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}

template class Reshape2Kernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
#endif
