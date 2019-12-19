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
#ifdef EXPAND_OP

#include "operators/kernel/expand_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ExpandKernel<GPU_CL, float>::Init(ExpandParam<GPU_CL>* param) {
  const framework::DDim& input_dims = param->InputX()->dims();
  PADDLE_MOBILE_ENFORCE(input_dims.size() == 4,
                        "expend now support 4 size dims");
  if (input_dims[1] == 1) {
    this->cl_helper_.AddKernel("expend_c1", "expend.cl");
  } else if (input_dims[1] == 2) {
    this->cl_helper_.AddKernel("expend_c2", "expend.cl");
  } else if (input_dims[1] == 4) {
    this->cl_helper_.AddKernel("expend_c4", "expend.cl");
  } else {
    PADDLE_MOBILE_ENFORCE(false, "expend did not supported this type");
  }
  return true;
}

template <>
void ExpandKernel<GPU_CL, float>::Compute(const ExpandParam<GPU_CL>& param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  DLOG << "param.Out()->dims():  " << param.Out()->dims();
  const framework::DDim& image_dims = param.Out()->ImageDims();
  DLOG << "param.Out()->image_dims():  " << image_dims;

  auto out_work_size = this->cl_helper_.DefaultWorkSize(*param.Out());
  DLOG << "out_work_size:  " << out_work_size;

  int out_c_block = out_work_size[0];
  int out_w = out_work_size[1];
  int out_nh = out_work_size[2];

  auto in_work_size = this->cl_helper_.DefaultWorkSize(*param.InputX());
  int in_c_block = in_work_size[0];
  int in_w = in_work_size[1];
  int in_nh = in_work_size[2];

  int input_width = param.InputX()->dims()[3];
  int input_height = param.InputX()->dims()[2];
  int output_width = param.Out()->dims()[3];
  int output_height = param.Out()->dims()[2];

  const auto* input = param.InputX();
  auto* output = param.Out();
  vector<int> expandTimes = {1, 1, 1, 1};
  DLOG << "param.expand_times: " << param.expand_times;

  for (int i = 0; i < param.expand_times.size(); ++i) {
    expandTimes[i] = param.expand_times[i];
  }

  DLOG << "expandTimes: " << expandTimes;

  auto inputImage = input->GetCLImage();
  auto outputImage = output->GetCLImage();

  input->dims();

  int idx = 0;

  cl_int status;
  status = clSetKernelArg(kernel, idx++, sizeof(int), &out_c_block);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, idx++, sizeof(int), &out_w);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, idx++, sizeof(int), &out_nh);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, idx++, sizeof(int), &in_c_block);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, idx++, sizeof(int), &in_w);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, idx++, sizeof(int), &in_nh);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, idx++, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, idx++, sizeof(int), &input_height);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, idx++, sizeof(int), &output_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, idx++, sizeof(int), &output_height);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &inputImage);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &outputImage);
  CL_CHECK_ERRORS(status);

  status = clSetKernelArg(kernel, idx++, sizeof(int), &expandTimes[0]);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, idx++, sizeof(int), &expandTimes[1]);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, idx++, sizeof(int), &expandTimes[2]);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, idx++, sizeof(int), &expandTimes[3]);
  CL_CHECK_ERRORS(status);

  status =
      clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 3, NULL,
                             out_work_size.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);

  DLOG << *output;
}

template class ExpandKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile
#endif
