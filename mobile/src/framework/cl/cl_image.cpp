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

#include "framework/cl/cl_image.h"
#include "framework/cl/cl_tensor.h"

namespace paddle_mobile {
namespace framework {

void CLImageToTensor(CLImage *cl_image, Tensor *tensor, cl_context context,
                     cl_command_queue commandQueue, cl_kernel kernel) {
  tensor->mutable_data<float>();
  const auto &dim = cl_image->dims();
  size_t new_dims[] = {1, 1, 1, 1};
  for (int j = 0; j < dim.size(); ++j) {
    new_dims[4 - dim.size() + j] = dim[j];
  }
  size_t C, in_height, in_width;

  C = new_dims[1];
  in_height = new_dims[2];
  in_width = new_dims[3];

  CLTensor out_cl_tensor(context, commandQueue);
  out_cl_tensor.Resize(tensor->dims());
  cl_mem outBuffer = out_cl_tensor.mutable_data<float>();

  auto input_image = cl_image->GetCLImage();

  cl_int status;
  status = clSetKernelArg(kernel, 0, sizeof(int), &in_height);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(int), &in_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &outBuffer);
  CL_CHECK_ERRORS(status);
  int size_ch = in_height * in_width;
  int size_block = size_ch * 4;
  int size_batch = size_ch * C;
  status = clSetKernelArg(kernel, 4, sizeof(int), &size_ch);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(int), &size_block);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(int), &size_batch);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(int), &C);
  CL_CHECK_ERRORS(status);
  size_t global_work_size[3] = {(new_dims[1] + 3) / 4, new_dims[3],
                                new_dims[0] * new_dims[2]};
  status = clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL,
                                  global_work_size, NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
  memcpy(tensor->data<float>(), out_cl_tensor.Data<float>(),
         tensor->memory_size());
}

void TensorToCLImage(Tensor *tensor, CLImage *cl_image, cl_context context,
                     cl_command_queue commandQueue, cl_kernel kernel) {
  const auto &dim = cl_image->dims();
  size_t new_dims[] = {1, 1, 1, 1};
  for (int j = 0; j < dim.size(); ++j) {
    new_dims[4 - dim.size() + j] = dim[j];
  }
  cl_int status;
  auto output = cl_image;
  const Tensor *input = tensor;
  const float *input_data = input->data<float>();
  auto output_image = output->GetCLImage();
  const int out_C = new_dims[1];
  const int out_H = new_dims[2];
  const int out_W = new_dims[3];
  const int Stride2 = out_C * out_H * out_W;
  const int Stride1 = out_H * out_W;
  const int Stride0 = out_W;
  DLOG << out_C;
  DLOG << out_H;
  DLOG << out_W;
  CLTensor input_cl_tensor(context, commandQueue);
  input_cl_tensor.Resize(input->dims());
  cl_mem inputBuffer = input_cl_tensor.mutable_with_data<float>(input_data);

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_int), &out_H);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_int), &out_W);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_int), &out_C);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(cl_int), &Stride0);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(cl_int), &Stride1);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(cl_int), &Stride2);
  CL_CHECK_ERRORS(status);

  size_t global_work_size[3] = {(new_dims[1] + 3) / 4, new_dims[3],
                                new_dims[0] * new_dims[2]};
  status = clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL,
                                  global_work_size, NULL, 0, NULL, NULL);

  CL_CHECK_ERRORS(status);
}

#ifdef PADDLE_MOBILE_DEBUG
Print &operator<<(Print &printer, const CLImage &cl_image) {
  int width = cl_image.ImageDims()[0];
  int height = cl_image.ImageDims()[1];

  half_t *image_data = new half_t[height * width * 4];
  cl_int err;
  cl_mem image = cl_image.GetCLImage();
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {width, height, 1};
  err = clEnqueueReadImage(cl_image.CommandQueue(), image, CL_TRUE, origin,
                           region, 0, 0, image_data, 0, NULL, NULL);

  CL_CHECK_ERRORS(err);

  PADDLE_MOBILE_ENFORCE(cl_image.numel() != 0,
                        "cl_image numel should not be 0 ");
  float *tensor_data = new float[cl_image.numel()];
  auto converter = cl_image.Converter();
  converter->ImageToNCHW(image_data, tensor_data, cl_image.ImageDims(),
                         cl_image.dims());
  int stride = cl_image.numel() / 20;
  stride = stride > 0 ? stride : 1;

  printer << " dims: " << cl_image.dims() << "\n";
  for (int i = 0; i < cl_image.numel(); i += stride) {
    printer << tensor_data[i] << " ";
  }

  delete[](tensor_data);
  delete[](image_data);

  return printer;
}
#endif
}  // namespace framework
}  // namespace paddle_mobile
