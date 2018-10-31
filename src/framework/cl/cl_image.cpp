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

namespace paddle_mobile {
namespace framework {

void CLImageToTensor(CLImage *cl_image, Tensor *tensor,
                     cl_command_queue commandQueue) {
  // TODO(yangfei): need imp
}

void TensorToCLImage(const Tensor *tensor, CLImage *cl_image,
                     cl_command_queue commandQueue) {
  // TODO(yangfei): need imp
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
