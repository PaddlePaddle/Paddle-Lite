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

#include "cl_image.h"
namespace paddle_mobile {
namespace framework {
void CLImageToTensor(CLImage *cl_image, Tensor *tensor,
                     cl_command_queue commandQueue) {
  DDim ddim = cl_image->dims();
  size_t N, C, H, W;
  if (ddim.size() == 4) {
    N = ddim[0];
    if (N < 0) {
      N = 1;
    }
    C = ddim[1];
    H = ddim[2];
    W = ddim[3];
  } else if (ddim.size() == 1) {
    N = 1;
    C = ddim[0];
    H = 1;
    W = 1;
  }

  size_t width = W * ((C + 3) / 4);
  size_t height = H * N;

  float *p = tensor->mutable_data<float>();
  half imageData[width * height * 4];
  cl_int err;
  cl_mem image = cl_image->GetCLImage();
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {width, height, 1};
  err = clEnqueueReadImage(commandQueue, image, CL_TRUE, origin, region, 0, 0,
                           imageData, 0, NULL, NULL);
  size_t i0 = 0;
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      size_t i1 = i0;
      for (int h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (int w = 0; w < W; w++) {
          *p = Half2Float(imageData[i2]);
          i2 += 4;
          p++;
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }

  if (err != CL_SUCCESS) {
    // TODO: error handling
  }
}
void TensorToCLImage(const Tensor *tensor, CLImage *cl_image,
                     cl_command_queue commandQueue) {
  DDim ddim = cl_image->dims();
  size_t N, C, H, W;
  if (ddim.size() == 4) {
    N = ddim[0];
    if (N < 0) {
      N = 1;
    }
    C = ddim[1];
    H = ddim[2];
    W = ddim[3];
  } else if (ddim.size() == 1) {
    N = 1;
    C = ddim[0];
    H = 1;
    W = 1;
  }

  size_t width = W * ((C + 3) / 4);
  size_t height = H * N;

  const float *p = tensor->data<float>();
  half imageData[width * height * 4];
  cl_mem image = cl_image->GetCLImage();
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {width, height, 1};
  cl_int err;
  err = clEnqueueReadImage(commandQueue, image, CL_TRUE, origin, region, 0, 0,
                           imageData, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    // TODO: error handling
  }
  size_t i0 = 0;
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      size_t i1 = i0;
      for (int h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (int w = 0; w < W; w++) {
          imageData[i2] = Float2Half(*p);
          i2 += 4;
          p++;
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }
}
#ifdef PADDLE_MOBILE_DEBUG
Print &operator<<(Print &printer, const CLImage &cl_image){
  printer << " dims: " << cl_image.dims() << "\n";
  int stride = cl_image.numel() / 20;
  stride = stride > 0 ? stride : 1;
  float *data = new float[cl_image.numel()];
  DDim ddim = cl_image.dims();
  size_t N, C, H, W;
  if (ddim.size() == 4) {
    N = ddim[0];
    if (N < 0) {
      N = 1;
    }
    C = ddim[1];
    H = ddim[2];
    W = ddim[3];
  } else if (ddim.size() == 1) {
    N = 1;
    C = ddim[0];
    H = 1;
    W = 1;
  }

  size_t width = W * ((C + 3) / 4);
  size_t height = H * N;

  float *p = data;
  half imageData[width * height * 4];
  cl_int err;
  cl_mem image = cl_image.GetCLImage();
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {width, height, 1};
  err = clEnqueueReadImage(cl_image.CommandQueue(), image, CL_TRUE, origin, region, 0, 0,
                           imageData, 0, NULL, NULL);
  size_t i0 = 0;
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      size_t i1 = i0;
      for (int h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (int w = 0; w < W; w++) {
          *p = Half2Float(imageData[i2]);
          i2 += 4;
          p++;
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }

  if (err != CL_SUCCESS) {
    // TODO: error handling
  }
    for (int i = 0; i < cl_image.numel(); i += stride) {
            printer << data[i] << " ";
    }
  return printer;
        }
#endif
}  // namespace framework
}  // namespace paddle_mobile
