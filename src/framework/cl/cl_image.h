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

#pragma once

#include "framework/ddim.h"
#include "framework/tensor.h"
#include "CL/cl.h"
#include "cl_half.h"

namespace paddle_mobile {
namespace framework {

class CLImage {
 public:
  CLImage() = default;

  void Init(cl_context context, float *tensorInput, DDim ddim) {
    tensorDims_ = ddim;
    cl_image_format cf = {
      .image_channel_order = CL_RGBA,
      .image_channel_data_type = CL_HALF_FLOAT
    };
    // NCHW -> [W * (C+3)/4, H * N]
    DLOG<<tensorDims_;
      size_t N,C,H,W;
    if(tensorDims_.size()==4){
        N = tensorDims_[0];
        if(N<0){
            N = 1;
        }
        C = tensorDims_[1];
        H = tensorDims_[2];
        W = tensorDims_[3];
    }else if(tensorDims_.size()==1){
        N = 1;
        C = tensorDims_[0];
        H = 1;
        W = 1;
    }

      DLOG<<"-------InitMemory-------";
    size_t width = W * ((C + 3) / 4);
    size_t height = H * N;
    std::unique_ptr<half_t[]> imageData{};
      int count = 0;
    if (tensorInput != nullptr) {
      imageData.reset(new half_t[width * height * 4]);
                  float *p = tensorInput;
                  size_t i0 = 0;
                  for (int n = 0; n < N; n++) {
                    for (int c = 0; c < C; c++) {
                      size_t i1 = i0;
                      for (int h = 0; h < H; h++) {
                        size_t i2 = (i1<<2) + c % 4;
                        for (int w = 0; w < W; w++) {
                            if (i2 >= width * height * 4) {
                                printf("%d > %d ----> %d, %d, %d, %d --- %d, %d, %d\n", i2, width*height*4, n, c, h, w, i0, i1, i2);
                            }
                            assert(i2 < width * height * 4);

                            imageData[i2] = float2half(*p);
                          i2 += 4;
                          p++;
            //              count++;
            //              DLOG<<count;
                        }
                        i1 += width;
                      }
                    }
                    i0 += width * H;
                  }
    }
      DLOG<<"-------InitMemory-------";
    cl_int err;
    cl_image_ = clCreateImage2D(
      context, // cl_context context
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, // cl_mem_flags flags
      &cf, // const cl_image_format *image_format
      width, // size_t image_width
      height, // size_t image_height
      0, // size_t image_row_pitch
      reinterpret_cast<void*>(imageData.get()), // void *host_ptr
      &err // cl_int *errcode_ret
    );
    if (err != CL_SUCCESS) {
      // TODO: error handling
    }
  }

  void Init(cl_context context, DDim ddim) {
    Init(context, nullptr, ddim);
  }

  inline CLImage &Resize(const DDim &dims) {
    tensorDims_ = dims;
    return *this;
  }

  const DDim &dims() const {
    return tensorDims_;
  }

  std::vector<size_t> DefaultWorkSize() {
    return {};
  }

  cl_mem GetCLImage() {
    return cl_image_;
  }

 private:
  bool initialized_ = false;
  cl_mem cl_image_;
  DDim tensorDims_;
  cl_context context_;
};

//void TensorToCLImage(Tensor *tensor, CLImage *image) {
//
//}
//
//void CLImageToTensor(CLImage *image, Tensor *tensor) {
//
//}

}
}