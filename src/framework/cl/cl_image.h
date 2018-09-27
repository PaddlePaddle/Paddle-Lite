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

namespace paddle_mobile {
namespace framework {

class CLImage {
 public:
  CLImage(cl_context context, float *tensorInput, DDim ddim) : tensorDims_(ddim), context_(context) {

  }

  const DDim &TensorDim();

 private:
  cl_mem cl_image_;
  DDim tensorDims_;
  cl_context context_;
};

void TensorToCLImage(Tensor *tensor, CLImage *image) {

}

void CLImageToTensor(CLImage *image, Tensor *tensor) {

}

}
}