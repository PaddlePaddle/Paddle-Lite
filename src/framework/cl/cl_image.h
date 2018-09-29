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

namespace paddle_mobile {
namespace framework {

class CLImage {
 public:
  CLImage() = default;

  void Init(cl_context context, float *tensorInput, DDim ddim) {
  }

  void Init(cl_context context, DDim ddim) {

  }

  inline CLImage &Resize(const DDim &dims) {
    tensorDims_ = dims;
    return *this;
  }

  const DDim &dims() const {
    return DDim();
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