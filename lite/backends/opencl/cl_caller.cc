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

#include "lite/backends/opencl/cl_caller.h"
#include <string>
#include "lite/backends/opencl/cl_context.h"
#include "lite/backends/opencl/cl_image.h"
#include "lite/backends/opencl/cl_runtime.h"
#include "lite/backends/opencl/cl_utility.h"
#include "lite/core/tensor.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

static void CopyImageData(CLContext* context,
                          const CLImage& cl_image,
                          float* out) {
  int width = cl_image.image_dims()[0];
  int height = cl_image.image_dims()[1];

  uint16_t* image_data = new uint16_t[height * width * 4];
  cl::Image* image = cl_image.cl_image();
  cl::array<size_t, 3> origin = {0, 0, 0};
  cl::array<size_t, 3> region = {
      static_cast<size_t>(width), static_cast<size_t>(height), 1};
  cl_int err = context->GetCommandQueue().enqueueReadImage(
      *image, CL_TRUE, origin, region, 0, 0, image_data, nullptr, nullptr);
  CL_CHECK_FATAL_SOLID(err);

  auto* converter = cl_image.image_converter();
  converter->ImageToNCHW(
      image_data, out, cl_image.image_dims(), cl_image.tensor_dims());

  delete[] image_data;
}

bool InitOpenCLRuntime() {
  auto* runtime = CLRuntime::Global();
  return runtime->IsInitSuccess();
}

}  // namespace lite
}  // namespace paddle
