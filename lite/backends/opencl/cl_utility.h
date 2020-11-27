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

#include "lite/backends/opencl/cl_include.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

const char* opencl_error_to_str(cl_int error);

#define CL_CHECK_ERROR(err_code__)                                   \
  if (err_code__ != CL_SUCCESS) {                                    \
    LOG(ERROR) << string_format(                                     \
        "OpenCL error with code %s happened in file %s at line %d. " \
        "Exiting.\n",                                                \
        opencl_error_to_str(err_code__),                             \
        __FILE__,                                                    \
        __LINE__);                                                   \
  }

// CL_CHECK_FATAL, not hurt performance , must check
#define CL_CHECK_FATAL_SOLID(err_code__)                             \
  if (err_code__ != CL_SUCCESS) {                                    \
    LOG(FATAL) << string_format(                                     \
        "OpenCL error with code %s happened in file %s at line %d. " \
        "Exiting.\n",                                                \
        opencl_error_to_str(err_code__),                             \
        __FILE__,                                                    \
        __LINE__);                                                   \
  }

// CL_CHECK_FATAL, hurt performance ,will shutdown check when relase
#ifdef LITE_WITH_LOG
#define CL_CHECK_FATAL(err_code__)                                   \
  if (err_code__ != CL_SUCCESS) {                                    \
    LOG(FATAL) << string_format(                                     \
        "OpenCL error with code %s happened in file %s at line %d. " \
        "Exiting.\n",                                                \
        opencl_error_to_str(err_code__),                             \
        __FILE__,                                                    \
        __LINE__);                                                   \
  }
#else
#define CL_CHECK_FATAL(err_code__)
#endif

#define EnqueueNDRangeKernel(                                      \
    context, kernel, gws_offset, gws, lws, event_wait_list, event) \
  context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(    \
      kernel, gws_offset, gws, lws, event_wait_list, &event)

// mutable_data
#define MUTABLE_DATA_GPU(tensor_instance_p, img_w, img_h, ptr)     \
  (fp16_support_)                                                  \
      ? (tensor_instance_p)                                        \
            ->mutable_data<half_t, cl::Image2D>(img_w, img_h, ptr) \
      : (tensor_instance_p)                                        \
            ->mutable_data<float, cl::Image2D>(img_w, img_h, ptr)

#define DATA_GPU(tensor_instance_p)                                          \
  (fp16_support_) ? (tensor_instance_p)->mutable_data<half_t, cl::Image2D>() \
                  : (tensor_instance_p)->mutable_data<float, cl::Image2D>()

#define GET_DATA_GPU(tensor_instance_p)                              \
  (fp16_support_) ? (tensor_instance_p)->data<half_t, cl::Image2D>() \
                  : (tensor_instance_p)->data<float, cl::Image2D>()

#define MUTABLE_DATA_CPU(tensor_instance_p)                             \
  (fp16_support_)                                                       \
      ? static_cast<void*>((tensor_instance_p)->mutable_data<half_t>()) \
      : static_cast<void*>((tensor_instance_p)->mutable_data<float>())

}  // namespace lite
}  // namespace paddle
