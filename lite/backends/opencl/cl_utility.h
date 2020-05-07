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
}  // namespace lite
}  // namespace paddle
