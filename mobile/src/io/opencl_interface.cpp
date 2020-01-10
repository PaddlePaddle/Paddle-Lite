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
#ifdef PADDLE_MOBILE_CL

#include "io/opencl_interface.h"
#include "framework/cl/cl_engine.h"
#include "framework/cl/cl_scope.h"

namespace paddle_mobile {

cl_context getContext() {
  return framework::CLEngine::Instance()->getContext();
}

cl_command_queue getClCommandQueue() {
  return framework::CLEngine::Instance()->getClCommandQueue();
}

bool isInitSuccess() {
  prepareOpenclRuntime();
  return framework::CLEngine::Instance()->isInitSuccess();
}

bool prepareOpenclRuntime() {
#ifdef PREPARE_OPENCL_RUNTIME
  DLOG << "cl runtime prepared. ";
  cl_uint numPlatforms;  // the NO. of platforms
  cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (status == CL_SUCCESS) {
    if (numPlatforms > 0) {
      cl_platform_id *platforms = reinterpret_cast<cl_platform_id *>(
          malloc(numPlatforms * sizeof(cl_platform_id)));
      status = clGetPlatformIDs(numPlatforms, platforms, NULL);
      free(platforms);
    }
  }
#endif
  return true;
}

}  // namespace paddle_mobile
#endif
