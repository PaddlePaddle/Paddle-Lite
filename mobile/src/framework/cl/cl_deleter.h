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

#include "CL/cl.h"
#include "common/log.h"
struct CLKernelDeleter {
  template <class T>
  void operator()(T *clKernelObj) {
    const cl_int status = clReleaseKernel(clKernelObj);
    LOG(paddle_mobile::kNO_LOG) << "clReleaseKernel  status:     " << status;
  }
};

struct CLMemDeleter {
  template <class T>
  void operator()(T *clMemObj) {
    const cl_int status = clReleaseMemObject(clMemObj);
    LOG(paddle_mobile::kNO_LOG) << "CLMemDeleter  status:     " << status;
  }
};

struct CLEventDeleter {
  template <class T>
  void operator()(T *clEventObj) {
    const cl_int status = clReleaseEvent(clEventObj);
    LOG(paddle_mobile::kNO_LOG) << "CLEventDeleter  status:     " << status;
  }
};

struct CLCommQueueDeleter {
  template <class T>
  void operator()(T *clQueueObj) {
    const cl_int status = clReleaseCommandQueue(clQueueObj);
    LOG(paddle_mobile::kNO_LOG) << "CLCommQueueDeleter  status:     " << status;
  }
};

struct CLContextDeleter {
  template <class T>
  void operator()(T *clContextObj) {
    const cl_int status = clReleaseContext(clContextObj);
    LOG(paddle_mobile::kNO_LOG) << "CLContextDeleter  status:     " << status;
  }
};

struct CLProgramDeleter {
  template <class T>
  void operator()(T *clProgramObj) {
    const cl_int status = clReleaseProgram(clProgramObj);
    LOG(paddle_mobile::kNO_LOG) << "CLProgramDeleter  status:   " << status;
  }
};
