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

struct CLKernelDeleter {
  template <class T>
  void operator()(T *clKernelObj) {
    clReleaseKernel(clKernelObj);
  }
};

struct CLMemDeleter {
  template <class T>
  void operator()(T *clMemObj) {
    clReleaseMemObject(clMemObj);
  }
};

struct CLEventDeleter {
  template <class T>
  void operator()(T *clEventObj) {
    clReleaseEvent(clEventObj);
  }
};

struct CLCommQueueDeleter {
  template <class T>
  void operator()(T *clQueueObj) {
    clReleaseCommandQueue(clQueueObj);
  }
};

struct CLContextDeleter {
  template <class T>
  void operator()(T *clContextObj) {
    clReleaseContext(clContextObj);
  }
};

struct CLProgramDeleter {
  template <class T>
  void operator()(T *clProgramObj) {
    clReleaseProgram(clProgramObj);
  }
};
