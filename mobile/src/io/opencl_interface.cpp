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
  return framework::CLEngine::Instance()->isInitSuccess();
}

}  // namespace paddle_mobile
#endif
