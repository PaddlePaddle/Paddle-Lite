// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#if defined _WIN32 || defined __CYGWIN__
#define NNADAPTER_EXPORT __declspec(dllexport)
#else
#if __GNUC__ >= 4
#define NNADAPTER_EXPORT __attribute__((visibility("default")))
#else
#define NNADAPTER_EXPORT
#endif
#endif

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(class__) \
  class__(const class__&) = delete;       \
  class__& operator=(const class__&) = delete;
#endif

#define NNADAPTER_AS_STR(x) #x
// Add prefix to the module symbol of each of NNAdapter device HAL library, such
// as "__nnadapter_driver__rockchip_npu"
#define NNADAPTER_AS_SYM(x) __nnadapter_device__##x
#define NNADAPTER_AS_STR2(x) NNADAPTER_AS_STR(x)
#define NNADAPTER_AS_SYM2(x) NNADAPTER_AS_SYM(x)
