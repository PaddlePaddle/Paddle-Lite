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

#include "driver/device.h"

#define GENERIC_DEVICE_NAME generic_device

// The following environment variables can be used at runtime:
// Specify the number of threads to use in thread pool, no thread
// pool/single-thread is used as default(default value is 0).
#define GENERIC_DEVICE_NUM_THREADS "GENERIC_DEVICE_NUM_THREADS"

extern nnadapter::driver::Device NNADAPTER_AS_SYM2(GENERIC_DEVICE_NAME);
