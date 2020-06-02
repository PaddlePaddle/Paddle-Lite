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

// Use internal log or glog, the priority is as follows:
// 1. tiny_publish should use internally implemented logging.
// 2. if LITE_WITH_LOG is turned off, internal logging is used.
// 3. use glog in other cases.

#if defined(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK) || \
    defined(LITE_ON_MODEL_OPTIMIZE_TOOL)
#include "lite/utils/logging.h"
#else
#ifndef LITE_WITH_LOG
#include "lite/utils/logging.h"
#else
#include <glog/logging.h>
#endif
#endif
