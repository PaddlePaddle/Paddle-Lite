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

#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "lite/utils/any.h"
#include "lite/utils/check.h"
#include "lite/utils/hash.h"
#include "lite/utils/io.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/macros.h"
#include "lite/utils/string.h"
#include "lite/utils/varient.h"

#ifdef LITE_ON_TINY_PUBLISH
#include "lite/utils/replace_stl/stream.h"
#endif

// On windows environment, min and max will be undefined to
// avoid compiling error.
#if defined(_MSC_VER)
#undef min
#undef max
#endif
