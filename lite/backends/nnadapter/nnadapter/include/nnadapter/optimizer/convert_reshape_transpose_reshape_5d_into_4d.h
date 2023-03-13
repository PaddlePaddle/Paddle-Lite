// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "core/types.h"

namespace nnadapter {

/*
 * Convert 5D reshape-transpose-reshape into 4D.
 * For example:
 *        in -> shape[1,32,32,24]
 *         |
 *      reshape -> shape[1,2,16,32,24]
 *         |
 *      transpose -> shape[1,16,2,32,24]
 *         |
 *      reshape -> shape[1,32,32,24]
 *         |
 *        out
 *
 * After applied:
 *
 *        in -> shape[1,32,32,24]
 *         |
 *      reshape -> shape[1,2,16,32*24]
 *         |
 *      transpose -> shape[1,16,2,32*24]
 *         |
 *      reshape -> shape[1,32,32,24]
 *         |
 *        out
 *
 */

void ConvertReshapeTransposeReshape5DInto4D(core::Model* model);

}  // namespace nnadapter
