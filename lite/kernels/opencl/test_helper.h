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
// limitations under the License

#pragma once

#define COMPUTE_ABS_DIFF(res0, res1) abs(res0 - res1)

#define COMPUTE_RELATIVE_DIFF(res0, res1) abs(abs(res0 - res1) / (res1 + 1e-5))

#define IS_DIFF_PASSED(res0, res1, threshold)        \
  (((COMPTUE_ABS_DIFF(res0, res1) < threshold) ||    \
    (COMPUTE_RELATIVE_DIFF(res0, res1) < threshold)) \
       ? true                                        \
       : false)
