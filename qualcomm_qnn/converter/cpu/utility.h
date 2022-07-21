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

namespace nnadapter {
namespace qualcomm_qnn {
namespace cpu {

#define QNN_CHECK_VALUE(value, return_error) \
  do {                                       \
    if (!(value)) {                          \
      return return_error;                   \
    }                                        \
  } while (0);

#define QNN_CHECK_STATUS(status)   \
  do {                             \
    if ((status) != QNN_SUCCESS) { \
      return status;               \
    }                              \
  } while (0);

#define QNN_CHECK_EQ(a, b, return_error) \
  do {                                   \
    if ((a) != (b)) {                    \
      return return_error;               \
    }                                    \
  } while (0);

}  // namespace cpu
}  // namespace qualcomm_qnn
}  // namespace nnadapter
