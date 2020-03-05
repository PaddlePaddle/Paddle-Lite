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
#include <cnml.h>
#include <cnrt.h>
#include <lite/utils/cp_logging.h>

/*
 * This file contains some MLU specific uitls.
 */

#define CNRT_CALL(msg)                                    \
  CHECK_EQ(static_cast<cnrtRet_t>(msg), CNRT_RET_SUCCESS) \
      << (msg)                                            \
      << " MLU CNRT: " << cnrtGetErrorStr(static_cast<cnrtRet_t>(msg))

#define CNML_CALL(msg)                                          \
  CHECK_EQ(static_cast<cnmlStatus_t>(msg), CNML_STATUS_SUCCESS) \
      << (msg) << " MLU CNML: "                                 \
      << ::paddle::lite::mlu::CnmlErrorInfo(static_cast<int>(msg))

namespace paddle {
namespace lite {
namespace mlu {

static const char* CnmlErrorInfo(int error) {
  switch (error) {
#define LITE_CNML_ERROR_INFO(xx) \
  case xx:                       \
    return #xx;                  \
    break;
    LITE_CNML_ERROR_INFO(CNML_STATUS_NODEVICE);
    LITE_CNML_ERROR_INFO(CNML_STATUS_SUCCESS);
    LITE_CNML_ERROR_INFO(CNML_STATUS_DOMAINERR);
    LITE_CNML_ERROR_INFO(CNML_STATUS_INVALIDARG);
    LITE_CNML_ERROR_INFO(CNML_STATUS_LENGTHERR);
    LITE_CNML_ERROR_INFO(CNML_STATUS_OUTOFRANGE);
    LITE_CNML_ERROR_INFO(CNML_STATUS_RANGEERR);
    LITE_CNML_ERROR_INFO(CNML_STATUS_OVERFLOWERR);
    LITE_CNML_ERROR_INFO(CNML_STATUS_UNDERFLOWERR);
    LITE_CNML_ERROR_INFO(CNML_STATUS_INVALIDPARAM);
    LITE_CNML_ERROR_INFO(CNML_STATUS_BADALLOC);
    LITE_CNML_ERROR_INFO(CNML_STATUS_BADTYPEID);
    LITE_CNML_ERROR_INFO(CNML_STATUS_BADCAST);
    LITE_CNML_ERROR_INFO(CNML_STATUS_UNSUPPORT);
#undef LITE_CNML_ERROR_INFO
    default:
      return "unknown error";
      break;
  }
}

}  // namespace mlu
}  // namespace lite
}  // namespace paddle
