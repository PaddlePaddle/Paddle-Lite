// Copyright (c) 2019 Cambricon Authors. All Rights Reserved.

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
      << (msg) << " MLU CNML: " << CnmlErrorInfo(static_cast<int>(msg))

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
