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

#if defined(LITE_WITH_ARM) || defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || \
    defined(LITE_WITH_PYTHON) || defined(LITE_WITH_XPU)
#include "lite/utils/log/logging.h"
#else
#ifndef LITE_WITH_LOG
#include "lite/utils/log/logging.h"
#else
#include <glog/logging.h>
#endif
#endif

// LOG System on opt tool:
//   OPT_LOG  :  print normal message onto the terminal screen.
//   OPT_LOG_ERROR : print error message onto the terminal screen.
//   OPT_LOG_FATAL : print error message onto the terminal screen and abort
//   current process.
//   OPT_LOG_DEBUG : print message if in debug mode (environmental val
//   GLOG_v=1).
// note: OPT_LOG_SYSTEM is only applicable when LITE_WITH_MODEL_OPTIMIZE_TOOL is
// defined,
//       otherwise, these commands will be replaced by corresponding glog
//       command
//       OPT_LOG--->LOG(INFO) , OPT_LOG_ERROR--->LOG(ERROR),
//       OPT_LOG_FATAL--->LOG(FATAL)

#if defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || defined(LITE_WITH_PYTHON)
// OPT_LOG SYSTEM
#define OPT_LOG paddle::lite::OptPrinter()
#define OPT_LOG_ERROR paddle::lite::OptErrorPrinter()
#define OPT_LOG_FATAL paddle::lite::OptFatalPrinter()
#define OPT_LOG_DEBUG VLOG(1)
#include <iostream>
namespace paddle {
namespace lite {

class OptPrinter {
 public:
  ~OptPrinter() { std::cout << std::endl; }
  template <typename T>
  OptPrinter& operator<<(const T& obj) {
    std::cout << obj;
    return *this;
  }
};
class OptErrorPrinter {
 public:
  virtual ~OptErrorPrinter() { std::cerr << std::endl; }
  template <typename T>
  OptErrorPrinter& operator<<(const T& obj) {
    std::cerr << obj;
    return *this;
  }
};
class OptFatalPrinter : public OptErrorPrinter {
 public:
  ~OptFatalPrinter() override {
    std::cerr << std::endl;
    abort();
  }
};
}  // namespace lite
}  // namespace paddle
#else
//  If LITE_WITH_MODEL_OPTIMIZE_TOOL is not defined, OPT_LOG commands
//  will be replaced by corresponding glog commands: OPT_LOG--->LOG(INFO) ,
//  OPT_LOG_ERROR--->LOG(ERROR), OPT_LOG_FATAL--->LOG(FATAL)
#define OPT_LOG LOG(INFO)
#define OPT_LOG_ERROR LOG(ERROR)
#define OPT_LOG_FATAL LOG(FATAL)
#define OPT_LOG_DEBUG VLOG(1)
#endif
