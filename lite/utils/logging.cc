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

/*
 * This file implements an lightweight alternative for glog, which is more
 * friendly for mobile.
 */

#include "lite/utils/logging.h"
#include <iomanip>

#if defined(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK) || \
    defined(LITE_ON_MODEL_OPTIMIZE_TOOL)
#ifdef LITE_WITH_LOG

namespace paddle {
namespace lite {

void gen_log(STL::ostream& log_stream_,
             const char* file,
             const char* func,
             int lineno,
             const char* level,
             const int kMaxLen) {
  const int len = strlen(file);

  std::string time_str;
  struct tm tm_time;  // Time of creation of LogMessage
  time_t timestamp = time(NULL);
  localtime_r(&timestamp, &tm_time);
  struct timeval tv;
  gettimeofday(&tv, NULL);

  // print date / time
  log_stream_ << '[' << level << ' ' << STL::setw(2) << 1 + tm_time.tm_mon
              << '/' << STL::setw(2) << tm_time.tm_mday << ' ' << STL::setw(2)
              << tm_time.tm_hour << ':' << STL::setw(2) << tm_time.tm_min << ':'
              << STL::setw(2) << tm_time.tm_sec << '.' << STL::setw(3)
              << tv.tv_usec / 1000 << " ";

  if (len > kMaxLen) {
    log_stream_ << "..." << file + len - kMaxLen << ":" << lineno << " " << func
                << "] ";
  } else {
    log_stream_ << file << " " << func << ":" << lineno << "] ";
  }
}

}  // namespace lite
}  // namespace paddle

#endif  // LITE_WITH_LOG
#endif  // LITE_WITH_LIGHT_FRAMEWORK
