/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "common/log.h"

int main() {
  LOG(paddle_mobile::kLOG_DEBUG3) << "test debug"
                                  << " next log";
  LOG(paddle_mobile::kLOG_DEBUG) << "test debug"
                                 << " next log";

  LOG(paddle_mobile::kLOG_DEBUG1) << "test debug1"
                                  << " next log";
  LOG(paddle_mobile::kLOG_DEBUG2) << "test debug2"
                                  << " next log";
  LOG(paddle_mobile::kLOG_INFO) << "INFO!!!";
  LOG(paddle_mobile::kLOG_WARNING) << "WARNING!!!";
  LOG(paddle_mobile::kLOG_VERBOSE) << "VERBOSE!!!";
  DLOG << "test DLOG";

  LOG(paddle_mobile::kLOG_ERROR) << "ERROR !";

  return 0;
}
