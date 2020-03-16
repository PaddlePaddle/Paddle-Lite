/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "io.hpp"

namespace paddle {
namespace zynqmp {

// FpgaIO::FpgaIO() {}
// void FpgaIO::setMutex(std::mutex* mtx) { mtx_ = mtx; }

// void FpgaIO::setConditionVariable(std::condition_variable* condition) {
//   condition_ = condition;
// }

// void FpgaIO::lock() {
//   if (mtx_ != nullptr && !locked_) {
//     mtx_->lock();
//     locked_ = true;
//   }
// }

// void FpgaIO::unlock() {
//   if (mtx_ != nullptr) {
//     mtx_->unlock();
//     condition_->notify_one();
//   }
//   locked_ = false;
// }

}  // namespace zynqmp
}  // namespace paddle
