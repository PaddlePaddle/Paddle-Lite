//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#ifdef PADDLE_WITH_MKLML
#include <omp.h>
#include "lite/backends/x86/mklml.h"
#endif

namespace paddle {
namespace lite {
namespace x86 {

static void SetNumThreads(int num_threads) {
#ifdef PADDLE_WITH_MKLML
  int real_num_threads = (std::max)(num_threads, 1);
#ifdef LITE_WITH_STATIC_MKL
  MKL_Set_Num_Threads(real_num_threads);
#else
  x86::MKL_Set_Num_Threads(real_num_threads);
#endif
  omp_set_num_threads(real_num_threads);
#endif
}

static inline int64_t GetMaxThreads() {
  int64_t num_threads = 1;
#ifdef PADDLE_WITH_MKLML
  // Do not support nested omp parallem.
  num_threads = omp_in_parallel() ? 1 : omp_get_max_threads();
#endif
  return std::max<int>(num_threads, 1L);
}

using ThreadHandler =
    std::function<void(const int64_t begin, const int64_t end)>;

static inline void RunParallelFor(const int64_t begin,
                                  const int64_t end,
                                  const ThreadHandler& f) {
  if (begin >= end) {
    return;
  }

#ifdef PADDLE_WITH_MKLML
  int64_t num_threads = (std::min)(GetMaxThreads(), end - begin);
  if (num_threads > 1) {
#pragma omp parallel num_threads(num_threads)
    {
      int64_t tid = omp_get_thread_num();
      int64_t chunk_size = (end - begin + num_threads - 1) / num_threads;
      int64_t begin_tid = begin + tid * chunk_size;
      f(begin_tid, (std::min)(end, chunk_size + begin_tid));
    }
    return;
  }
#endif

  f(begin, end);
}

}  // namespace x86
}  // namespace lite
}  // namespace paddle
