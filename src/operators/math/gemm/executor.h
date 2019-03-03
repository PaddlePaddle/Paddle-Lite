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

#pragma once

#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <sys/time.h>
#include <iostream>
#include "common/log.h"
#include "memory/t_malloc.h"
#include "operators/math/gemm/cpu_info.h"
#include "operators/math/gemm/gemm_kernel.h"

namespace paddle_mobile {
namespace operators {
namespace math {

inline int CeilDiv(const int &x, const int &y) { return (x + y - 1) / y; }

class Executor {
 public:
  Executor() : num_threads_(1) {
#ifdef _OPENMP
    num_threads_ = omp_get_max_threads();
#endif
  }
  virtual ~Executor() {}

 protected:
  int num_threads_;
};

template <typename Strategy>
class GemmExecutor : public Executor {
  typedef typename Strategy::Itype Itype;
  typedef typename Strategy::Otype Otype;

 public:
  GemmExecutor(const CPUInfo *info, const bool transA, const bool transB,
               const int M, const int N, const int K)
      : Executor(),
        info_(info),
        transA_(transA),
        transB_(transB),
        M_(M),
        N_(N),
        K_(K) {
    unsigned int L1_size = info->L1_cache;
    unsigned int L2_size = info->L2_cache;
    if (N_ > 30000 && K_ > 100) L1_size *= 2;
    if (num_threads_ >= 2) L1_size /= 2;

    rhs_tile_num_ = L1_size / (K * sizeof(Itype));
    if (rhs_tile_num_ == 0) {
      rhs_tile_num_ = Strategy::out_width();
    } else {
      int n_block = CeilDiv(N, rhs_tile_num_);
      rhs_tile_num_ = CeilDiv(N, n_block);
      rhs_tile_num_ = CeilDiv(rhs_tile_num_, Strategy::out_width());
      rhs_tile_num_ *= Strategy::out_width();
    }

    //  lhs_tile_num_ = CeilDiv(M, Strategy::out_height()) *
    //  Strategy::out_height();
    lhs_tile_num_ = L2_size / (K * sizeof(Itype));
    if (lhs_tile_num_ == 0) {
      lhs_tile_num_ = Strategy::out_height();
    } else {
      int m_block = CeilDiv(M, lhs_tile_num_);
      lhs_tile_num_ = CeilDiv(M, m_block);
      lhs_tile_num_ = CeilDiv(lhs_tile_num_, Strategy::out_height());
      lhs_tile_num_ *= Strategy::out_height();
    }
  }

  void operator()(const float alpha, const Itype *A, const int lda,
                  const Itype *B, const int ldb, const float beta, Otype *C,
                  const int ldc) {
    //  struct timeval tv_begin, tv_end;
    //  gettimeofday(&tv_begin,NULL);

    int mblock = CeilDiv(M_, Strategy::out_height()) * Strategy::out_height();
    lhs_worksize_ = sizeof(Itype) * mblock * K_;
    rhs_worksize_ = sizeof(Itype) * K_ * rhs_tile_num_ * num_threads_;
    out_worksize_ = sizeof(Otype) * mblock * rhs_tile_num_ * num_threads_;

    lhs_workspace_ =
        static_cast<Itype *>(paddle_mobile::memory::Alloc(lhs_worksize_));
    rhs_workspace_ =
        static_cast<Itype *>(paddle_mobile::memory::Alloc(rhs_worksize_));
    out_workspace_ =
        static_cast<Otype *>(paddle_mobile::memory::Alloc(out_worksize_));

    strategy_.pack_lhs(M_, K_, A, lda, lhs_workspace_, true);

    //  std::cout << "M: " << M_ << ", N: " << N_
    //            << ", K: " << K_ << std::endl;
    //  std::cout << "rhs_block: " << CeilDiv(N_, rhs_tile_num_)
    //            << std::endl;

    #pragma omp parallel for if (N_ > 128)
    for (int rhs_block = 0; rhs_block < N_; rhs_block += rhs_tile_num_) {
      int rhs_range = std::min(N_ - rhs_block, rhs_tile_num_);
#ifdef _OPENMP
      int thread_id = omp_get_thread_num();
#else
      int thread_id = 0;
#endif
      float *local_B = rhs_workspace_ + K_ * rhs_tile_num_ * thread_id;
      float *local_C =
          out_workspace_ + lhs_tile_num_ * rhs_tile_num_ * thread_id;
      // load rhs into rhs_workspace
      strategy_.pack_rhs(K_, rhs_range, B + rhs_block, ldb, local_B, false);
      for (int lhs_block = 0; lhs_block < M_; lhs_block += lhs_tile_num_) {
        int lhs_range = std::min(M_ - lhs_block, lhs_tile_num_);
        float *local_A = lhs_workspace_ + lhs_block * lda;
        for (int lhs_tile = 0; lhs_tile < lhs_range;
             lhs_tile += Strategy::out_height()) {
          for (int rhs_tile = 0; rhs_tile < rhs_range;
               rhs_tile += Strategy::out_width()) {
            int offset = (lhs_block + lhs_tile) * rhs_tile_num_ + rhs_tile;
            strategy_.kernel(local_A + lhs_tile * K_, local_B + rhs_tile * K_,
                             K_, local_C + offset, rhs_tile_num_);
          }
        }
      }
      strategy_.write(M_, rhs_range, local_C, rhs_tile_num_, C + rhs_block,
                      ldc);
    }

    paddle_mobile::memory::Free(lhs_workspace_);
    paddle_mobile::memory::Free(rhs_workspace_);
    paddle_mobile::memory::Free(out_workspace_);

    //  gettimeofday(&tv_end,NULL);
    //  float elapsed = (tv_end.tv_sec - tv_begin.tv_sec) * 1000.f +
    //                  (tv_end.tv_usec - tv_begin.tv_usec) / 1000.f;
    //  std::cout << "elapsed: " << elapsed << "ms, speed: "
    //            << (M_ * N_ * K_ / 1000.f / 1000.f) / elapsed
    //            << " gflops" << std::endl;
  }

  virtual ~GemmExecutor() {}

 private:
  const CPUInfo *info_;

  const unsigned int M_;
  const unsigned int N_;
  const unsigned int K_;
  const bool transA_;
  const bool transB_;

  unsigned int lhs_tile_num_ = 0;
  unsigned int rhs_tile_num_ = 0;
  unsigned int out_tile_num_ = 0;

  unsigned int lhs_worksize_ = 0;
  unsigned int rhs_worksize_ = 0;
  unsigned int out_worksize_ = 0;

  Itype *lhs_workspace_ = nullptr;
  Itype *rhs_workspace_ = nullptr;
  Otype *out_workspace_ = nullptr;

  Strategy strategy_;
};

template <typename Strategy>
class GemvExecutor : public Executor {
  typedef typename Strategy::Itype Itype;
  typedef typename Strategy::Otype Otype;

 public:
  GemvExecutor(const CPUInfo *info, const bool transA, const int M, const int N)
      : Executor(), info_(info), M_(M), N_(N) {}

  void operator()(const float alpha, const Itype *A, const int lda,
                  const Itype *B, const float beta, Otype *C) {
    //  strategy_.kernel();
  }

  virtual ~GemvExecutor() {}

 private:
  const CPUInfo *const info_;

  const unsigned int M_;
  const unsigned int N_;

  Strategy strategy_;
};

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
