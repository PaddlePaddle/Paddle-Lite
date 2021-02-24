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

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <memory>  // std::unique_ptr
#include <mutex>
#include "lite/backends/xpu/xpu_header_sitter.h"  // xpu_free
#include "lite/core/target_wrapper.h"             // TargetWrapper
#include "lite/utils/cp_logging.h"                // CHECK_EQ
#include "lite/utils/macros.h"

#define gettidv1() syscall(__NR_gettid)
#define XPU_CALL(func)                                        \
  {                                                           \
    auto e = (func);                                          \
    CHECK_EQ(e, 0) << "XPU: (" << #func << ") returns " << e; \
  }

namespace paddle {
namespace lite {

// MAX(lod.size()) = 32
const int XPU_MAX_LOD_SIZE = 32;
// MAX(lod[i + 1] - lod[i]) = 512
const int XPU_MAX_LOD_SEQ_LEN = 512;
// the simulator xpu dev id is 64
const int MAX_XPU_DEV_NUM = 65;

using TargetWrapperXPU = TargetWrapper<TARGET(kXPU)>;

struct XPUScratchPad {
  XPUScratchPad(void* addr, size_t size, bool is_l3)
      : addr_(addr), size_(size), is_l3_(is_l3) {}

  // XXX(miaotianxiang): |size_| increases monotonically
  void Reserve(size_t new_size);

  void* addr_{nullptr};
  size_t size_{0};
  bool is_l3_{false};
};

struct XPUScratchPadDeleter {
  void operator()(XPUScratchPad* sp) const;
};

using XPUScratchPadGuard = std::unique_ptr<XPUScratchPad, XPUScratchPadDeleter>;

template <>
class TargetWrapper<TARGET(kXPU)> {
 public:
  static size_t num_devices() { return 1; }
  static size_t maximum_stream() { return 0; }

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir);

  static XPUScratchPadGuard MallocScratchPad(size_t size, bool use_l3 = false);

  static xdnn::Context* GetRawContext() {
    if (tls_raw_ctx_ == nullptr) {
      tls_raw_ctx_ = xdnn::create_context();
      CHECK(tls_raw_ctx_);
      if (conv_autotune) {
        tls_raw_ctx_->_xpu1_conv_selector.set_autotune_loop(true);
        tls_raw_ctx_->_xpu1_conv_selector.set_inference_mode(true);
      }
      if (!conv_autotune_file.empty()) {
        tls_raw_ctx_->_xpu1_conv_selector.set_autotune_file(
            conv_autotune_file.c_str());
      }
      int r = xdnn::set_workspace_l3_size(tls_raw_ctx_,
                                          workspace_l3_size_per_thread);
      if (r != 0) {
        LOG(WARNING) << "xdnn::set_workspace_l3_size() failed, r = " << r
                     << ", workspace_l3_size_per_thread = "
                     << workspace_l3_size_per_thread;
      }
    }
    return tls_raw_ctx_;
  }

  // **DEPRECATED**, use xpu_set_device() at the very beginning of each worker
  // thread
  static void SetDev(int dev_no = 0) {
    const char* dev_env = getenv("LITE_XPU_DEV");
    if (dev_env) {
      dev_no = atoi(dev_env);
    }

    XPU_CALL(xpu_set_device(dev_no));
  }
  static bool AtomicSetMutexIfNull(pthread_mutex_t** obj,
                                   pthread_mutex_t* value) {
    static std::mutex mutex_dev;
    std::lock_guard<std::mutex> lock(mutex_dev);
    if ((*obj) == nullptr) {
      *obj = value;
      return true;
    } else {
      return false;
    }
  }
  static int InitXPUMutex(int pd) {
    // multiple update of enable_xpu_concurrent is safe
    auto xpu_concurrent_str = std::getenv("ENABLE_XPU_CONCURRENT");
    if (xpu_concurrent_str && std::atoi(xpu_concurrent_str) > 0 &&
        enable_xpu_concurrent == false) {
      enable_xpu_concurrent = true;
      VLOG(5) << "ENABLE_XPU_CONCURRENT";
    }
    if (enable_xpu_concurrent == false) {
      return 0;
    }
    std::string xpu_dev_shm = "/shm_xpu_dev" + std::to_string(pd);
    std::string xpu_dev_shm_tmp = xpu_dev_shm + "_" + std::to_string(getpid()) +
                                  "_" + std::to_string(gettidv1());
    // every thread in a process could shm_open succeed;
    // but only one thread could 'link' succeed
    int fd = shm_open(
        xpu_dev_shm_tmp.c_str(), O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR);
    CHECK(fd >= 0) << "thread shm_open failed: " << errno;
    // construct the global mutex
    int r = ftruncate(fd, sizeof(pthread_mutex_t));
    CHECK(r == 0) << "ftruncate failed" << errno;
    pthread_mutex_t* mutex =
        reinterpret_cast<pthread_mutex_t*>(mmap(NULL,
                                                sizeof(pthread_mutex_t),
                                                PROT_READ | PROT_WRITE,
                                                MAP_SHARED,
                                                fd,
                                                0));
    CHECK(mutex != MAP_FAILED) << "mmap failed, errno: " << errno;
    close(fd);

    pthread_mutexattr_t ma;
    pthread_mutexattr_init(&ma);
    pthread_mutexattr_setpshared(&ma, PTHREAD_PROCESS_SHARED);
    pthread_mutexattr_setrobust(&ma, PTHREAD_MUTEX_ROBUST);
    pthread_mutex_init(mutex, &ma);

    std::string full_path_xpu_shm = "/dev/shm" + xpu_dev_shm;
    std::string full_path_xpu_shm_tmp = "/dev/shm" + xpu_dev_shm_tmp;
    // only one process/thread could link successfully
    if (link(full_path_xpu_shm_tmp.c_str(), full_path_xpu_shm.c_str()) == 0) {
      // the first creator, this mutex is picked
      if (AtomicSetMutexIfNull(&xpu_dev_mutex_[pd], mutex)) {
        VLOG(5) << "AtomicSetMutex succeed, pid/tid/mutex: " << getpid() << ", "
                << gettidv1() << ", " << mutex;
      } else {
        // other threads updated the mutex
        pthread_mutex_destroy(mutex);
        munmap(mutex, sizeof(pthread_mutex_t));
        VLOG(5) << "other thread has AtomicSetMutex succeed: "
                << xpu_dev_mutex_[pd];
      }
      VLOG(5) << "link " << full_path_xpu_shm << " successed";
    } else {
      // other process created the xpu shm, now get the mutex in
      // full_path_xpu_shm
      pthread_mutex_destroy(mutex);
      munmap(mutex, sizeof(pthread_mutex_t));
      if (xpu_dev_mutex_[pd] == nullptr) {
        int xpu_shm_fd = open(full_path_xpu_shm.c_str(), O_RDWR, 0);
        CHECK(xpu_shm_fd >= 0) << " assumed xpu shm existed, but open failed: "
                               << errno;
        mutex = reinterpret_cast<pthread_mutex_t*>(mmap(NULL,
                                                        sizeof(pthread_mutex_t),
                                                        PROT_READ | PROT_WRITE,
                                                        MAP_SHARED,
                                                        xpu_shm_fd,
                                                        0));
        CHECK(mutex != MAP_FAILED) << " mmap failed: " << errno;
        if (!AtomicSetMutexIfNull(&xpu_dev_mutex_[pd], mutex)) {
          pthread_mutex_destroy(mutex);
          munmap(mutex, sizeof(pthread_mutex_t));
          VLOG(5) << "shm and mutex already existed, pid/tid/mutex: "
                  << getpid() << ", " << gettidv1() << ", "
                  << xpu_dev_mutex_[pd];
        } else {
          VLOG(5) << "get mutex from shm, pid/tid/mutex: " << getpid() << ", "
                  << gettidv1() << ", " << mutex;
        }
        close(xpu_shm_fd);
      }
    }
    shm_unlink(xpu_dev_shm_tmp.c_str());
    return 0;
  }
  static void LockXPU() {
    int pd = -1;
    XPU_CALL(xpu_current_device(&pd));
    CHECK(pd >= 0) << "Wrong Current XPU Device Num:" << pd;
    CHECK(pd < MAX_XPU_DEV_NUM) << " Wrong XPU Device Num: " << pd
                                << " should < " << MAX_XPU_DEV_NUM;
    // check ctx and init
    if (tls_raw_ctx_ == nullptr) {
      GetRawContext();
      CHECK(tls_raw_ctx_);
      InitXPUMutex(pd);
      auto xpu_l3_lock_size_str = std::getenv("XPU_L3_LOCK_SIZE");
      if (xpu_l3_lock_size_str && (std::atoi(xpu_l3_lock_size_str) > 0)) {
        xpu_l3_lock_size = std::atoi(xpu_l3_lock_size_str);
      }
    }
    // check after InitXPUMutex
    if (enable_xpu_concurrent == false) {
      return;
    }
    CHECK(xpu_dev_mutex_[pd] != nullptr) << " InitXPUMutex failed";
    int r = 0;
    pthread_mutex_t* mutex = xpu_dev_mutex_[pd];
    while ((r = pthread_mutex_lock(mutex)) == EOWNERDEAD) {
      CHECK(pthread_mutex_consistent(mutex) == 0)
          << " pthread_mutex_consistent failed:" << errno;
      pthread_mutex_unlock(mutex);
      VLOG(5) << "pthread_mutex_consistent succeed";
    }
    CHECK(r == 0) << " pthread_mutex_lock failed: " << errno;
    VLOG(5) << "pid: " << getpid() << ", tid:" << gettidv1() << " LockXPU";
    // this should be put in a separate funciton;
    // set l3 cache
    void* xpu_l3_ptr = nullptr;
    // free thread level L3 Cache
    xpu_l3_ptr = tls_raw_ctx_->_l3_mgr.get_ptr();
    if (xpu_l3_ptr != nullptr) {
      XPU_CALL(xpu_free(xpu_l3_ptr));
      tls_raw_ctx_->_l3_mgr.set(nullptr, 0);
    }
    // malloc process level L3 Cache
    XPU_CALL(xpu_malloc(
        reinterpret_cast<void**>(&xpu_l3_ptr), xpu_l3_lock_size, XPU_MEM_L3));
    CHECK(xpu_l3_ptr != nullptr)
        << "XPU L3 Cache Malloc Fail, No Enough L3 Cache";
    tls_raw_ctx_->_l3_mgr.set(xpu_l3_ptr, xpu_l3_lock_size);
  }
  static void UnLockXPU() {
    if (enable_xpu_concurrent == false) {
      return;
    }
    int pd = -1;
    XPU_CALL(xpu_current_device(&pd));
    CHECK(pd >= 0) << "Wrong Current XPU Device Num";
    CHECK(pd < MAX_XPU_DEV_NUM) << " Wrong XPU Device Num";
    if (xpu_l3_lock_size > 0) {
      void* l3_ptr = tls_raw_ctx_->_l3_mgr.get_ptr();
      if (l3_ptr != nullptr) {
        XPU_CALL(xpu_free(l3_ptr));
        tls_raw_ctx_->_l3_mgr.set(nullptr, 0);
        VLOG(5) << "pid: " << getpid() << ", tid:" << gettidv1()
                << " UnLockXPU";
      }
    }
    pthread_mutex_t* mutex = xpu_dev_mutex_[pd];
    CHECK(mutex != nullptr) << " xpu_dev_mutex_[pd] invalid in UnLockXPU";
    pthread_mutex_unlock(mutex);
  }
  static void LockL3Cache() {
    auto xpu_l3_lock_size_str = std::getenv("XPU_L3_LOCK_SIZE");
    xpu_l3_lock_size = -1;
    xpu_l3_lock_fd = -1;
    struct flock f_lock;
    f_lock.l_whence = 0;
    f_lock.l_len = 0;

    if (xpu_l3_lock_size_str && (std::atoi(xpu_l3_lock_size_str) > 0)) {
      int pd = -1;
      XPU_CALL(xpu_current_device(&pd));
      CHECK(pd >= 0) << "Wrong Current XPU Device Num";
      CHECK(pd < MAX_XPU_DEV_NUM) << " Wrong XPU Device Num";
      std::string buf = "/opt/xpu_lock" + std::to_string(pd);

      xpu_l3_lock_fd = open(buf.c_str(), O_RDWR);
      CHECK(xpu_l3_lock_fd > 0) << "open " << buf << " failed "
                                << xpu_l3_lock_fd;

      // lock
      f_lock.l_type = F_WRLCK;
      fcntl(xpu_l3_lock_fd, F_SETLKW, &f_lock);
      // check ctx and init
      if (tls_raw_ctx_ == nullptr) {
        GetRawContext();
        CHECK(tls_raw_ctx_);
      }
      // set l3 cache
      void* xpu_l3_ptr = nullptr;
      // free thread level L3 Cache
      xpu_l3_ptr = tls_raw_ctx_->_l3_mgr.get_ptr();
      if (xpu_l3_ptr != nullptr) {
        XPU_CALL(xpu_free(xpu_l3_ptr));
      }
      // malloc process level L3 Cache
      xpu_l3_lock_size = std::atoi(xpu_l3_lock_size_str);
      XPU_CALL(xpu_malloc(
          reinterpret_cast<void**>(&xpu_l3_ptr), xpu_l3_lock_size, XPU_MEM_L3));
      CHECK(xpu_l3_ptr != nullptr)
          << "XPU L3 Cache Malloc Fail, No Enough L3 Cache";
      tls_raw_ctx_->_l3_mgr.set(xpu_l3_ptr, xpu_l3_lock_size);
    }
  }

  static void FreeL3Cache() {
    if (xpu_l3_lock_size > 0) {
      void* l3_ptr = tls_raw_ctx_->_l3_mgr.get_ptr();
      if (l3_ptr != nullptr) {
        XPU_CALL(xpu_free(l3_ptr));
        VLOG(5) << "getpid: " << gettidv1() << "FreeL3Cache";
      }
      struct flock f_lock;
      f_lock.l_whence = 0;
      f_lock.l_len = 0;
      f_lock.l_type = F_UNLCK;
      fcntl(xpu_l3_lock_fd, F_SETLKW, &f_lock);
      close(xpu_l3_lock_fd);
    }
  }

  static std::string multi_encoder_precision;  // NOLINT
  static int workspace_l3_size_per_thread;
  static bool conv_autotune;
  static std::string conv_autotune_file;  // NOLINT

 private:
  static LITE_THREAD_LOCAL xdnn::Context* tls_raw_ctx_;
  static int xpu_l3_lock_size;
  static int xpu_l3_lock_fd;
  // use multitple process or thread in one device
  static bool enable_xpu_concurrent;
  static pthread_mutex_t* xpu_dev_mutex_[MAX_XPU_DEV_NUM];
};

}  // namespace lite
}  // namespace paddle
