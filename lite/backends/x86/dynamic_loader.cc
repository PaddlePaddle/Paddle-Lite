/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "lite/backends/x86/dynamic_loader.h"

#include <memory>
#include <mutex>  // NOLINT
#include <string>

#include "lite/backends/x86/port.h"
#include "lite/backends/x86/warpctc_lib_path.h"
#include "lite/utils/env.h"
#include "lite/utils/log/cp_logging.h"

// DEFINE_string(warpctc_dir, "", "Specify path for loading libwarpctc.so.");
std::string f_warpctc_dir =                         // NOLINT
    paddle::lite::GetStringFromEnv("warpctc_dir");  // NOLINT

// DEFINE_string(
//     tensorrt_dir,
//     "",
//     "Specify path for loading tensorrt library, such as libnvinfer.so.");
std::string tensorrt_dir =                           // NOLINT
    paddle::lite::GetStringFromEnv("tensorrt_dir");  // NOLINT

// DEFINE_string(mklml_dir, "", "Specify path for loading libmklml_intel.so.");
std::string mklml_dir = paddle::lite::GetStringFromEnv("mklml_dir");  // NOLINT

namespace paddle {
namespace lite {
namespace x86 {
static constexpr char warpctc_lib_path[] = WARPCTC_LIB_PATH;

static inline std::string join(const std::string& part1,
                               const std::string& part2) {
  // directory separator
  const char sep = '/';
  if (!part2.empty() && part2.front() == sep) {
    return part2;
  }
  std::string ret;
  ret.reserve(part1.size() + part2.size() + 1);
  ret = part1;
  if (!ret.empty() && ret.back() != sep) {
    ret += sep;
  }
  ret += part2;
  return ret;
}

static inline void* GetDsoHandleFromDefaultPath(const std::string& dso_path,
                                                int dynload_flags) {
  VLOG(3) << "Try to find library: " << dso_path
          << " from default system path.";
  // default search from LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
  // and /usr/local/lib path
  void* dso_handle = dlopen(dso_path.c_str(), dynload_flags);

// DYLD_LIBRARY_PATH is disabled after Mac OS 10.11 to
// bring System Integrity Projection (SIP), if dso_handle
// is null, search from default package path in Mac OS.
#if defined(__APPLE__) || defined(__OSX__)
  if (nullptr == dso_handle) {
    dso_handle =
        dlopen(join("/usr/local/cuda/lib/", dso_path).c_str(), dynload_flags);
    if (nullptr == dso_handle) {
      if (dso_path == "libcudnn.dylib") {
        LOG(WARNING) << "Note: [Recommend] copy cudnn into /usr/local/cuda/ \n "
                        "For instance, sudo tar -xzf "
                        "cudnn-7.5-osx-x64-v5.0-ga.tgz -C /usr/local \n sudo "
                        "chmod a+r /usr/local/cuda/include/cudnn.h "
                        "/usr/local/cuda/lib/libcudnn*";
      }
    }
  }
#endif

  if (nullptr == dso_handle) {
    LOG(WARNING) << "Can not find library: " << dso_path
                 << ". The process maybe hang. Please try to add the lib path "
                    "to LD_LIBRARY_PATH.";
  }
  return dso_handle;
}

static inline void* GetDsoHandleFromSearchPath(const std::string& search_root,
                                               const std::string& dso_name,
                                               bool throw_on_error = true) {
#if !defined(_WIN32)
  int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
#else
  int dynload_flags = 0;
#endif  // !_WIN32
  void* dso_handle = nullptr;

  std::string dlPath = dso_name;
  if (search_root.empty()) {
    dso_handle = GetDsoHandleFromDefaultPath(dlPath, dynload_flags);
  } else {
    // search xxx.so from custom path
    dlPath = join(search_root, dso_name);
    dso_handle = dlopen(dlPath.c_str(), dynload_flags);
#if !defined(_WIN32)
    auto errorno = dlerror();
#else
    auto errorno = GetLastError();
#endif  // !_WIN32
    // if not found, search from default path
    if (nullptr == dso_handle) {
      LOG(WARNING) << "Failed to find dynamic library: " << dlPath << " ("
                   << errorno << ")";
      if (dlPath.find("nccl") != std::string::npos) {
        LOG(INFO)
            << "You may need to install 'nccl2' from NVIDIA official website: "
            << "https://developer.nvidia.com/nccl/nccl-download"
            << "before install PaddlePaddle";
      }
      dlPath = dso_name;
      dso_handle = GetDsoHandleFromDefaultPath(dlPath, dynload_flags);
    }
  }
/*
auto error_msg =
    "Failed to find dynamic library: %s ( %s ) \n Please specify "
    "its path correctly using following ways: \n Method. set "
    "environment variable LD_LIBRARY_PATH on Linux or "
    "DYLD_LIBRARY_PATH on Mac OS. \n For instance, issue command: "
    "export LD_LIBRARY_PATH=... \n Note: After Mac OS 10.11, "
    "using the DYLD_LIBRARY_PATH is impossible unless System "
    "Integrity Protection (SIP) is disabled.";
*/
#if !defined(_WIN32)
// auto errorno = dlerror();
#else
  auto errorno = GetLastError();
#endif  // !_WIN32
  if (throw_on_error) {
    CHECK(dso_handle != nullptr);
    // CHECK(nullptr != dso_handle, error_msg, dlPath, errorno);
  } else if (nullptr == dso_handle) {
    // LOG(WARNING) << string::Sprintf(error_msg, dlPath, errorno);
  }

  return dso_handle;
}

void* GetWarpCTCDsoHandle() {
  std::string warpctc_dir = warpctc_lib_path;
  if (!f_warpctc_dir.empty()) {
    warpctc_dir = f_warpctc_dir;
  }
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(warpctc_dir, "libwarpctc.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(warpctc_dir, "warpctc.dll");
#else
  return GetDsoHandleFromSearchPath(warpctc_dir, "libwarpctc.so");
#endif
}

void* GetTensorRtDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(tensorrt_dir, "libnvinfer.dylib");
#else
  return GetDsoHandleFromSearchPath(tensorrt_dir, "libnvinfer.so");
#endif
}

void* GetMKLMLDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(mklml_dir, "libmklml.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(mklml_dir, "mklml.dll");
#else
  return GetDsoHandleFromSearchPath(mklml_dir, "libmklml_intel.so");
#endif
}

}  // namespace x86
}  // namespace lite
}  // namespace paddle
