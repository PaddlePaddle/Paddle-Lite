// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_map>
#include <utility>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/backends/xpu/xpu_scratch.h"
#include "lite/core/dim.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {

struct XPUQuantData {
  XPUQuantData() : data_ptr_(nullptr), max_ptr_(nullptr) {}
  XPUQuantData(float* max_ptr, void* data_ptr)
      : data_ptr_(data_ptr), max_ptr_(max_ptr) {}
  void* data_ptr_{nullptr};
  float* max_ptr_{nullptr};
};

class XPUQuantizer {
 public:
  template <typename Tcpu, typename Txpu>
  XPUQuantData quant(const Tcpu* cpu_data,
                     const DDimLite& dims,
                     bool data_transpose,
                     size_t max_ptr_len);

 private:
  template <typename T>
  void QuantFP32ToIntX(const float* src_ptr,
                       T* dst_ptr,
                       float max_val,
                       int numel);

  template <typename T>
  void ConvertWithoutQuant(const T* cpu_data,
                           const DDimLite& dims,
                           bool data_transpose,
                           size_t hashed_key,
                           size_t max_ptr_len);

  template <typename Tcpu,
            typename Txpu,
            typename std::enable_if<std::is_same<Tcpu, float>::value,
                                    Tcpu>::type* ptr = nullptr>
  void ConvertWithQuant(const Tcpu* cpu_data,
                        const DDimLite& dims,
                        bool data_transpose,
                        size_t hashed_key,
                        size_t max_ptr_len);

  template <typename Tcpu,
            typename Txpu,
            typename std::enable_if<!std::is_same<Tcpu, float>::value,
                                    Tcpu>::type* ptr = nullptr>
  void ConvertWithQuant(const Tcpu* cpu_data,
                        const DDimLite& dims,
                        bool data_transpose,
                        size_t hashed_key,
                        size_t max_ptr_len);

  template <typename Tcpu,
            typename Txpu,
            typename std::enable_if<!std::is_same<Tcpu, Txpu>::value,
                                    Tcpu>::type* ptr = nullptr>
  void ConvertWrapper(const Tcpu* cpu_data,
                      const DDimLite& dims,
                      bool data_transpose,
                      size_t hashed_key,
                      size_t max_ptr_len) {
    ConvertWithQuant<Tcpu, Txpu>(
        cpu_data, dims, data_transpose, hashed_key, max_ptr_len);
  }

  template <typename Tcpu,
            typename Txpu,
            typename std::enable_if<std::is_same<Tcpu, Txpu>::value,
                                    Tcpu>::type* ptr = nullptr>
  void ConvertWrapper(const Tcpu* cpu_data,
                      const DDimLite& dims,
                      bool data_transpose,
                      size_t hashed_key,
                      size_t max_ptr_len) {
    ConvertWithoutQuant<Tcpu>(
        cpu_data, dims, data_transpose, hashed_key, max_ptr_len);
  }

  // cpu data to xpu quant data
  std::unordered_map<size_t, std::pair<XPUScratchPadGuard, XPUScratchPadGuard>>
      weight_cache_;
};

}  // namespace lite
}  // namespace paddle
