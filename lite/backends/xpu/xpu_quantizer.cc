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

#include "lite/backends/xpu/xpu_quantizer.h"
#include <algorithm>
#include <string>
#include "lite/backends/xpu/math.h"
#include "lite/utils/hash.h"

namespace paddle {
namespace lite {

template <typename T>
static double AveGrowCompute(const T* in, const size_t length) {
  const double eps = 1e-5;
  double ave_grow_rate = 0.0f;
  for (size_t i = 1; i < length; ++i) {
    ave_grow_rate += (in[i] - in[i - 1]) / (in[i - 1] + eps);
  }
  ave_grow_rate /= (length + eps);
  return ave_grow_rate;
}

template <typename T>
static size_t Hashed(const T* cpu_data,
                     int numel,
                     const std::string& precision,
                     bool trans) {
  std::hash<const void*> ptr_hasher;
  auto hash_res = ptr_hasher(reinterpret_cast<const void*>(cpu_data));
  double ave_grow_rate = AveGrowCompute(cpu_data, numel);
  CombineHash(ave_grow_rate, &hash_res);
  CombineHash(numel, &hash_res);
  CombineHash(precision, &hash_res);
  CombineHash(trans, &hash_res);
  return hash_res;
}

template <typename T>
static inline const std::string CppTypeToString() {
  return "unkown";
}
template <>
inline const std::string CppTypeToString<float>() {
  return "float";
}
template <>
inline const std::string CppTypeToString<float16>() {
  return "float16";
}
template <>
inline const std::string CppTypeToString<int64_t>() {
  return "int64_t";
}
template <>
inline const std::string CppTypeToString<int>() {
  return "int";
}
template <>
inline const std::string CppTypeToString<int16_t>() {
  return "int16_t";
}
template <>
inline const std::string CppTypeToString<int8_t>() {
  return "int8_t";
}

template <typename T>
void XPUQuantizer::QuantFP32ToIntX(const float* src_ptr,
                                   T* dst_ptr,
                                   float max_val,
                                   int numel) {
  LOG(FATAL) << "Not support for T is " << CppTypeToString<T>();
}
template <>
void XPUQuantizer::QuantFP32ToIntX<float>(const float* src_ptr,
                                          float* dst_ptr,
                                          float max_val,
                                          int numel) {
  std::copy(src_ptr, src_ptr + numel, dst_ptr);
}
template <>
void XPUQuantizer::QuantFP32ToIntX<int16_t>(const float* src_ptr,
                                            int16_t* dst_ptr,
                                            float max_val,
                                            int numel) {
  paddle::lite::xpu::math::ConvertFP32ToInt16(src_ptr, dst_ptr, max_val, numel);
}
template <>
void XPUQuantizer::QuantFP32ToIntX<int8_t>(const float* src_ptr,
                                           int8_t* dst_ptr,
                                           float max_val,
                                           int numel) {
  paddle::lite::xpu::math::ConvertFP32ToInt8(src_ptr, dst_ptr, max_val, numel);
}

template <
    typename Tcpu,
    typename Txpu,
    typename std::enable_if<!std::is_same<Tcpu, float>::value, Tcpu>::type* ptr>
void XPUQuantizer::ConvertWithQuant(const Tcpu* cpu_data,
                                    const DDimLite& dims,
                                    bool data_transpose,
                                    size_t hashed_key,
                                    size_t max_ptr_len) {
  LOG(FATAL) << "Not support for Tcpu is " << CppTypeToString<Tcpu>();
}

template <
    typename Tcpu,
    typename Txpu,
    typename std::enable_if<std::is_same<Tcpu, float>::value, Tcpu>::type* ptr>
void XPUQuantizer::ConvertWithQuant(const Tcpu* cpu_data,
                                    const DDimLite& dims,
                                    bool data_transpose,
                                    size_t hashed_key,
                                    size_t max_ptr_len) {
  // transpose
  const Tcpu* cpu_ptr = nullptr;
  int numel = dims.production();
  std::vector<Tcpu> transpose_data(numel, 0);
  if (data_transpose) {
    CHECK_EQ(dims.size(), 2UL);
    paddle::lite::xpu::math::Transpose<Tcpu>(
        cpu_data, transpose_data.data(), dims[0], dims[1]);
    cpu_ptr = transpose_data.data();
  } else {
    cpu_ptr = cpu_data;
  }
  // findmax
  XPUScratchPadGuard weight_max_guard;
  XPUScratchPadGuard quant_weight_guard;
  float max_val = paddle::lite::xpu::math::FindMaxAbs(cpu_ptr, numel);
  size_t max_ptr_size = max_ptr_len;
  std::vector<float> max_vec(max_ptr_size, max_val);
  weight_max_guard =
      std::move(XPUMemory::MallocScratchPad(max_ptr_size * sizeof(float)));
  XPUMemory::MemcpyHtoDSync(
      weight_max_guard->addr_, max_vec.data(), max_ptr_size * sizeof(float));
  // quant
  quant_weight_guard =
      std::move(XPUMemory::MallocScratchPad(numel * sizeof(Txpu)));
  std::vector<Txpu> quant_data_cpu(numel, 0);
  QuantFP32ToIntX<Txpu>(cpu_ptr, quant_data_cpu.data(), max_val, numel);
  XPUMemory::MemcpyHtoDSync(
      quant_weight_guard->addr_, quant_data_cpu.data(), numel * sizeof(Txpu));
  // add to cache
  weight_cache_[hashed_key] = std::make_pair(std::move(weight_max_guard),
                                             std::move(quant_weight_guard));
}

template <typename T>
void XPUQuantizer::ConvertWithoutQuant(const T* cpu_data,
                                       const DDimLite& dims,
                                       bool data_transpose,
                                       size_t hashed_key,
                                       size_t max_ptr_len) {
  // transpose
  const T* cpu_ptr = nullptr;
  int numel = dims.production();
  size_t max_ptr_size = max_ptr_len;
  std::vector<T> transpose_data(numel, 0);
  if (data_transpose) {
    CHECK(dims.size() == 2) << "Not support: dims.size = " << dims.size();
    paddle::lite::xpu::math::Transpose<T>(
        cpu_data, transpose_data.data(), dims[0], dims[1]);
    cpu_ptr = transpose_data.data();
  } else {
    cpu_ptr = cpu_data;
  }
  // copy to XPU
  XPUScratchPadGuard weight_max_guard(new XPUScratchPad(nullptr, 0));
  if (std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value) {
    // prepare max_w space for slim int8 quant
    // just allocate buffer, set max value in kernel
    weight_max_guard =
        std::move(XPUMemory::MallocScratchPad(max_ptr_size * sizeof(float)));
  }
  XPUScratchPadGuard quant_weight_guard;
  quant_weight_guard =
      std::move(XPUMemory::MallocScratchPad(numel * sizeof(T)));
  XPUMemory::MemcpyHtoDSync(
      quant_weight_guard->addr_, cpu_ptr, numel * sizeof(T));
  // add to cache
  weight_cache_[hashed_key] = std::make_pair(std::move(weight_max_guard),
                                             std::move(quant_weight_guard));
}

template <typename Tcpu, typename Txpu>
XPUQuantData XPUQuantizer::quant(const Tcpu* cpu_data,
                                 const DDimLite& dims,
                                 bool data_transpose,
                                 size_t max_ptr_len) {
  int numel = dims.production();
  const std::string cpu_dtype = CppTypeToString<Tcpu>();
  const std::string xpu_dtype = CppTypeToString<Txpu>();
  const std::string precision = cpu_dtype + xpu_dtype;
  auto hashed_key = Hashed<Tcpu>(cpu_data, numel, precision, data_transpose);
  VLOG(3) << "cpu_data=" << cpu_data << ", numel=" << numel
          << ", precision=" << precision << ", transpose=" << data_transpose
          << ", hashed_key=" << hashed_key;
  if (weight_cache_.find(hashed_key) == weight_cache_.end()) {
    ConvertWrapper<Tcpu, Txpu>(
        cpu_data, dims, data_transpose, hashed_key, max_ptr_len);
  }

  float* max_ptr =
      reinterpret_cast<float*>(weight_cache_[hashed_key].first->addr_);
  void* qdata_ptr = weight_cache_[hashed_key].second->addr_;
  XPUQuantData xpu_qdata(max_ptr, qdata_ptr);
  return xpu_qdata;
}

template XPUQuantData XPUQuantizer::quant<float, float>(const float*,
                                                        const DDimLite&,
                                                        bool,
                                                        size_t);
template XPUQuantData XPUQuantizer::quant<float, int16_t>(const float*,
                                                          const DDimLite&,
                                                          bool,
                                                          size_t);
template XPUQuantData XPUQuantizer::quant<float, int8_t>(const float*,
                                                         const DDimLite&,
                                                         bool,
                                                         size_t);
template XPUQuantData XPUQuantizer::quant<int8_t, int8_t>(const int8_t*,
                                                          const DDimLite&,
                                                          bool,
                                                          size_t);
template XPUQuantData XPUQuantizer::quant<int16_t, int16_t>(const int16_t*,
                                                            const DDimLite&,
                                                            bool,
                                                            size_t);
}  // namespace lite
}  // namespace paddle
