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

#include "lite/backends/xpu/target_wrapper.h"
#include <limits>
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {

void XPUScratchPad::Reserve(size_t new_size) {
  if (new_size <= size_) {
    return;
  }
  TargetWrapperXPU::Free(addr_);
  addr_ = TargetWrapperXPU::Malloc(new_size);
  size_ = new_size;
}

void XPUScratchPadDeleter::operator()(XPUScratchPad* sp) const {
  TargetWrapperXPU::Free(sp->addr_);
  delete sp;
}

XPUL3CacheBlock* TargetWrapperXPU::CreateL3CacheBlock() {
  l3_block_dict.push_back(new XPUL3CacheBlock());
  return l3_block_dict.back();
}

void* TargetWrapperXPU::Malloc(size_t size) {
  void* ptr{nullptr};
  if (size > 0) {
    XPU_CALL(xpu_malloc(&ptr, size));
  }
  return ptr;
}

void TargetWrapperXPU::Free(void* ptr) {
  XPU_CALL(xpu_wait());
  XPU_CALL(xpu_free(ptr));
}

void TargetWrapperXPU::MemcpySync(void* dst,
                                  const void* src,
                                  size_t size,
                                  IoDirection dir) {
  switch (dir) {
    case IoDirection::HtoD:
      XPU_CALL(xpu_wait());
      XPU_CALL(xpu_memcpy(dst, src, size, XPU_HOST_TO_DEVICE));
      break;
    case IoDirection::DtoH:
      XPU_CALL(xpu_wait());
      XPU_CALL(xpu_memcpy(dst, src, size, XPU_DEVICE_TO_HOST));
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

XPUScratchPadGuard TargetWrapperXPU::MallocScratchPad(size_t size) {
  void* ptr = TargetWrapperXPU::Malloc(size);
  CHECK(ptr) << "XPU Malloc Fail, Malloc Size is: " << size;
  return XPUScratchPadGuard(new XPUScratchPad(ptr, size));
}

template <typename T>
static inline size_t hash_combine(size_t seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}

static size_t Hashed(const float* cpu_data,
                     int numel,
                     const std::string& precision,
                     bool trans) {
  std::hash<const float*> ptr_hasher;
  auto hash_res = ptr_hasher(cpu_data);
  hash_res = hash_combine(hash_res, numel);
  hash_res = hash_combine(hash_res, precision);
  hash_res = hash_combine(hash_res, trans);
  return hash_res;
}

XPUQuantData TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight(
    const float* cpu_data,
    const DDimLite& dims,
    const std::string& precision,
    bool data_transpose) {
  int numel = dims.production();
  auto hashed_key = Hashed(cpu_data, numel, precision, data_transpose);
  VLOG(3) << "cpu_data=" << cpu_data << ", numel=" << numel
          << ", precision=" << precision << ", transpose=" << data_transpose
          << ", hashed_key=" << hashed_key;
  if (w_map_.find(hashed_key) == w_map_.end()) {
    const float* cpu_ptr = nullptr;
    std::vector<float> transpose_data(numel, 0);
    if (data_transpose) {
      CHECK(dims.size() == 2) << "Not support: dims.size = " << dims.size();
      paddle::lite::xpu::math::Transpose(
          cpu_data, transpose_data.data(), dims[0], dims[1]);
      cpu_ptr = transpose_data.data();
    } else {
      cpu_ptr = cpu_data;
    }

    XPUScratchPadGuard weight_max_guard;
    XPUScratchPadGuard quant_weight_guard;

    float max_val = paddle::lite::xpu::math::FindMaxAbs(cpu_ptr, numel);
    int max_ptr_size = xdnn::get_max_ptr_size(GetRawContext());
    std::vector<float> max_vec(max_ptr_size, max_val);
    weight_max_guard =
        std::move(MallocScratchPad(max_ptr_size * sizeof(float)));
    MemcpySync(weight_max_guard->addr_,
               max_vec.data(),
               max_ptr_size * sizeof(float),
               IoDirection::HtoD);

    if (precision == "int31") {
      quant_weight_guard = std::move(MallocScratchPad(numel * sizeof(float)));
      MemcpySync(quant_weight_guard->addr_,
                 cpu_ptr,
                 numel * sizeof(float),
                 IoDirection::HtoD);
    } else if (precision == "int16") {
      quant_weight_guard = std::move(MallocScratchPad(numel * sizeof(int16_t)));
      std::vector<int16_t> quant_data_cpu(numel, 0);
      paddle::lite::xpu::math::ConvertFP32ToInt16(
          cpu_ptr, quant_data_cpu.data(), max_val, numel);
      MemcpySync(quant_weight_guard->addr_,
                 quant_data_cpu.data(),
                 numel * sizeof(int16_t),
                 IoDirection::HtoD);
    } else if (precision == "int8") {
      quant_weight_guard = std::move(MallocScratchPad(numel * sizeof(int8_t)));
      std::vector<int8_t> quant_data_cpu(numel, 0);
      paddle::lite::xpu::math::ConvertFP32ToInt8(
          cpu_ptr, quant_data_cpu.data(), max_val, numel);
      MemcpySync(quant_weight_guard->addr_,
                 quant_data_cpu.data(),
                 numel * sizeof(int8_t),
                 IoDirection::HtoD);
    } else {
      CHECK(false) << "Unknown precision: " << precision;
    }

    w_map_[hashed_key] = std::make_pair(std::move(weight_max_guard),
                                        std::move(quant_weight_guard));
  }

  float* max_ptr = reinterpret_cast<float*>(w_map_[hashed_key].first->addr_);
  void* qdata_ptr = w_map_[hashed_key].second->addr_;
  XPUQuantData xpu_qdata(max_ptr, qdata_ptr);

  return xpu_qdata;
}

void TargetWrapperXPU::ScatterL3Cache(
    void* l3_ptr,
    size_t l3_size,
    const std::vector<std::vector<int64_t>>& query_shape) {
  if (l3_size == 0 || l3_ptr == nullptr) {
    LOG(FATAL) << "Unable to scatter l3 cache, l3 cache size is: " << l3_size
               << ", l3 addr is: " << l3_ptr;
  }
  CHECK(l3_planner_);
  size_t total_block_l3_size = 0, xdnn_ctx_l3_size = 0;
  l3_planner_->set_current_query_shape(query_shape, l3_size);
  std::vector<size_t>* plan = l3_planner_->get_current_plan();
  if (plan == nullptr) {
    XPU_CALL(tls_raw_ctx_->_l3_mgr.set(l3_ptr, l3_size));
    VLOG(3) << "XDNN CTX L3 Size is " << tls_raw_ctx_->_l3_mgr.get_size()
            << ", Remain L3 Size for Lite is " << 0;
  } else {
    CHECK_EQ(plan->size(), l3_block_dict.size() + 1);
    xdnn_ctx_l3_size = plan->back();
    for (size_t i = 0; i < l3_block_dict.size(); i++) {
      size_t cur_block_size = plan->data()[i];
      if (cur_block_size > 0) {
        l3_block_dict[i]->set(
            reinterpret_cast<int8_t*>(l3_ptr) + total_block_l3_size,
            cur_block_size);
        total_block_l3_size += cur_block_size;
      }
    }
    if (xdnn_ctx_l3_size > 0) {
      XPU_CALL(tls_raw_ctx_->_l3_mgr.set(
          reinterpret_cast<int8_t*>(l3_ptr) + l3_size - xdnn_ctx_l3_size,
          xdnn_ctx_l3_size));
    }
    VLOG(3) << "XDNN CTX L3 Size is " << tls_raw_ctx_->_l3_mgr.get_size()
            << ", Remain L3 Size for Lite is " << total_block_l3_size;
  }
}

void TargetWrapperXPU::MallocL3Cache(
    const std::vector<std::vector<int64_t>>& query_shape) {
  TargetWrapperXPU::GetRawContext();
  // malloc shared l3
  if (!TargetWrapperXPU::IsSharedL3Created() && shared_l3_size > 0) {
    mutex_l3_.lock();
    if (!TargetWrapperXPU::IsSharedL3Created()) {
      XPU_CALL(xpu_malloc(reinterpret_cast<void**>(&shared_l3_ptr_),
                          shared_l3_size,
                          XPU_MEM_L3));
    }
    mutex_l3_.unlock();
  }
  if (local_l3_size != 0) {
    // malloc local_l3
    VLOG(3) << "Try To Malloc Local L3 Cache Size is" << local_l3_size;
    int ret = xpu_malloc(
        reinterpret_cast<void**>(&local_l3_ptr_), local_l3_size, XPU_MEM_L3);
    if (ret != 0) {
      VLOG(3) << "No Enough L3 Cache For Current Predictor.";
      local_l3_ptr_ = nullptr;
    } else {
      VLOG(3) << "Success!";
      TargetWrapperXPU::ScatterL3Cache(
          local_l3_ptr_, local_l3_size, query_shape);
    }
  } else if (need_l3_mutex && TargetWrapperXPU::IsSharedL3Created()) {
    // lock and use shared_l3
    mutex_l3_.lock();
    TargetWrapperXPU::ScatterL3Cache(
        shared_l3_ptr_, shared_l3_size, query_shape);
  }
}

void TargetWrapperXPU::FreeL3Cache() {
  VLOG(3) << "raw_ctx gm_mgr size=" << tls_raw_ctx_->_gm_mgr.get_size();
  VLOG(3) << "BEGIN PRINT MEMORY INFOS";
  for (size_t block_idx = 0; block_idx < l3_block_dict.size(); block_idx++) {
    XPUL3CacheBlock* cur_block = l3_block_dict[block_idx];
    std::vector<size_t>& history = cur_block->history_;
    int history_size = history.size();
    std::sort(history.begin(), history.end());
    size_t max_size = 0;
    std::stringstream ss;
    ss << "Size History: ";
    for (int i = 0; i < history_size - 1; i++) {
      ss << history[i] << ", ";
      max_size = std::max(max_size, history[i]);
    }
    if (history_size > 0) {
      ss << history[history_size - 1];
      max_size = std::max(max_size, history[history_size - 1]);
    }
    VLOG(3) << "Block Idx is " << block_idx << ", history_size=" << history_size
            << ", cur max size=" << max_size;
    VLOG(3) << ss.str();
  }
  if (local_l3_size != 0) {
    if (local_l3_ptr_ != nullptr) {
      TargetWrapperXPU::Free(local_l3_ptr_);
      local_l3_ptr_ = nullptr;
      XPU_CALL(tls_raw_ctx_->_l3_mgr.set(nullptr, 0));
    }
    l3_planner_->run_autotune(l3_block_dict, local_l3_size);
  } else if (need_l3_mutex && TargetWrapperXPU::IsSharedL3Created()) {
    XPU_CALL(xpu_wait());
    XPU_CALL(tls_raw_ctx_->_l3_mgr.set(nullptr, 0));
    mutex_l3_.unlock();
    l3_planner_->run_autotune(l3_block_dict, shared_l3_size);
  }
  for (size_t i = 0; i < l3_block_dict.size(); i++) {
    l3_block_dict[i]->clear();
  }
}

// xpu context
LITE_THREAD_LOCAL xdnn::Context* TargetWrapperXPU::tls_raw_ctx_{nullptr};
// multi encoder config
LITE_THREAD_LOCAL std::string
    TargetWrapperXPU::multi_encoder_precision;  // NOLINT
LITE_THREAD_LOCAL bool TargetWrapperXPU::multi_encoder_adaptive_seqlen{false};
// conv autotune config
LITE_THREAD_LOCAL bool TargetWrapperXPU::conv_autotune{false};
LITE_THREAD_LOCAL std::string TargetWrapperXPU::conv_autotune_file;
// l3 cache config
LITE_THREAD_LOCAL bool TargetWrapperXPU::need_l3_mutex{false};
LITE_THREAD_LOCAL size_t TargetWrapperXPU::local_l3_size{
    std::numeric_limits<size_t>::max()};
LITE_THREAD_LOCAL size_t TargetWrapperXPU::local_gm_size{
    0x4000000};  // 64 * 1024 * 1024
LITE_THREAD_LOCAL void* TargetWrapperXPU::local_l3_ptr_{nullptr};
void* TargetWrapperXPU::shared_l3_ptr_{nullptr};
size_t TargetWrapperXPU::shared_l3_size{0};
LITE_THREAD_LOCAL std::vector<XPUL3CacheBlock*> TargetWrapperXPU::l3_block_dict;
// l3 mutex
std::mutex TargetWrapperXPU::mutex_l3_;
// l3 planner
LITE_THREAD_LOCAL XPUL3Planner* TargetWrapperXPU::l3_planner_{nullptr};
// cpu data to xpu quant data
LITE_THREAD_LOCAL
std::unordered_map<size_t, std::pair<XPUScratchPadGuard, XPUScratchPadGuard>>
    TargetWrapperXPU::w_map_;

}  // namespace lite
}  // namespace paddle
