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
#include <vector>
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {

__inline__ size_t align_by_4(float in) {
  return static_cast<size_t>(in / 4) * 4;
}

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
  CHECK(ptr != nullptr) << "XPU Malloc Fail, Malloc Size is: " << size;
  return XPUScratchPadGuard(new XPUScratchPad(ptr, size));
}

void TargetWrapperXPU::ScatterL3Cache(void* l3_ptr, size_t l3_size) {
  if (l3_size == 0 || l3_ptr == nullptr) {
    LOG(FATAL) << "Unable to scatter l3 cache, l3 cache size is: " << l3_size
               << ", l3 addr is: " << l3_ptr;
  }
  if (l3_size <= inside_kernel_l3_size) {
    XPU_CALL(tls_raw_ctx_->_l3_mgr.set(l3_ptr, l3_size));
    for (auto it = l3_block_dict.begin(); it != l3_block_dict.end(); ++it) {
      XPUL3CacheBlock& cur_block = it->second;
      cur_block.clear();
    }
    VLOG(3) << "Inside kernel l3 size is: " << tls_raw_ctx_->_l3_mgr.get_size()
            << ", l3 size between op layers is 0.";
  } else {
    if (inside_kernel_l3_size == 0) {
      XPU_CALL(tls_raw_ctx_->_l3_mgr.set(nullptr, 0));
    } else {
      XPU_CALL(
          tls_raw_ctx_->_l3_mgr.set(l3_ptr, align_by_4(inside_kernel_l3_size)));
    }
    VLOG(3) << "Inside kernel l3 size is: " << tls_raw_ctx_->_l3_mgr.get_size();
    size_t block_l3_size = l3_size - align_by_4(inside_kernel_l3_size);
    size_t remain_l3_size = block_l3_size;
    uint8_t* remain_l3_ptr =
        reinterpret_cast<uint8_t*>(l3_ptr) + align_by_4(inside_kernel_l3_size);
    for (auto it = l3_block_dict.begin(); it != l3_block_dict.end(); ++it) {
      XPUL3CacheBlock* cur_block = &(it->second);
      size_t cur_size = align_by_4(cur_block->proportion() * block_l3_size);
      cur_size = std::min(cur_size, remain_l3_size);
      if (cur_size > 0) {
        cur_block->set(reinterpret_cast<void*>(remain_l3_ptr), cur_size);
      } else {
        cur_block->clear();
      }
      remain_l3_ptr = remain_l3_ptr + cur_size;
      remain_l3_size = remain_l3_size - cur_size;
      VLOG(3) << "Tensor(" << it->first << ") l3 size is " << it->second.size()
              << ", proportion is " << it->second.proportion();
    }
  }
}

void TargetWrapperXPU::MallocL3Cache() {
  TargetWrapperXPU::GetRawContext();
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
    VLOG(3) << "Try To Malloc Local L3 Cache Size is" << local_l3_size;
    int ret = xpu_malloc(
        reinterpret_cast<void**>(&local_l3_ptr_), local_l3_size, XPU_MEM_L3);
    if (ret != 0) {
      VLOG(3) << "No Enough L3 Cache For Current Predictor.";
    } else {
      VLOG(3) << "Success!";
      TargetWrapperXPU::ScatterL3Cache(local_l3_ptr_, local_l3_size);
    }
  } else if (TargetWrapperXPU::IsSharedL3Created()) {
    mutex_l3_.lock();
    TargetWrapperXPU::ScatterL3Cache(shared_l3_ptr_, shared_l3_size);
  }
}

void TargetWrapperXPU::FreeL3Cache() {
  if (local_l3_size != 0) {
    XPU_CALL(xpu_wait());
    XPU_CALL(xpu_free(local_l3_ptr_));
    XPU_CALL(tls_raw_ctx_->_l3_mgr.set(nullptr, 0));
  } else if (TargetWrapperXPU::IsSharedL3Created()) {
    XPU_CALL(xpu_wait());
    XPU_CALL(tls_raw_ctx_->_l3_mgr.set(nullptr, 0));
    mutex_l3_.unlock();
  }
  if (local_l3_size != 0 || TargetWrapperXPU::IsSharedL3Created()) {
    if (TargetWrapperXPU::l3_block_autotune_lr >= 0.25) {
      TargetWrapperXPU::l3_block_autotune_lr *= 0.9;
      TargetWrapperXPU::AutoTuneL3BlockSize();
    }
    for (auto it = l3_block_dict.begin(); it != l3_block_dict.end(); ++it) {
      XPUL3CacheBlock* cur_block = &(it->second);
      cur_block->clear();
    }
  }
}

void TargetWrapperXPU::AutoTuneL3BlockSize() {
  if (l3_block_dict.empty()) {
    return;
  }
  LOG(INFO) << "AutoTune XPU L3 Cache Block Start.";
  size_t block_l3_size = TargetWrapperXPU::IsSharedL3Created()
                             ? shared_l3_size - inside_kernel_l3_size
                             : local_l3_size - inside_kernel_l3_size;
  // Here is a trick.
  // Since each memory size is computed by proportion * block_l3_size
  // and each memory size need to be aligned by 4 bytes, it is possible that
  // final size is smaller than
  // what it really need. So a de-factor is added to block_l3_size
  block_l3_size = block_l3_size - 20;
  struct node {
    float weights = 0.0f;
    size_t scores = 0;
    std::vector<float> choices{0.0f};
  };
  std::vector<std::vector<node>> records;
  for (auto it = l3_block_dict.begin(); it != l3_block_dict.end(); ++it) {
    VLOG(3) << "Nodes in XPU L3 Block Tensor(" << it->first << "):";
    XPUL3CacheBlock* cur_block = &(it->second);
    auto history = cur_block->history_;
    std::sort(history.begin(), history.end());
    std::vector<node> block_nodes{node()};
    size_t score = 0;
    for (size_t i = 0; i < history.size();) {
      int j = i;
      while (j < history.size() && history[i] == history[j]) {
        score += history[j];
        j++;
      }
      node cur_node;
      cur_node.weights =
          static_cast<float>(history[i]) / static_cast<float>(block_l3_size);
      if (cur_node.weights > 1.0f) {
        break;
      }
      cur_node.scores = score;
      cur_node.choices = {cur_node.weights};
      block_nodes.push_back(cur_node);
      i = j;
      VLOG(3) << "Node Weights is:" << cur_node.weights
              << ", Node Scores is: " << score;
    }
    records.push_back(block_nodes);
  }

  std::vector<node> res(records[0]);
  for (size_t block_idx = 1; block_idx < records.size(); block_idx++) {
    std::vector<node> new_nodes;
    for (size_t node_idx = 0; node_idx < records[block_idx].size();
         node_idx++) {
      for (size_t res_idx = 0; res_idx < res.size(); res_idx++) {
        node cur_node;
        float cur_weights =
            records[block_idx][node_idx].weights + res[res_idx].weights;
        if (cur_weights > 1.0f) {
          break;
        }
        cur_node.scores =
            records[block_idx][node_idx].scores + res[res_idx].scores;
        cur_node.weights = cur_weights;
        cur_node.choices = res[res_idx].choices;
        cur_node.choices.push_back(records[block_idx][node_idx].choices[0]);
        new_nodes.push_back(cur_node);
      }
    }
    struct {
      bool operator()(node a, node b) const {
        if (a.weights < b.weights) {
          return true;
        } else if (a.weights == b.weights) {
          return a.scores > b.scores;
        } else {
          return false;
        }
      }
    } customLess;

    std::sort(new_nodes.begin(), new_nodes.end(), customLess);
    std::vector<bool> stay(new_nodes.size(), true);
    for (int i = new_nodes.size() - 1; i >= 0; i--) {
      for (int j = i - 1; j >= 0; j--) {
        if (new_nodes[j].scores >= new_nodes[i].scores) {
          stay[i] = false;
          break;
        }
      }
    }
    res.clear();
    for (int i = 0; i < new_nodes.size(); i++) {
      if (stay[i] == true) {
        res.push_back(new_nodes[i]);
      }
    }
    VLOG(3) << "XPU L3 Block IDX is " << block_idx
            << ", Choices before filter are " << new_nodes.size()
            << ", Choices after filter are " << res.size();
  }

  float total_proportion = 0.0f;
  for (int i = 0; i < res.back().choices.size(); i++) {
    total_proportion += res.back().choices[i];
  }
  int idx = 0;
  for (auto it = l3_block_dict.begin(); it != l3_block_dict.end(); ++it) {
    auto before = it->second.proportion();
    auto after = res.back().choices[idx] / total_proportion;
    auto diff = (after - before) * TargetWrapperXPU::l3_block_autotune_lr;
    it->second.init(before + diff);
    LOG(INFO) << "Tensor( " << it->first << ") Proportion before is " << before
              << ", after autotune is " << it->second.proportion();
    idx++;
  }
  LOG(INFO) << "AutoTune XPU L3 Cache Block End.";
}

// xpu context
LITE_THREAD_LOCAL xdnn::Context* TargetWrapperXPU::tls_raw_ctx_{nullptr};
// multi encoder config
LITE_THREAD_LOCAL std::string
    TargetWrapperXPU::multi_encoder_precision;  // NOLINT
LITE_THREAD_LOCAL bool TargetWrapperXPU::multi_encoder_adaptive_seqlen{false};
// conv autotune config
LITE_THREAD_LOCAL bool TargetWrapperXPU::conv_autotune{true};
LITE_THREAD_LOCAL std::string TargetWrapperXPU::conv_autotune_file{
    "/opt/xpu_conv_autotune"};  // NOLINT
// l3 cache config
LITE_THREAD_LOCAL size_t TargetWrapperXPU::local_l3_size{0xfffc00};
LITE_THREAD_LOCAL void* TargetWrapperXPU::local_l3_ptr_{nullptr};
void* TargetWrapperXPU::shared_l3_ptr_{nullptr};
size_t TargetWrapperXPU::shared_l3_size{0};

LITE_THREAD_LOCAL std::map<std::string, XPUL3CacheBlock>
    TargetWrapperXPU::l3_block_dict;
LITE_THREAD_LOCAL size_t TargetWrapperXPU::inside_kernel_l3_size{0xfffc00};
LITE_THREAD_LOCAL float TargetWrapperXPU::l3_block_autotune_lr{0.5};

std::mutex TargetWrapperXPU::mutex_l3_;

}  // namespace lite
}  // namespace paddle
