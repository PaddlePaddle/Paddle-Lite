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

#include "framework/lod_tensor.h"
#include <algorithm>

namespace paddle_mobile {
namespace framework {

LoD SliceInLevel(const LoD &in, size_t level, size_t elem_begin,
                 size_t elem_end) {
  PADDLE_MOBILE_ENFORCE(level < in.size(), "level should >= in.size()");
  PADDLE_MOBILE_ENFORCE(elem_end < in[level].size(),
                        "elem_end >= in[level].size()");
  LoD res;
  res.resize(in.size() - level);
  // copy the first level
  res[0].assign(in[level].begin() + elem_begin,
                in[level].begin() + elem_end + 1);
  for (size_t lvl = 1; lvl < res.size(); lvl++) {
    const auto &in_level = in[level + lvl];
    const auto &above_level = res[lvl - 1];
    auto &out_level = res[lvl];
    out_level.assign(in_level.begin() + above_level.front(),
                     in_level.begin() + above_level.back() + 1);
  }
  for (size_t lvl = 0; lvl < res.size(); lvl++) {
    // to make the first offset equals 0, all the elements minus the
    // first
    // element
    size_t front = res[lvl].front();
    for (auto &ele : res[lvl]) {
      ele -= front;
    }
  }
  return res;
}

LoD ToAbsOffset(const LoD &in) {
  // the lowest level stores relative offsets
  if (in.empty() || in.size() == 1) return in;
  LoD result = in;
  for (auto level = static_cast<int>(in.size() - 2); level >= 0; level--) {
    for (size_t i = 0; i < in[level].size(); ++i) {
      size_t index = in[level][i];
      result[level][i] = result[level + 1][index];
    }
  }
  return result;
}

bool operator==(const LoD &a, const LoD &b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); i++) {
    const auto &a_level = a[i];
    const auto &b_level = b[i];
    if (a_level.size() != b_level.size()) {
      return false;
    }
    for (size_t j = 0; j < a_level.size(); j++) {
      if (a_level[j] != b_level[j]) {
        return false;
      }
    }
  }
  return true;
}

bool CheckLoD(const LoD &in, int tensor_height) {
  if (in.empty()) return true;
  for (const auto &level : in) {
    // check: there should be more than 2 offsets existing in each
    // level.
    if (level.size() < 2) return false;
    // check: the first offset(the begin offset) of each level
    // should be 0.
    if (level.front() != 0) return false;
    // check: all the offsets in a level should be ascending(no same
    // items
    // allows).
    if (!std::is_sorted(level.begin(), level.begin(), [](size_t a, size_t b) {
          if (a < b) return true;
          return false;
        })) {
      PADDLE_MOBILE_THROW_EXCEPTION("ascending error")
      return false;
    }
  }
  // check: the lowest level's last offset should equals
  // `tensor_height` if
  //        tensor_height>0.
  if (tensor_height > 0 && (size_t)tensor_height != in.back().back())
    return false;

  // check: the higher level's last offset should equals the lower
  // level's
  // size-1.
  // NOTE LoD store the levels from top to bottom, so the higher level
  // goes
  // first.
  for (size_t level = 0; level < in.size() - 1; level++) {
    if (in[level].back() != in[level + 1].size() - 1) return false;
  }
  return true;
}

bool CheckAbsLoD(const LoD &in, int tensor_height) {
  if (in.empty()) return true;
  for (const auto &level : in) {
    // check: all the offsets in a level should be ascending(no same
    // items
    // allows).
    if (!std::is_sorted(level.begin(), level.begin(), [](size_t a, size_t b) {
          if (a < b) return true;
          return false;
        })) {
      return false;
    }

    // check: there should be more than 2 offsets existing in each
    // level.
    if (level.size() < 2) return false;

    // check: the first offset of each level should be 0, and the
    // last should be
    // the same(the height of underlying tensor).
    if (level.front() != 0) return false;
    if (tensor_height < 0) {
      tensor_height = level.back();
    } else if ((size_t)tensor_height != level.back()) {
      return false;
    }
  }
  return true;
}

using LoDAndOffset = std::pair<LoD, std::pair<size_t, size_t>>;

LoDAndOffset GetSubLoDAndAbsoluteOffset(const LoD &lod, size_t start_idx,
                                        size_t end_idx, size_t start_level) {
  LoD sub_lod;

  for (size_t level_idx = start_level; level_idx < lod.size(); ++level_idx) {
    PADDLE_MOBILE_ENFORCE(start_idx <= end_idx, "start_idx > end_idx");
    PADDLE_MOBILE_ENFORCE(end_idx < lod[level_idx].size(),
                          "end_idx >= lod[level_idx].size()");
    std::vector<size_t> level_lens;
    for (size_t i = start_idx; i < end_idx; ++i) {
      level_lens.push_back(lod[level_idx][i + 1] - lod[level_idx][i]);
    }
    sub_lod.emplace_back(level_lens);
    start_idx = lod[level_idx][start_idx];
    end_idx = lod[level_idx][end_idx];
  }

  return LoDAndOffset{sub_lod, {start_idx, end_idx}};
}

void AppendLoD(LoD *lod, const LoD &lod_length) {
  PADDLE_MOBILE_ENFORCE(
      lod->empty() || lod->size() == lod_length.size(),
      "The lod_length should has the same size with the appended lod.");
  if (lod->empty()) {
    for (size_t i = 0; i < lod_length.size(); ++i) {
      lod->emplace_back(1, 0);  // size = 1, value = 0;
    }
    *lod = LoD(lod_length.size(), std::vector<size_t>({0}));
  }
  for (size_t i = 0; i < lod->size(); ++i) {
    auto &level = (*lod)[i];
    for (size_t len : lod_length[i]) {
      level.push_back(level.back() + len);
    }
  }
}

}  // namespace framework
}  // namespace paddle_mobile
