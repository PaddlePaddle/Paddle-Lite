// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace xpu {
namespace vec {

template <typename T>
static std::vector<T> Vec2DTo1D(const std::vector<std::vector<T>>& vec) {
  std::vector<T> res;
  for (const auto& v : vec) {
    for (const auto& ele : v) {
      res.emplace_back(ele);
    }
  }
  return res;
}

template <typename T>
static std::vector<std::vector<T>> Vec1DTo2D(const std::vector<T>& vec,
                                             int dim) {
  std::vector<std::vector<T>> res;
  for (size_t i = 0; i < vec.size(); i += dim) {
    std::vector<T> tmp;
    for (size_t j = 0; j < dim; j++) {
      tmp.push_back(vec[i + j]);
    }
    res.emplace_back(std::move(tmp));
  }
  return res;
}

template <typename T>
static std::vector<std::vector<std::vector<T>>> Vec1DTo3D(
    const std::vector<T>& vec, std::vector<int> has_extra, int dim) {
  std::vector<std::vector<std::vector<T>>> res;
  std::vector<T> buff;
  std::vector<int> step;
  step.resize(has_extra.size());

  int start = 0;
  for (int i = 0; i < step.size(); i++) {
    buff.clear();
    step[i] = (has_extra[i] + 2) * dim;
    if (i == 0)
      start = 0;
    else
      start += step[i - 1];
    for (int j = start; j < start + step[i]; j++) {
      buff.push_back(vec[j]);
    }
    res.push_back(Vec1DTo2D(buff, dim));
  }

  return res;
}

template <typename T>
static std::vector<std::vector<T>> Vec3DTo2D(
    const std::vector<std::vector<std::vector<T>>>& vec) {
  std::vector<std::vector<T>> res;
  for (int i = 0; i < vec.size(); i++) {
    std::vector<T> buff;
    for (int j = 0; j < vec[i].size(); j++) {
      for (int k = 0; k < vec[i][j].size(); k++) {
        buff.push_back(vec[i][j][k]);
      }
    }
    res.push_back(buff);
  }
  return res;
}

template <typename T>
static std::vector<std::vector<T>> Vec1DTo2DWithIdx(
    const std::vector<T>& vec,
    const std::vector<int>& s_idx,
    const std::vector<int>& e_idx) {
  std::vector<std::vector<T>> res;
  for (int i = 0; i < s_idx.size(); i++) {
    std::vector<T> tmp;
    for (int j = s_idx[i]; j <= e_idx[i]; j++) {
      tmp.push_back(vec[j]);
    }
    res.push_back(tmp);
  }
  return res;
}

template <typename T>
static std::vector<std::vector<std::vector<T>>> Vec1DTo3DWithExtraInfo(
    const std::vector<T>& vec,
    const std::vector<std::vector<int>>& extra_info,
    int offset,
    bool is_conv) {
  int tmp_idx = 0;
  std::vector<std::vector<std::vector<T>>> res;
  for (int i = 0; i < extra_info.size(); ++i) {
    std::vector<std::vector<T>> tmp_1;
    for (int j = 0; j < extra_info[i].size(); ++j) {
      std::vector<T> tmp_2;
      int tmp_idx_buff = tmp_idx;
      if (is_conv) {
        tmp_idx += (offset + extra_info[i][j]);
      } else {
        tmp_idx += offset;
      }
      for (int k = tmp_idx_buff; k < tmp_idx; k++) {
        tmp_2.push_back(vec[k]);
      }
      tmp_1.push_back(tmp_2);
    }
    res.push_back(tmp_1);
  }
  return res;
}

template <typename T>
static std::vector<std::vector<std::vector<std::vector<T>>>>
Vec1DTo4DWithExtraInfo(const std::vector<T>& vec,
                       const std::vector<std::vector<T>>& extra_info,
                       int offset,
                       int dim,
                       bool is_conv = true) {
  std::vector<std::vector<std::vector<std::vector<T>>>> res;
  int tmp_idx = 0;
  for (int i = 0; i < extra_info.size(); ++i) {
    std::vector<std::vector<std::vector<T>>> tmp_1;
    for (int j = 0; j < extra_info[i].size(); ++j) {
      std::vector<std::vector<T>> tmp_2;
      int tmp_idx_buff = tmp_idx;
      if (is_conv) {
        tmp_idx += (offset + extra_info[i][j]) * dim;
      } else {
        tmp_idx += offset * dim;
      }

      for (int k = tmp_idx_buff; k < tmp_idx; k += dim) {
        std::vector<T> tmp_3;
        for (int l = k; l < k + dim; l++) {
          tmp_3.push_back(vec[l]);
        }
        tmp_2.push_back(tmp_3);
      }
      tmp_1.push_back(tmp_2);
    }
    res.push_back(tmp_1);
  }
  return res;
}

}  // namespace vec
}  // namespace xpu
}  // namespace lite
}  // namespace paddle
