// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/unique_with_counts_compute.h"

#include <unordered_map>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {
template <typename InT>
void UniqueFunc_int32(const lite::Tensor* x,
                      lite::Tensor* out,
                      lite::Tensor* index,
                      lite::Tensor* count) {
  const InT* in_data = x->template data<InT>();
  auto in_dim = x->dims();
  int32_t* index_data = index->mutable_data<int32_t>();
  std::unordered_map<InT, int64_t> dict;
  std::vector<InT> uniq;
  int64_t j = 0;

  for (auto i = 0; i < x->numel(); i++) {
    auto it = dict.find(in_data[i]);
    if (it == dict.end()) {
      dict.emplace(std::make_pair(in_data[i], j));
      uniq.emplace_back(in_data[i]);
      index_data[i] = static_cast<int32_t>(j);
      j++;
    } else {
      index_data[i] = static_cast<int32_t>(it->second);
    }
  }

  if (count != nullptr) {
    // Resize the count tensor dims to allocate the memory
    count->Resize({static_cast<int64_t>(uniq.size())});
    int32_t* count_data = count->template mutable_data<int32_t>();
    // init count_data to 0
    memset(count_data, 0, uniq.size() * sizeof(int32_t));
    for (auto i = 0; i < x->numel(); ++i) {
      const int32_t& index = index_data[i];
      count_data[index] += static_cast<int32_t>(1);
    }
  }
  out->Resize({static_cast<int64_t>(uniq.size())});
  auto out_data = out->mutable_data<InT>();
  std::memcpy(out_data, uniq.data(), uniq.size() * sizeof(InT));
}

template <typename InT>
void UniqueFunc_int64(const lite::Tensor* x,
                      lite::Tensor* out,
                      lite::Tensor* index,
                      lite::Tensor* count) {
  const InT* in_data = x->template data<InT>();
  auto in_dim = x->dims();
  int64_t* index_data = index->mutable_data<int64_t>();
  std::unordered_map<InT, int64_t> dict;
  std::vector<InT> uniq;
  int64_t j = 0;

  for (auto i = 0; i < x->numel(); i++) {
    auto it = dict.find(in_data[i]);
    if (it == dict.end()) {
      dict.emplace(std::make_pair(in_data[i], j));
      uniq.emplace_back(in_data[i]);
      index_data[i] = static_cast<int64_t>(j);
      j++;
    } else {
      index_data[i] = static_cast<int64_t>(it->second);
    }
  }

  if (count != nullptr) {
    // Resize the count tensor dims to allocate the memory
    count->Resize({static_cast<int64_t>(uniq.size())});
    int64_t* count_data = count->template mutable_data<int64_t>();
    // init count_data to 0
    memset(count_data, 0, uniq.size() * sizeof(int64_t));

    for (auto i = 0; i < x->numel(); ++i) {
      const int64_t& index = index_data[i];
      count_data[index] += static_cast<int64_t>(1);
    }
  }
  out->Resize({static_cast<int64_t>(uniq.size())});
  auto out_data = out->mutable_data<InT>();
  std::memcpy(out_data, uniq.data(), uniq.size() * sizeof(InT));
}

void UniqueWithCountsCompute::Run() {
  auto& param = Param<operators::UniqueWithCountsParam>();
  auto x = param.X;
  auto output = param.Out;
  auto index = param.Index;
  auto count = param.Count;
  auto in_dims = x->dims();
  lite_api::PrecisionType index_type = index->precision();
  bool index_type_match =
      index_type == PRECISION(kInt32) || index_type == PRECISION(kInt64);
  lite_api::PrecisionType type = x->precision();
  CHECK_EQ(index_type_match, true) << "Index holds the wrong type, it holds "
                                   << static_cast<int>(type)
                                   << "but desires to be int32 or int64";
  if (index_type == PRECISION(kInt32)) {
    switch (type) {
      case PRECISION(kFloat):
        UniqueFunc_int32<float>(x, output, index, count);
        break;
      case PRECISION(kInt32):
        UniqueFunc_int32<int32_t>(x, output, index, count);
        break;
      case PRECISION(kInt64):
        UniqueFunc_int32<int64_t>(x, output, index, count);
        break;
      default:
        LOG(FATAL) << "unique_with_counts does not implement for the "
                   << "input type:" << static_cast<int>(type);
    }
  } else {
    switch (type) {
      case PRECISION(kFloat):
        UniqueFunc_int64<float>(x, output, index, count);
        break;
      case PRECISION(kInt32):
        UniqueFunc_int64<int32_t>(x, output, index, count);
        break;
      case PRECISION(kInt64):
        UniqueFunc_int64<int64_t>(x, output, index, count);
        break;
      default:
        LOG(FATAL) << "unique_with_counts does not implement for the "
                   << "input type:" << static_cast<int>(type);
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(unique_with_counts,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::UniqueWithCountsCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("Index",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .BindOutput("Count",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();
