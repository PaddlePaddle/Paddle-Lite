// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/unique_compute.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename InT, typename IndexT>
void UniqueFunc(const lite::Tensor* x,
                lite::Tensor* out,
                lite::Tensor* index,
                lite::Tensor* count) {
  const InT* in_data = x->template data<InT>();
  IndexT* index_data = index->mutable_data<IndexT>();
  int64_t j = 0;
  std::unordered_map<InT, int64_t> dict;
  std::vector<InT> uniq;
  for (auto i = 0; i < x->numel(); i++) {
    auto it = dict.find(in_data[i]);
    if (it == dict.end()) {
      dict.emplace(std::make_pair(in_data[i], j));
      uniq.emplace_back(in_data[i]);
      index_data[i] = static_cast<IndexT>(j);
      j++;
    } else {
      index_data[i] = static_cast<IndexT>(it->second);
    }
  }
  if (count != nullptr) {
    // Resize the count tensor dims to allocate the memory
    count->Resize({static_cast<int64_t>(uniq.size())});
    IndexT* count_data = count->template mutable_data<IndexT>();
    // init count_data to 0
    memset(count_data, 0, uniq.size() * sizeof(IndexT));
    auto index_type = index->precision();
    bool index_type_match =
        (index_type == PRECISION(kInt64) || index_type == PRECISION(kInt32));
    CHECK(index_type_match) << "index type must be int32 or int64, but now is "
                            << static_cast<int>(index_type);
    for (auto i = 0; i < x->numel(); ++i) {
      const IndexT& index = index_data[i];
      count_data[index] += static_cast<IndexT>(1);
    }
  }
  out->Resize({static_cast<int64_t>(uniq.size())});
  auto out_data = out->mutable_data<InT>();
  std::memcpy(out_data, uniq.data(), uniq.size() * sizeof(InT));
}

template <typename InT, typename IndexT>
void UniqueFlattendTensorFunc(const lite::Tensor& in,
                              lite::Tensor* out,
                              lite::Tensor* index,
                              lite::Tensor* indices,
                              lite::Tensor* count,
                              bool return_index,
                              bool return_inverse,
                              bool return_counts) {
  const InT* in_data = in.data<InT>();
  std::set<InT> unique(in_data, in_data + in.numel());
  out->Resize({static_cast<int64_t>(unique.size())});
  auto out_data = out->mutable_data<InT>();
  std::copy(unique.begin(), unique.end(), out_data);
  if (return_index) {
    indices->Resize({out->numel()});
    auto indices_data = indices->mutable_data<IndexT>();
    std::unordered_map<InT, IndexT> indices_map;
    indices_map.reserve(out->numel());
    for (int64_t i = 0; i < in.numel(); ++i) {
      if (indices_map.find(in_data[i]) != indices_map.end()) continue;
      indices_map[in_data[i]] = i;
    }
    for (int64_t i = 0; i < out->numel(); ++i) {
      indices_data[i] = indices_map[out_data[i]];
    }
  }
  if (return_inverse) {
    index->Resize({in.numel()});
    auto inverse_data = index->mutable_data<IndexT>();
    std::unordered_map<InT, IndexT> inverse_map;
    inverse_map.reserve(out->numel());
    for (int64_t i = 0; i < out->numel(); ++i) {
      inverse_map[out_data[i]] = i;
    }
    for (int64_t i = 0; i < in.numel(); ++i) {
      inverse_data[i] = inverse_map[in_data[i]];
    }
  }
  if (return_counts) {
    count->Resize({out->numel()});
    auto count_data = count->mutable_data<IndexT>();
    std::unordered_map<InT, IndexT> counts_map;
    counts_map.reserve(out->numel());
    for (int64_t i = 0; i < out->numel(); ++i) {
      counts_map[out_data[i]] = 0;
    }
    for (int64_t i = 0; i < in.numel(); ++i) {
      counts_map[in_data[i]] += 1;
    }
    for (int64_t i = 0; i < out->numel(); ++i) {
      count_data[i] = counts_map[out_data[i]];
    }
  }
}

template <typename T>
void UniqueTransCompute(const Tensor& input,
                        Tensor* output,
                        const std::vector<int>& orders) {
  auto in_dims = input.dims();
  auto out_dims = output->dims();
  int num_axes = in_dims.size();
  int count = in_dims.production();
  const T* din = input.data<T>();
  T* dout = output->mutable_data<T>();
  std::vector<int> old_steps;
  int temp = 1;
  for (int i = 0; i < num_axes; ++i) {
    old_steps.push_back(temp);
    temp *= in_dims[num_axes - 1 - i];
  }
  std::reverse(old_steps.begin(), old_steps.end());
  std::vector<int> new_steps;
  temp = 1;
  for (int i = 0; i < num_axes; ++i) {
    new_steps.push_back(temp);
    temp *= out_dims[num_axes - 1 - i];
  }
  std::reverse(new_steps.begin(), new_steps.end());
  for (int i = 0; i < count; ++i) {
    int old_idx = 0;
    int idx = i;
    for (int j = 0; j < num_axes; ++j) {
      int order = orders[j];
      old_idx += (idx / new_steps[j]) * old_steps[order];
      idx %= new_steps[j];
    }
    dout[i] = din[old_idx];
  }
}

lite::DDim UniqueFlattenTo2d(const lite::DDim& src, int num_col_dims) {
  return DDim(std::vector<DDim::value_type>{
      src.Slice(0, num_col_dims).production(),
      src.Slice(num_col_dims, src.size()).production()});
}

template <typename T>
static std::vector<lite::Tensor> Unbind(const lite::Tensor& in) {
  int64_t size = in.dims()[0];
  std::vector<lite::Tensor> tensors(size);
  for (int64_t i = 0; i < size; ++i) {
    tensors[i] = in.Slice<T>(i, i + 1);
  }
  return tensors;
}

template <typename T>
void UniqueConcatFunc(const std::vector<lite::Tensor>& input,
                      const int axis,
                      lite::Tensor* output) {
  size_t num = input.size();
  auto dim_0 = input[0].dims();
  int64_t concat_input_size = 1;
  int64_t num_cancats = 1;
  for (int i = axis + 1; i < dim_0.size(); i++) {
    concat_input_size *= dim_0[i];
  }
  for (int i = 0; i < axis; i++) {
    num_cancats *= dim_0[i];
  }
  auto* dst_ptr = output->mutable_data<T>();
  const int out_concat_axis = output->dims()[axis];
  int64_t offset_concat_axis = 0;
  int64_t out_sum = out_concat_axis * concat_input_size;
  for (int n = 0; n < num; n++) {
    auto dims = input[n].dims();
    auto* src_ptr = input[n].data<T>();
    int64_t in_concat_axis = dims[axis];
    auto* dout_ptr = dst_ptr + offset_concat_axis * concat_input_size;
    int64_t in_sum = in_concat_axis * concat_input_size;
    for (int i = 0; i < num_cancats; i++) {
      std::memcpy(dout_ptr, src_ptr, sizeof(T) * in_sum);
      dout_ptr += out_sum;
      src_ptr += in_sum;
    }
    offset_concat_axis += in_concat_axis;
  }
}

template <typename T>
static bool Equal(const lite::Tensor& a, const lite::Tensor& b) {
  if (a.numel() != b.numel()) {
    return false;
  }
  for (int64_t i = 0; i < a.numel(); ++i) {
    if (a.data<T>()[i] != b.data<T>()[i]) {
      return false;
    }
  }
  return true;
}

template <class ForwardIt, typename InT, typename IndexT>
static ForwardIt UniqueDimImpl(ForwardIt first,
                               ForwardIt last,
                               const std::vector<IndexT>& sorted_indices_vec,
                               std::vector<IndexT>* inverse_vec,
                               std::vector<IndexT>* counts_vec,
                               std::vector<IndexT>* indices_vec) {
  if (first == last) {
    return last;
  }
  (*inverse_vec)[sorted_indices_vec[0]] = 0;
  (*counts_vec)[0] = 1;
  (*indices_vec)[0] = sorted_indices_vec[0];
  ForwardIt begin = first;
  ForwardIt result = first;
  while (++first != last) {
    int64_t idx_first = std::distance(begin, first);
    int64_t idx_result = std::distance(begin, result);
    if (!Equal<InT>(*result, *first)) {
      if (++result != first) {
        *result = std::move(*first);
      }
      idx_result += 1;
      (*indices_vec)[idx_result] = sorted_indices_vec[idx_first];
    }
    (*inverse_vec)[sorted_indices_vec[idx_first]] = idx_result;
    (*counts_vec)[idx_result] += 1;
  }
  return ++result;
}

template <typename T>
void UniqueTensorFromVector(const std::vector<T>& src, lite::Tensor* dst) {
  auto* src_ptr = static_cast<const void*>(src.data());
  dst->Resize({static_cast<int64_t>(src.size())});
  auto* dst_ptr = static_cast<void*>(dst->mutable_data<T>());
  auto size = src.size() * sizeof(T);
  lite::TargetWrapperHost::MemcpySync(
      dst_ptr, src_ptr, size, IoDirection::HtoH);
}

template <typename InT, typename IndexT>
void UniqueDimFunc(const lite::Tensor& in,
                   lite::Tensor* out,
                   lite::Tensor* index,
                   lite::Tensor* indices,
                   lite::Tensor* count,
                   int axis,
                   bool return_index,
                   bool return_inverse,
                   bool return_counts) {
  // transpose tensor: eg. axis=1, [dim0, dim1, dim2] -> [dim1, dim0, dim2]
  std::vector<int> permute(in.dims().size());
  std::iota(permute.begin(), permute.end(), 0);
  permute[axis] = 0;
  permute[0] = axis;
  std::vector<int64_t> in_trans_dims_vec(in.dims().Vectorize());
  in_trans_dims_vec[axis] = in.dims()[0];
  in_trans_dims_vec[0] = in.dims()[axis];
  lite::Tensor in_trans;
  lite::DDim in_trans_dims = DDim(in_trans_dims_vec);
  in_trans.Resize(in_trans_dims);
  in_trans.mutable_data<InT>();
  UniqueTransCompute<InT>(in, &in_trans, permute);
  // reshape tensor: eg. [dim1, dim0, dim2] -> [dim1, dim0*dim2]
  lite::DDim in_trans_flat_dims = UniqueFlattenTo2d(in_trans_dims, 1);
  in_trans.Resize(in_trans_flat_dims);

  // sort indices
  std::vector<IndexT> sorted_indices_vec(in_trans.dims()[0]);
  std::iota(sorted_indices_vec.begin(), sorted_indices_vec.end(), 0);
  int64_t col = in_trans.dims()[1];
  const InT* in_trans_data = in_trans.data<InT>();
  std::sort(sorted_indices_vec.begin(),
            sorted_indices_vec.end(),
            [&](int64_t a, int64_t b) -> bool {
              for (int64_t i = 0; i < col; ++i) {
                InT lhs = in_trans_data[i + a * col];
                InT rhs = in_trans_data[i + b * col];
                if (lhs < rhs) {
                  return true;
                } else if (lhs > rhs) {
                  return false;
                }
              }
              return false;
            });
  // sort tensor according to indices
  lite::Tensor input_sorted;
  input_sorted.Resize(in_trans_dims);
  InT* input_sorted_data = input_sorted.mutable_data<InT>();
  for (size_t i = 0; i < sorted_indices_vec.size(); ++i) {
    memcpy(input_sorted_data + i * col,
           in_trans_data + static_cast<int64_t>(sorted_indices_vec[i]) * col,
           col * sizeof(InT));
  }
  std::vector<lite::Tensor> input_unbind = Unbind<InT>(input_sorted);
  std::vector<IndexT> inverse_vec(sorted_indices_vec.size(), 0);
  std::vector<IndexT> counts_vec(sorted_indices_vec.size(), 0);
  std::vector<IndexT> indices_vec(sorted_indices_vec.size(), 0);
  auto last = UniqueDimImpl<std::vector<lite::Tensor>::iterator, InT, IndexT>(
      input_unbind.begin(),
      input_unbind.end(),
      sorted_indices_vec,
      &inverse_vec,
      &counts_vec,
      &indices_vec);
  input_unbind.erase(last, input_unbind.end());
  counts_vec.erase(counts_vec.begin() + input_unbind.size(), counts_vec.end());
  indices_vec.erase(indices_vec.begin() + input_unbind.size(),
                    indices_vec.end());
  lite::Tensor out_trans;
  std::vector<int64_t> out_trans_dims_vec = in_trans_dims_vec;
  out_trans_dims_vec[0] = input_unbind.size();
  out_trans.Resize(out_trans_dims_vec);
  out_trans.mutable_data<InT>();
  std::swap(out_trans_dims_vec[0], out_trans_dims_vec[axis]);
  out->Resize(out_trans_dims_vec);
  out->mutable_data<InT>();
  UniqueConcatFunc<InT>(input_unbind, 0, &out_trans);
  UniqueTransCompute<InT>(out_trans, out, permute);
  if (return_inverse) {
    UniqueTensorFromVector(inverse_vec, index);
  }
  if (return_counts) {
    UniqueTensorFromVector(counts_vec, count);
  }
  if (return_index) {
    UniqueTensorFromVector(indices_vec, indices);
  }
}

#define UNIQUE_SET_OUT_PRECISION(out, dtype) \
  if (out) {                                 \
    if (dtype == 3) {                        \
      out->set_precision(PRECISION(kInt64)); \
    } else {                                 \
      out->set_precision(PRECISION(kInt32)); \
    }                                        \
  }

template <typename InT>
void UniqueCompute<InT>::Run() {
  auto& param = Param<operators::UniqueParam>();
  auto x = param.X;
  auto output = param.Out;
  auto index = param.Index;
  auto indices = param.Indices;
  auto count = param.Counts;
  auto dtype = param.dtype;
  bool return_index = param.return_index;
  bool return_inverse = param.return_inverse;
  bool return_counts = param.return_counts;
  auto axis_vec = param.axis;
  auto is_sorted = param.is_sorted;
  CHECK(dtype == 3 || dtype == 2) << "dtype must be int or int64, but now is "
                                  << static_cast<int>(dtype);
  // set output precision
  UNIQUE_SET_OUT_PRECISION(index, dtype);
  UNIQUE_SET_OUT_PRECISION(indices, dtype);
  UNIQUE_SET_OUT_PRECISION(count, dtype);

  if (!is_sorted) {
    if (dtype == 3) {
      UniqueFunc<InT, int64_t>(x, output, index, count);
    } else {
      UniqueFunc<InT, int>(x, output, index, count);
    }
    return;
  }
  if (x->numel() == 0) {
    output->template mutable_data<InT>();
    return;
  }
  if (axis_vec.empty()) {
    if (dtype == 3) {
      UniqueFlattendTensorFunc<InT, int64_t>(*x,
                                             output,
                                             index,
                                             indices,
                                             count,
                                             return_index,
                                             return_inverse,
                                             return_counts);
    } else {
      UniqueFlattendTensorFunc<InT, int>(*x,
                                         output,
                                         index,
                                         indices,
                                         count,
                                         return_index,
                                         return_inverse,
                                         return_counts);
    }
  } else {
    int axis = axis_vec[0];
    if (dtype == 3) {
      UniqueDimFunc<InT, int64_t>(*x,
                                  output,
                                  index,
                                  indices,
                                  count,
                                  axis,
                                  return_index,
                                  return_inverse,
                                  return_counts);
    } else {
      UniqueDimFunc<InT, int>(*x,
                              output,
                              index,
                              indices,
                              count,
                              axis,
                              return_index,
                              return_inverse,
                              return_counts);
    }
  }
}

#undef UNIQUE_SET_OUT_PRECISION

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using unique_compute_fp32 = paddle::lite::kernels::host::UniqueCompute<float>;
REGISTER_LITE_KERNEL(unique, kHost, kFloat, kNCHW, unique_compute_fp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Index",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Counts",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();

using unique_compute_int32 = paddle::lite::kernels::host::UniqueCompute<int>;
REGISTER_LITE_KERNEL(unique, kHost, kFloat, kNCHW, unique_compute_int32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Index",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Counts",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();

using unique_compute_int64 =
    paddle::lite::kernels::host::UniqueCompute<int64_t>;
REGISTER_LITE_KERNEL(unique, kHost, kFloat, kNCHW, unique_compute_int64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Index",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Counts",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();
