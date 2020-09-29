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

#pragma once

#include <cstring>
#include <functional>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
T naive_add(T a, T b) {
  return a + b;
}

template <class T>
T naive_sub(T a, T b) {
  return a - b;
}

template <class T>
T naive_mul(T a, T b) {
  return a * b;
}

template <class T>
T naive_max(T a, T b) {
  return a > b ? a : b;
}

template <class T>
T naive_div(T a, T b) {
  return a / b;
}

template <class T>
T naive_mod(T a, T b) {
  return a % b;
}

/**
 * in Z = X op Y , there must be a minimal continuos mem in X and Y that could
 * do SIMD there are only three patterns a. X's minimal continuos mem has same
 * length with Y -> BOTH_CONTINUOS b. X's minimal continuos mem could do SIMD
 * with Y's single value -> X_AS_CONTINUOS c. Y's minimal continuos mem could do
 * SIMD with X's single value -> Y_AS_CONTINUOS
 */
enum class BroadcastType {
  UNKNOWN,
  DIM_NOT_MATCH,  // could not do elementwise
  SAME_DIM,  // if x and y had a same dim, it could be treated as broadcast,but
  // not recommended.
  X_AS_CONTINUOS,  // e.g. X.shape=[1,1,3,5],Y.shape=[2,4,1,1]
  Y_AS_CONTINUOS,  // e.g. X.shape=[2,4,1,1],Y.shape=[1,1,3,5]
  BOTH_CONTINUOS   // e.g. X.Shape=[1,1,3,5],Y.shape=[8,9,3,5]
};

/**
 * Get broadcast type, x and y must have same dim_size, the dimension which will
 * be brodcasted should be set to 1 e.g. x_dims=[3,1,1,5] and y_dims=[1,2,4,1]
 * is ok
 * @tparam DimValue_t data type of dim's value
 * @param x_dims pointer to x's dim array, which is `dim_size` length
 * @param y_dims pointer to y's dim array, which is `dim_size` length
 * @param dim_size dim_size of x and y
 * @return
 */
template <class DimValue_t>
BroadcastType get_broadcast_type(DimValue_t *x_dims,
                                 DimValue_t *y_dims,
                                 int dim_size) {
  if (memcmp(x_dims, y_dims, sizeof(DimValue_t) * dim_size) == 0) {
    return BroadcastType::SAME_DIM;
  }

  BroadcastType ret = BroadcastType::UNKNOWN;
  // check if it is broadcast
  for (int i = 0; i < dim_size; ++i) {
    if (x_dims[i] != 1 && y_dims[i] != 1 && x_dims[i] != y_dims[i]) {
      return ret = BroadcastType::DIM_NOT_MATCH;
    }
  }

  int pos = dim_size - 1;
  while (pos >= 0 && x_dims[pos] == y_dims[pos] && x_dims[pos] == 1) {
    --pos;
  }
  if (x_dims[pos] == y_dims[pos]) {
    return ret = BroadcastType::BOTH_CONTINUOS;
  }
  if (x_dims[pos] != 1) {
    return ret = BroadcastType::X_AS_CONTINUOS;
  }
  if (y_dims[pos] != 1) {
    return ret = BroadcastType::Y_AS_CONTINUOS;
  }
  return ret;
}

template <class Elem_t, class DimValue_t>
struct BatchElementWiseArg {
  BroadcastType BcastType() const { return broadcast_type_; }
  int64_t ElemNumPerBatch() const { return continuos_length_; }
  int64_t BatchNum() const { return z_num_ / continuos_length_; }
  const Elem_t *XAtBatch(int64_t batch_id) const {
    return x_data_ +
           ElemID2Offset(batch_id * continuos_length_, bcast_x_stride_);
  }
  const Elem_t *YAtBatch(int64_t batch_id) const {
    return y_data_ +
           ElemID2Offset(batch_id * continuos_length_, bcast_y_stride_);
  }
  Elem_t *ZAtBatch(int64_t batch_id) const {
    return z_data_ + ElemID2Offset(batch_id * continuos_length_, z_stride_);
  }

  /**
   * @tparam Elem_t data type of element
   * @tparam DimValue_t data type of dim's value
   * @param x_data pointer to x's data
   * @param y_data pointer to y's data
   * @param z_data pointer to z's data
   * @param x_dims pointer to x's dim array, which is `dim_size` length
   * @param y_dims pointer to y's dim array, which is `dim_size` length
   * @param z_dims pointer to z's dim array, which is `dim_size` length
   * @param x_stride x's memory stride, e.g. &data[i][j][k] == data+i*stride[0]
   * + j*stride[1] + k*stride[3]
   * @param y_stride y's memory stride, e.g. &data[i][j][k] == data+i*stride[0]
   * + j*stride[1] + k*stride[3]
   * @param z_stride z's memory stride, e.g. &data[i][j][k] == data+i*stride[0]
   * + j*stride[1] + k*stride[3]
   * @param dim_size dim array's length
   * @param broadcast_type Could get from get_broadcast_type(), if set to
   * BroadcastType::UNKNOWN, this function will call get_broadcast_type
   * automatically
   */
  void Update(const Elem_t *x_data,
              const Elem_t *y_data,
              Elem_t *z_data,
              const DimValue_t *x_dims,
              const DimValue_t *y_dims,
              const DimValue_t *z_dims,
              const DimValue_t *x_stride,
              const DimValue_t *y_stride,
              const DimValue_t *z_stride,
              int dim_size,
              BroadcastType broadcast_type = BroadcastType::UNKNOWN);

 private:
  const Elem_t *x_data_ = nullptr;
  const Elem_t *y_data_ = nullptr;
  Elem_t *z_data_ = nullptr;
  int64_t z_num_ = 0;

  int dim_size_ = 0;
  int64_t continuos_length_ = 0;
  BroadcastType broadcast_type_ = BroadcastType::UNKNOWN;

  std::vector<DimValue_t> bcast_x_stride_;
  std::vector<DimValue_t> bcast_y_stride_;
  std::vector<DimValue_t> z_stride_;
  std::vector<DimValue_t> element_id_stride_;

  int64_t ElemID2Offset(int64_t elem_id,
                        const std::vector<DimValue_t> &stride) const {
    int64_t ind = 0;
    int64_t ret = 0;
    for (int64_t i = 0; i < dim_size_; ++i) {
      ind = elem_id / element_id_stride_[i];
      ret += stride[i] * ind;
      elem_id -= (element_id_stride_[i] * ind);
    }
    return ret;
  }

  bool HasGapToNextDim(const DimValue_t *dims,
                       const DimValue_t *stride,
                       int this_dim) const {
    return (dims[this_dim + 1] * stride[this_dim + 1]) != stride[this_dim];
  }
};

template <class Elem_t, class DimValue_t>
void BatchElementWiseArg<Elem_t, DimValue_t>::Update(
    const Elem_t *x_data,
    const Elem_t *y_data,
    Elem_t *z_data,
    const DimValue_t *x_dims,
    const DimValue_t *y_dims,
    const DimValue_t *z_dims,
    const DimValue_t *x_stride,
    const DimValue_t *y_stride,
    const DimValue_t *z_stride,
    int dim_size,
    BroadcastType broadcast_type) {
  // pre process
  if (broadcast_type == BroadcastType::UNKNOWN) {
    broadcast_type = get_broadcast_type(x_dims, y_dims, dim_size);
  }
  if (broadcast_type == BroadcastType::UNKNOWN ||
      broadcast_type == BroadcastType::DIM_NOT_MATCH) {
    return;
  }
  if (broadcast_type == BroadcastType::SAME_DIM) {
    broadcast_type = BroadcastType::BOTH_CONTINUOS;
    // SAME_DIM is a special case of BOTH_CONTINUOS
  }

  // generate element_id stride
  std::vector<DimValue_t> element_id_stride(dim_size, 1);
  for (int i = dim_size - 2; i >= 0; --i) {
    element_id_stride[i] = z_dims[i + 1] * element_id_stride[i + 1];
  }

  // generate broadcast_stride
  std::vector<DimValue_t> bcast_x_stride(x_stride, x_stride + dim_size);
  std::vector<DimValue_t> bcast_y_stride(y_stride, y_stride + dim_size);
  int total_elem_num = 1;
  for (int i = 0; i < dim_size; ++i) {
    if (x_dims[i] == 1) {
      bcast_x_stride[i] = 0;
    }
    if (y_dims[i] == 1) {
      bcast_y_stride[i] = 0;
    }
    total_elem_num *= z_dims[i];
  }

  // get_continuos_length
  if (x_stride[dim_size - 1] != 1 || y_stride[dim_size - 1] != 1 ||
      z_stride[dim_size - 1] != 1) {
    return;
  }
  int64_t continuos_elem_num = z_dims[dim_size - 1];
  int end_pos = dim_size - 2;
  switch (broadcast_type) {
    case BroadcastType::X_AS_CONTINUOS: {
      while (end_pos >= 0 && y_dims[end_pos] == 1) {
        if (HasGapToNextDim(z_dims, z_stride, end_pos) ||
            HasGapToNextDim(x_dims, x_stride, end_pos)) {
          break;
        }
        continuos_elem_num *= z_dims[end_pos];
        --end_pos;
      }
      break;
    }
    case BroadcastType::Y_AS_CONTINUOS: {
      while (end_pos >= 0 && x_dims[end_pos] == 1) {
        if (HasGapToNextDim(z_dims, z_stride, end_pos) ||
            HasGapToNextDim(y_dims, y_stride, end_pos)) {
          break;
        }
        continuos_elem_num *= z_dims[end_pos];
        --end_pos;
      }
      break;
    }
    case BroadcastType::BOTH_CONTINUOS: {
      while (end_pos >= 0 && x_dims[end_pos] == y_dims[end_pos]) {
        if (HasGapToNextDim(z_dims, z_stride, end_pos) ||
            HasGapToNextDim(x_dims, x_stride, end_pos) ||
            HasGapToNextDim(y_dims, y_stride, end_pos)) {
          break;
        }
        continuos_elem_num *= z_dims[end_pos];
        --end_pos;
      }
      break;
    }
    default: {
      return;  // code should never goes to here
    }
  }

  // do update
  x_data_ = x_data;
  y_data_ = y_data;
  z_data_ = z_data;
  z_num_ = total_elem_num;

  dim_size_ = dim_size;
  continuos_length_ = continuos_elem_num;
  broadcast_type_ = broadcast_type;

  bcast_x_stride_ = std::move(bcast_x_stride);
  bcast_y_stride_ = std::move(bcast_y_stride);
  z_stride_ = std::vector<DimValue_t>(z_stride, z_stride + dim_size);
  element_id_stride_ = std::move(element_id_stride);
}

template <class T>
void element_wise_one_to_range(const T *x,
                               const T *y,
                               T *z,
                               int64_t range_length,
                               std::function<T(T, T)> op) {
  for (int64_t i = 0; i < range_length; ++i) {
    z[i] = op(*x, y[i]);
  }
}

template <class T>
void element_wise_range_to_one(const T *x,
                               const T *y,
                               T *z,
                               int64_t range_length,
                               std::function<T(T, T)> op) {
  for (int64_t i = 0; i < range_length; ++i) {
    z[i] = op(x[i], *y);
  }
}

template <class T>
void element_wise_range_to_range(const T *x,
                                 const T *y,
                                 T *z,
                                 int64_t range_length,
                                 std::function<T(T, T)> op) {
  for (int64_t i = 0; i < range_length; ++i) {
    z[i] = op(x[i], y[i]);
  }
}

/**
 * This function can handle any kinds of element-wise operation,
 * But it's recommended to use this function only when there is broadcast
 * needed.
 * @note: This function is easy to parallelized, check the final part,
 *  1. in every loop, we are handling a batch
 *  2. different loop process individual batch
 *
 * @see BatchElementWiseArg::Update to get info about args
 */
template <class Elem_t, class DimValue_t>
void common_elmentwise_op_naive_cpu(
    const BatchElementWiseArg<Elem_t, DimValue_t> &batch_arg,
    std::function<Elem_t(Elem_t, Elem_t)> op) {
  int batch_num = batch_arg.BatchNum();
  auto bcast_type = batch_arg.BcastType();
  int range_length = batch_arg.ElemNumPerBatch();
  switch (bcast_type) {
    case (BroadcastType::X_AS_CONTINUOS): {
      for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
        element_wise_range_to_one(batch_arg.XAtBatch(batch_id),
                                  batch_arg.YAtBatch(batch_id),
                                  batch_arg.ZAtBatch(batch_id),
                                  range_length,
                                  op);
      }
      break;
    }
    case (BroadcastType::Y_AS_CONTINUOS): {
      for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
        element_wise_one_to_range(batch_arg.XAtBatch(batch_id),
                                  batch_arg.YAtBatch(batch_id),
                                  batch_arg.ZAtBatch(batch_id),
                                  range_length,
                                  op);
      }
      break;
    }
    case (BroadcastType::BOTH_CONTINUOS): {
      for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
        element_wise_range_to_range(batch_arg.XAtBatch(batch_id),
                                    batch_arg.YAtBatch(batch_id),
                                    batch_arg.ZAtBatch(batch_id),
                                    range_length,
                                    op);
      }
      break;
    }
  }
}

template <class DimValue_t>
void fix_x_y_dims(const Tensor *X,
                  const Tensor *Y,
                  const Tensor *Out,
                  int axis,
                  int out_dim_size,
                  std::vector<DimValue_t> *p_x_dims,
                  std::vector<DimValue_t> *p_y_dims) {
  auto &x_dims = *p_x_dims;
  auto &y_dims = *p_y_dims;
  // fix missing dim in x_dims and y_dims
  if (axis == -1) {
    int i_raw = 0;
    int i_new = out_dim_size - X->dims().size();
    for (; i_raw < X->dims().size(); ++i_raw, ++i_new) {
      x_dims[i_new] = X->dims()[i_raw];
    }
    i_raw = 0;
    i_new = out_dim_size - Y->dims().size();
    for (; i_raw < Y->dims().size(); ++i_raw, ++i_new) {
      y_dims[i_new] = Y->dims()[i_raw];
    }
  } else {
    if (X->dims().size() != Out->dims().size()) {
      LOG(FATAL) << "X and OUT dim size mismatch";
    }
    for (int i = 0; i < out_dim_size; ++i) {
      x_dims[i] = X->dims()[i];
    }
    for (int i = axis; i < out_dim_size; ++i) {
      y_dims[i + axis] = Y->dims()[i];
    }
  }
}

template <class T>
BatchElementWiseArg<T, int64_t> GenBatchElementWiseArg(const lite::Tensor *X,
                                                       const lite::Tensor *Y,
                                                       lite::Tensor *Out,
                                                       int axis = -1) {
  int out_dim_size = Out->dims().size();
  std::vector<int64_t> x_dims(out_dim_size, 1);
  std::vector<int64_t> y_dims(out_dim_size, 1);
  fix_x_y_dims<int64_t>(X, Y, Out, axis, out_dim_size, &x_dims, &y_dims);

  auto &z_dims = Out->dims().data();
  // gen stride
  std::vector<int64_t> x_stride(out_dim_size, 1);
  std::vector<int64_t> y_stride(out_dim_size, 1);
  std::vector<int64_t> z_stride(out_dim_size, 1);
  for (int i = out_dim_size - 2; i >= 0; --i) {
    x_stride[i] = x_stride[i + 1] * x_dims[i + 1];
    y_stride[i] = y_stride[i + 1] * y_dims[i + 1];
    z_stride[i] = z_stride[i + 1] * z_dims[i + 1];
  }

  BatchElementWiseArg<T, int64_t> batch_arg;
  batch_arg.Update(X->data<T>(),
                   Y->data<T>(),
                   Out->mutable_data<T>(),
                   x_dims.data(),
                   y_dims.data(),
                   z_dims.data(),
                   x_stride.data(),
                   y_stride.data(),
                   z_stride.data(),
                   out_dim_size);
  return batch_arg;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
