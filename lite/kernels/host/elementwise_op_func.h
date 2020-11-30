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

template <class T>
T naive_pow(T a, T b) {
  return std::pow(a, b);
}

template <class T>
using BinaryOpFn = T(T, T);

template <class T>
using UaryOpFn = T(T);

template <class T, BinaryOpFn<T> binary_op, UaryOpFn<T> uary_op>
T naive_fused_op(T a, T b) {
  return uary_op(binary_op(a, b));
}

template <class T>
T naive_relu(T a) {
  return a > 0 ? a : 0;
}

template <class T>
T naive_tanh(T a) {
  return std::tanh(static_cast<double>(a));
}

enum class ElementwiseFusedActType { NONE, RELU, TANH };

/**
 * in Z = X op Y , there must be a minimal continuous mem in X or Y that could
 * do SIMD.
 */
enum class BroadcastType {
  UNKNOWN,
  DIM_NOT_MATCH,  // could not do elementwise
  SAME_DIM,  // if x and y had a same dim, it could be treated as broadcast,but
  // not recommended.
  X_AS_CONTINUOUS,  // e.g. X.shape=[1,1,3,5],Y.shape=[2,4,1,1]
  Y_AS_CONTINUOUS,  // e.g. X.shape=[2,4,1,1],Y.shape=[1,1,3,5]
  BOTH_CONTINUOUS   // e.g. X.Shape=[1,1,3,5],Y.shape=[8,9,3,5]
};

/**
 * Get broadcast type, x_dims and x_dims must have same dim_size. The dimension
 * which will be broadcast should be set to 1, and the 1 at high dimension
 * should not be omitted
 * e.g. x_dims=[3,1,1,5] and y_dims=[1,2,4,1] is fine, but y_dims should not be
 * [2,4,1]
 * @tparam DimValue_t data type of dim's value
 * @param x_dims pointer to x's dim array, which is `dim_size` length
 * @param y_dims pointer to y's dim array, which is `dim_size` length
 * @param dim_size dim_size of x and y
 */
template <class DimValue_t>
BroadcastType get_broadcast_type(DimValue_t *x_dims,
                                 DimValue_t *y_dims,
                                 int dim_size) {
  if (memcmp(x_dims, y_dims, sizeof(DimValue_t) * dim_size) == 0) {
    return BroadcastType::SAME_DIM;
  }

  // check if it is broadcast
  for (int i = 0; i < dim_size; ++i) {
    if (x_dims[i] != 1 && y_dims[i] != 1 && x_dims[i] != y_dims[i]) {
      return BroadcastType::DIM_NOT_MATCH;
    }
  }

  int pos = dim_size - 1;
  while (pos >= 0 && x_dims[pos] == y_dims[pos] && x_dims[pos] == 1) {
    --pos;
  }
  if (x_dims[pos] == y_dims[pos]) {
    return BroadcastType::BOTH_CONTINUOUS;
  }
  if (x_dims[pos] != 1) {
    return BroadcastType::X_AS_CONTINUOUS;
  }
  if (y_dims[pos] != 1) {
    return BroadcastType::Y_AS_CONTINUOUS;
  }
  return BroadcastType::UNKNOWN;
}

template <class Elem_t>
struct BatchElementWiseArgMemPointer {
  const Elem_t *x_data = nullptr;
  const Elem_t *y_data = nullptr;
  Elem_t *z_data = nullptr;
};

template <class Elem_t, class DimValue_t>
struct BatchElementWiseArg {
  BroadcastType BcastType() const { return broadcast_type_; }
  int64_t ElemNumPerBatch() const { return continuous_length_; }
  int64_t BatchNum() const { return z_num_ / continuous_length_; }

  BatchElementWiseArgMemPointer<Elem_t> AllAtBatch(int64_t elem_id) {
    BatchElementWiseArgMemPointer<Elem_t> ret = {x_data_, y_data_, z_data_};
    int64_t ind = 0;
    for (int64_t i = 0; i < dim_size_; ++i) {
      ind = elem_id / element_id_stride_[i];
      ret.x_data += bcast_x_stride_[i] * ind;
      ret.y_data += bcast_y_stride_[i] * ind;
      ret.z_data += z_stride_[i] * ind;
      elem_id -= (element_id_stride_[i] * ind);
    }
    return ret;
  }

  const Elem_t *XAtBatch(int64_t batch_id) const {
    return x_data_ +
           ElemID2Offset(batch_id * continuous_length_, bcast_x_stride_);
  }
  const Elem_t *YAtBatch(int64_t batch_id) const {
    return y_data_ +
           ElemID2Offset(batch_id * continuous_length_, bcast_y_stride_);
  }
  Elem_t *ZAtBatch(int64_t batch_id) const {
    return z_data_ + ElemID2Offset(batch_id * continuous_length_, z_stride_);
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
   * @param x_stride x's memory stride
   * @param y_stride y's memory stride
   * @param z_stride z's memory stride
   * @param dim_size dim array's length
   * @param broadcast_type Could get from get_broadcast_type(), if set to
   * BroadcastType::UNKNOWN, this function will call get_broadcast_type
   * automatically
   *
   * @Note the memory stride describes how element are stored in memory.
   * e.g. Given tensor X has a dim [C,H,W], then X.At(i,j,k) should be stored
   * at X.data() + x_stride[0]*i + x_stride[1]*j + x_stride[2]*k
   *
   * @Warning the element in X, Y and Z must be stored as low dimension first,
   * e.g. Tensor X has a dim [5,6,7] ,then X.At(0,0,0) -> X.At(0,0,7) must be
   * stored at X.data() continuously
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
  int64_t continuous_length_ = 0;
  BroadcastType broadcast_type_ = BroadcastType::UNKNOWN;

  std::vector<DimValue_t> bcast_x_stride_;
  std::vector<DimValue_t> bcast_y_stride_;
  std::vector<DimValue_t> z_stride_;
  std::vector<DimValue_t> element_id_stride_;

  /**
   * Every element of some **FULL** tensor has its own logic id, ElemID2Offset
   * will convert this id to its memory offset
   * eg. given x.dims=[1,1,2,3] y.dims=[4,5,1,1],
   * then the full tensor's dim should be [4,5,2,3],
   * and, the element at [i,j,k,l] will get the
   * elem_id of `i*30 + j*6 + k*3 +l`
   * this elem_id works for all tensor X, Y and Z.
   */
  int64_t ElemID2Offset(int64_t elem_id,
                        const std::vector<DimValue_t> &bcast_stride) const {
    int64_t ind = 0;
    int64_t offset = 0;
    for (int64_t i = 0; i < dim_size_; ++i) {
      ind = elem_id / element_id_stride_[i];
      offset += bcast_stride[i] * ind;
      elem_id -= (element_id_stride_[i] * ind);
    }
    return offset;
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
  // arg checking
  if (broadcast_type == BroadcastType::UNKNOWN) {
    VLOG(4) << "No broadcast type input";
    broadcast_type = get_broadcast_type(x_dims, y_dims, dim_size);
  }
  if (broadcast_type == BroadcastType::UNKNOWN ||
      broadcast_type == BroadcastType::DIM_NOT_MATCH) {
    LOG(FATAL) << "Wrong broadcast type";
    return;
  }
  if (broadcast_type == BroadcastType::SAME_DIM) {
    broadcast_type = BroadcastType::BOTH_CONTINUOUS;
    VLOG(4) << "Same dim detected";
    // SAME_DIM should not be treated as broadcast. For SAME_DIM is a special
    // case of BOTH_CONTINUOUS, we could still process it.
  }

  if (x_stride[dim_size - 1] != 1 || y_stride[dim_size - 1] != 1 ||
      z_stride[dim_size - 1] != 1) {
    LOG(FATAL) << "data are not stored continuously";
    return;
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

  // get_continuous_length
  int64_t continuous_elem_num = z_dims[dim_size - 1];
  int end_pos = dim_size - 2;
  switch (broadcast_type) {
    case BroadcastType::X_AS_CONTINUOUS: {
      while (end_pos >= 0 && y_dims[end_pos] == 1) {
        if (HasGapToNextDim(z_dims, z_stride, end_pos) ||
            HasGapToNextDim(x_dims, x_stride, end_pos)) {
          break;
        }
        continuous_elem_num *= z_dims[end_pos];
        --end_pos;
      }
      break;
    }
    case BroadcastType::Y_AS_CONTINUOUS: {
      while (end_pos >= 0 && x_dims[end_pos] == 1) {
        if (HasGapToNextDim(z_dims, z_stride, end_pos) ||
            HasGapToNextDim(y_dims, y_stride, end_pos)) {
          break;
        }
        continuous_elem_num *= z_dims[end_pos];
        --end_pos;
      }
      break;
    }
    case BroadcastType::BOTH_CONTINUOUS: {
      while (end_pos >= 0 && x_dims[end_pos] == y_dims[end_pos]) {
        if (HasGapToNextDim(z_dims, z_stride, end_pos) ||
            HasGapToNextDim(x_dims, x_stride, end_pos) ||
            HasGapToNextDim(y_dims, y_stride, end_pos)) {
          break;
        }
        continuous_elem_num *= z_dims[end_pos];
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
  continuous_length_ = continuous_elem_num;
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
 * This function can handle any kinds of element-wise operation technically,
 * But it's recommended to use this function only when there is broadcast
 * needed.
 * @note: This function is easy to parallelized, check the final part,
 *  1. in every loop, we are handling a batch
 *  2. different loop process individual batch
 *
 * @note see BatchElementWiseArg::Update to get info about args
 */
template <class Elem_t, class DimValue_t>
void common_elmentwise_op_naive_cpu(
    const BatchElementWiseArg<Elem_t, DimValue_t> &batch_arg,
    std::function<Elem_t(Elem_t, Elem_t)> op) {
  int batch_num = batch_arg.BatchNum();
  auto bcast_type = batch_arg.BcastType();
  int range_length = batch_arg.ElemNumPerBatch();
  switch (bcast_type) {
    case (BroadcastType::X_AS_CONTINUOUS): {
      for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
        element_wise_range_to_one(batch_arg.XAtBatch(batch_id),
                                  batch_arg.YAtBatch(batch_id),
                                  batch_arg.ZAtBatch(batch_id),
                                  range_length,
                                  op);
      }
      break;
    }
    case (BroadcastType::Y_AS_CONTINUOUS): {
      for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
        element_wise_one_to_range(batch_arg.XAtBatch(batch_id),
                                  batch_arg.YAtBatch(batch_id),
                                  batch_arg.ZAtBatch(batch_id),
                                  range_length,
                                  op);
      }
      break;
    }
    case (BroadcastType::BOTH_CONTINUOUS): {
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

/**
 * fix missing dim of paddle lite tensor to fit this broadcast system.
 * @tparam DimValue_t
 * @param X
 * @param Y
 * @param Out
 * @param axis axis defined by paddle
 * @param out_dim_size dim size of Out
 * @param [out] p_x_dims fixed dim value of x
 * @param [out] p_y_dims fixed dim value of y
 */
template <class DimValue_t>
void fix_x_y_dims(const Tensor *X,
                  const Tensor *Y,
                  const Tensor *Out,
                  int axis,
                  std::vector<DimValue_t> *p_x_dims,
                  std::vector<DimValue_t> *p_y_dims) {
  auto &x_dims = *p_x_dims;
  auto &y_dims = *p_y_dims;
  int out_dim_size = Out->dims().size();
  x_dims.resize(out_dim_size, 1);
  y_dims.resize(out_dim_size, 1);

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
      if (Y->dims().size() != Out->dims().size()) {
        LOG(FATAL) << "X/Y and OUT dim size mismatch";
      } else {
        VLOG(4) << "Arguments broke API reference, for X.dims().size() is "
                   "smaller and axis is set";
        for (int i = 0; i < out_dim_size; ++i) {
          y_dims[i] = Y->dims()[i];
        }
        for (int i = 0; i < X->dims().size(); ++i) {
          x_dims[i + axis] = X->dims()[i];
        }
      }
    } else {
      for (int i = 0; i < out_dim_size; ++i) {
        x_dims[i] = X->dims()[i];
      }
      for (int i = 0; i < Y->dims().size(); ++i) {
        y_dims[i + axis] = Y->dims()[i];
      }
    }
  }
}

template <class T>
BatchElementWiseArg<T, int64_t> GenBatchElementWiseArg(const lite::Tensor *X,
                                                       const lite::Tensor *Y,
                                                       lite::Tensor *Out,
                                                       int axis = -1) {
  int out_dim_size = Out->dims().size();
  std::vector<int64_t> x_dims;
  std::vector<int64_t> y_dims;
  fix_x_y_dims<int64_t>(X, Y, Out, axis, &x_dims, &y_dims);

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
