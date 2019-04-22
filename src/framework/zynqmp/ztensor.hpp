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

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "common/enforce.h"
#include "framework/data_layout.h"
#include "framework/tensor_base.h"
#include "memory/t_malloc.h"

#ifdef PADDLE_MOBILE_FPGA_KD

#include "fpga/KD/tensor.hpp"

namespace paddle_mobile {
namespace framework {

class LoDTensor;

class Tensor : public TensorBase {
 public:
  Tensor() {}
  template <typename T>
  Tensor(std::vector<T> input, DDim ddim) {
    PADDLE_MOBILE_ENFORCE(
        input.size() == framework::product(ddim),
        "input vector'length should be equal to tensor's length");

    auto input_ptr = mutable_data<T>(ddim);
    for (int i = 0; i < input.size(); ++i) {
      input_ptr[i] = input[i];
    }
  }

  Tensor(const Tensor &inTensor) {
    this->dims_ = inTensor.dims_;
    this->holder_ = inTensor.holder_;
    this->offset_ = inTensor.offset_;
  }

  /*! Resize the dimensions of the memory block. */
  inline Tensor &Resize(const DDim &dims) {
    dims_ = dims;
    // TODO(chonwhite) resize holder?
    return *this;
  }

  /*! The internal of two tensors share the same memory block. */
  inline Tensor &ShareDataWith(const Tensor &src) {
    src.check_memory_size();
    if (holder_.get() != src.holder_.get()) {
      *this = src;
    }
    return *this;
  }

  /*! The internal of two tensors share the same memory block. */
  inline Tensor &ShareHolderWith(const Tensor &src) {
    src.check_memory_size();
    if (holder_.get() != src.holder_.get()) {
      holder_ = src.holder_;
    }
    return *this;
  }

  inline zynqmp::Tensor *zynqmpTensor() const {
    PlaceholderImpl *holder = static_cast<PlaceholderImpl *>(holder_.get());
    // mutable_data(holder->type());
    return holder->tensor_;
  }

  inline void *mutable_data(const kTypeId_t type) {
    if (holder_ != nullptr) {
      holder_->set_type(type);
    }
    PADDLE_MOBILE_ENFORCE(numel() >= 0, "the Tensor's numel must >=0.")
    int64_t size = numel() * SizeOfType(type);
    if (holder_ == nullptr || holder_->size() < size + offset_) {
      PlaceholderImpl *impl = nullptr;
      if (holder_ == nullptr) {
        std::cout << "holder null" << std::endl;
        impl = new PlaceholderImpl(dims_, type);
        holder_.reset(impl);
      } else {
        impl = static_cast<PlaceholderImpl *>(holder_.get());
        std::cout << "holder reize" << std::endl;
        // holder_->resize(size);
      }
      impl->resize(dims_, type);
      offset_ = 0;
    }
    return reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(holder_->ptr()) + offset_);
  }

  /**
   * @brief   Return a pointer to mutable memory block.
   * @note    If not exist, then allocation.
   */
  template <typename T>
  inline T *mutable_data() {
    static_assert(std::is_pod<T>::value, "T must be POD");
    return reinterpret_cast<T *>(mutable_data(type_id<T>().hash_code()));
  }

  /**
   * @brief     Return a pointer to mutable memory block.
   *
   * @param[in] dims    The dimensions of the memory block.
   * @param[in] place   The place of the memory block.
   *
   * @note      If not exist, then allocation.
   */
  template <typename T>
  inline T *mutable_data(DDim dims) {
    static_assert(std::is_pod<T>::value, "T must be POD");
    Resize(dims);
    return mutable_data<T>();
  }

  /**
   * @brief  Return a sub-tensor of the given tensor.
   *
   * @param[in] begin_idx   The index of the start row(inclusive) to
   * slice.
   *                        The index number begins from 0.
   * @param[in] end_idx     The index of the end row(exclusive) to
   * slice.
   *                        The index number begins from 0.
   */
  inline Tensor Slice(int begin_idx, int end_idx) const {
    check_memory_size();
    PADDLE_MOBILE_ENFORCE(begin_idx >= 0,
                          "The start row index must be greater than 0.")
    PADDLE_MOBILE_ENFORCE(end_idx <= dims_[0],
                          "The end row index is out of bound.")
    PADDLE_MOBILE_ENFORCE(
        begin_idx < end_idx,
        "The start row index must be lesser than the end row index")
    if (dims_[0] == 1) {
      return *this;
    } else {
      size_t base = numel() / dims_[0];
      Tensor dst;
      dst.holder_ = holder_;
      DDim dst_dims = dims_;
      dst_dims[0] = end_idx - begin_idx;
      dst.Resize(dst_dims);
      dst.offset_ = offset_ + begin_idx * base * SizeOfType(type());
      return dst;
    }
  }

  /*! Return a pointer to mutable memory block. */
  template <typename T>
  inline T *data() {
    check_memory_size();
    PADDLE_MOBILE_ENFORCE(
        (std::is_same<T, void>::value ||
         holder_->type() == type_id<T>().hash_code()),
        "Tensor holds the wrong type, it holds %d, requested %d",
        this->holder_->type(), type_id<T>().hash_code());

    return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 offset_);
  }

  /*! Return a pointer to constant memory block. */
  template <typename T>
  inline const T *data() const {
    check_memory_size();
    PADDLE_MOBILE_ENFORCE(
        (std::is_same<T, void>::value ||
         holder_->type() == type_id<T>().hash_code()),
        "Tensor holds the wrong type, it holds %d, requested %d",
        this->holder_->type(), type_id<T>().hash_code());

    return reinterpret_cast<const T *>(
        reinterpret_cast<uintptr_t>(holder_->ptr()) + offset_);
  }

 private:
  struct PlaceholderImpl : public Placeholder {
    PlaceholderImpl(DDim ddim, const kTypeId_t type) {
      tensor_ = new zynqmp::Tensor();
      type_ = type;
      std::vector<int> v = framework::vectorize2int(ddim);

      zynqmp::LayoutType layout_type = zynqmp::NCHW;
      switch (v.size()) {
        case 1:
          layout_type = zynqmp::N;
          break;
        case 2:
          layout_type = zynqmp::NC;
          break;
        case 3:
          layout_type = zynqmp::NHW;
          break;
        case 4:
          layout_type = zynqmp::NCHW;
          break;
      }
      zynqmp::Shape input_shape(layout_type, v);

      // for (int i = 0; i < v.size(); i++) {
      //   std::cout << ":" << v[i] << std::endl;
      // }
      zynqmp::DataType dtype = type == _float ? zynqmp::FP32 : zynqmp::FP16;
      tensor_->mutableData<float>(dtype, input_shape);
    }

    virtual size_t size() const { return size_; }

    virtual void *ptr() const {
      void *ptr = tensor_->data<void *>();
      return ptr;
    }

    virtual kTypeId_t type() const { return type_; }

    virtual void set_type(const kTypeId_t type) { type_ = type; }

    virtual void resize(size_t size) {
      if (size > capatity_) {
        capatity_ = size;
        // TODO(chonwhite) implement;
      }
      size_ = size;
    }

    void resize(DDim ddim, const kTypeId_t type) {
      std::vector<int> v = framework::vectorize2int(ddim);

      zynqmp::LayoutType layout_type = zynqmp::NCHW;
      switch (v.size()) {
        case 1:
          layout_type = zynqmp::N;
          break;
        case 2:
          layout_type = zynqmp::NC;
          break;
        case 3:
          layout_type = zynqmp::NHW;
          break;
        case 4:
          layout_type = zynqmp::NCHW;
          break;
      }
      zynqmp::Shape input_shape(layout_type, v);
      zynqmp::DataType dtype = type == _float ? zynqmp::FP32 : zynqmp::FP16;
      tensor_->mutableData<float>(dtype, input_shape);
    }

    /*! the size of memory block. */
    size_t size_;

    size_t capatity_;

    /* the current type of memory */
    kTypeId_t type_;

    zynqmp::Tensor *tensor_;
    // zynqmp::Shape* shape_;
  };
};

#ifdef PADDLE_MOBILE_DEBUG
inline Print &operator<<(Print &printer, const Tensor &tensor) {
  printer << " dims: " << tensor.dims() << "\n";
  int stride = tensor.numel() / 20;
  stride = stride > 0 ? stride : 1;
  return printer;
}

#endif

inline Tensor ReshapeToMatrix(const Tensor &src, int num_col_dims) {
  Tensor res;
  res.ShareDataWith(src);
  res.Resize(flatten_to_2d(src.dims(), num_col_dims));
  return res;
}

}  // namespace framework
}  // namespace paddle_mobile

#endif
