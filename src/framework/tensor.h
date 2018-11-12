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
#include <type_traits>
#include <typeindex>
#include <vector>

#include "common/enforce.h"
#include "common/types.h"
#include "framework/data_layout.h"
#include "framework/ddim.h"
#include "memory/t_malloc.h"

namespace paddle_mobile {
namespace framework {
template <typename... T>
struct SizeOfTypeFunctor;

template <typename T>
struct SizeOfTypeFunctor<T> {
  size_t operator()(std::type_index type) const {
    if (typeid(T).hash_code() == type.hash_code()) {
      return sizeof(T);
    } else {
      return 0UL;
    }
  }
};

template <>
struct SizeOfTypeFunctor<> {
  size_t operator()(std::type_index type) const { return 0UL; }
};

template <typename HEAD, typename... TAIL>
struct SizeOfTypeFunctor<HEAD, TAIL...> {
  size_t operator()(std::type_index type) const {
    SizeOfTypeFunctor<HEAD> head;
    size_t head_size = head(type);
    if (head_size != 0) {
      return head_size;
    }
    SizeOfTypeFunctor<TAIL...> tail;
    return tail(type);
  }
};

static inline size_t SizeOfType(std::type_index type) {
  SizeOfTypeFunctor<int8_t, int, half, float, double, int16_t, int64_t, bool,
                    size_t>
      functor;
  size_t size = functor(type);

  PADDLE_MOBILE_ENFORCE(size != 0UL, "Cannot get size of type %s", type.name());
  return size;
}

class LoDTensor;

class Tensor {
 public:
  Tensor() : offset_(0) {}
  template <typename T>
  Tensor(std::vector<T> input, DDim ddim) : offset_(0) {
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

  /*! Return a pointer to mutable memory block. */
  template <typename T>
  inline T *data() {
    check_memory_size();
    PADDLE_MOBILE_ENFORCE(
        (std::is_same<T, void>::value ||
         holder_->type().hash_code() == typeid(T).hash_code()),
        "Tensor holds the wrong type, it holds %s",
        this->holder_->type().name());

    return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 offset_);
  }

  /*! Return a pointer to constant memory block. */
  template <typename T>
  inline const T *data() const {
    check_memory_size();
    PADDLE_MOBILE_ENFORCE(
        (std::is_same<T, void>::value ||
         holder_->type().hash_code() == typeid(T).hash_code()),
        "Tensor holds the wrong type, it holds %s ,requested:%s",
        this->holder_->type().name(), typeid(T).name());

    return reinterpret_cast<const T *>(
        reinterpret_cast<uintptr_t>(holder_->ptr()) + offset_);
  }

  inline bool IsInitialized() const { return holder_ != nullptr; }

  /**
   * @brief   Return a pointer to mutable memory block.
   * @note    If not exist, then allocation.
   */
  template <typename T>
  inline T *mutable_data() {
    static_assert(std::is_pod<T>::value, "T must be POD");
    return reinterpret_cast<T *>(mutable_data(typeid(T)));
  }

  inline void *mutable_data(std::type_index type) {
    if (holder_ != nullptr) {
      holder_->set_type(type);
    }
    PADDLE_MOBILE_ENFORCE(numel() >= 0, "the Tensor's numel must >=0.")
    int64_t size = numel() * SizeOfType(type);
    if (holder_ == nullptr || holder_->size() < size + offset_) {
      holder_.reset(new PlaceholderImpl(size, type));
      offset_ = 0;
    }
    return reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(holder_->ptr()) + offset_);
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

  /*! Return the dimensions of the memory block. */
  inline const DDim &dims() const { return dims_; }

  /*! Return the numel of the memory block. */
  inline int64_t numel() const { return product(dims_); }

  /*! Resize the dimensions of the memory block. */
  inline Tensor &Resize(const DDim &dims) {
    dims_ = dims;
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

  std::type_index type() const {
    PADDLE_MOBILE_ENFORCE(
        holder_ != nullptr,
        "Tensor not initialized yet when Tensor::type() is called.")
    return holder_->type();
  }

  // memory size returns the holding memory size in byte.
  size_t memory_size() const {
    return holder_ == nullptr ? 0UL : holder_->size() - offset_;
  }

  inline void check_memory_size() const {
    PADDLE_MOBILE_ENFORCE(
        holder_ != nullptr,
        "Tensor holds no memory. Call Tensor::mutable_data first.");
    PADDLE_MOBILE_ENFORCE(numel() * SizeOfType(type()) <= memory_size(),
                          "Tensor's dims_ is out of bound. ");
  }

 private:
  /**
   * @note    Placeholder hides type T, so it doesn't appear as a
   * template
   *          parameter of Variable.
   */
  struct Placeholder {
    virtual ~Placeholder() = default;

    virtual void *ptr() const = 0;

    virtual size_t size() const = 0;

    virtual std::type_index type() const = 0;

    virtual void set_type(std::type_index type) = 0;
  };

  struct PlaceholderImpl : public Placeholder {
    PlaceholderImpl(size_t size, std::type_index type)
        : ptr_(static_cast<uint8_t *>(memory::Alloc(size)),
               memory::PODDeleter<uint8_t>()),
          size_(size),
          type_(type) {
      PADDLE_MOBILE_ENFORCE(ptr_ != nullptr,
                            "Insufficient memory to allocation");
    }

    virtual size_t size() const { return size_; }

    virtual void *ptr() const { return static_cast<void *>(ptr_.get()); }

    virtual std::type_index type() const { return type_; }

    virtual void set_type(std::type_index type) { type_ = type; }

    std::unique_ptr<uint8_t, memory::PODDeleter<uint8_t>> ptr_;

    /*! the size of memory block. */
    size_t size_;

    /* the current type of memory */
    std::type_index type_;
  };

  /*! holds the memory block if allocated. */
  std::shared_ptr<Placeholder> holder_;

  /**
   * @brief points to elements dimensions.
   *
   * @note dims_ do not indicate the memory block size.
   */

  DDim dims_;

  /**
   * @brief   A PlaceHolder may be shared by more than one tensor.
   *
   * @note    Some of them may be slices of the others. So the offset_
   *          is introduced here to indicate the byte offset between
   *          PlaceHolder::ptr_ and where the tensor data really
   * begins.
   */
  size_t offset_;

#ifdef PADDLE_MOBILE_FPGA
 public:  // NOLINT
  inline void reset_data_ptr(void *p) {
    ((PlaceholderImpl *)(holder_.get()))->ptr_.reset((uint8_t *)p);  // NOLINT
  }
  float scale[2];  // scale[0]= MAX/127.0, scale[1]= 127.0/MAX
#endif
};

#ifdef PADDLE_MOBILE_DEBUG
inline Print &operator<<(Print &printer, const Tensor &tensor) {
  printer << " dims: " << tensor.dims() << "\n";
  int stride = tensor.numel() / 20;
  stride = stride > 0 ? stride : 1;
#ifndef PADDLE_MOBILE_FPGA
  for (int i = 0; i < tensor.numel(); i += stride) {
    if (tensor.type() == typeid(float)) {
      printer << tensor.data<float>()[i] << " ";
    } else if (tensor.type() == typeid(int32_t)) {
      printer << tensor.data<int32_t>()[i] << " ";
    } else if (tensor.type() == typeid(int64_t)) {
      printer << tensor.data<int64_t>()[i] << " ";
    } else if (tensor.type() == typeid(int8_t)) {
      printer << static_cast<int>(tensor.data<int8_t>()[i]) << " ";
    } else if (tensor.type() == typeid(int32_t)) {
      printer << tensor.data<int32_t>()[i] << " ";
    }
  }
#endif

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
