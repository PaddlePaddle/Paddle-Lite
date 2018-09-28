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

#include <type_traits>
#include <typeindex>

#include "common/enforce.h"
#include "common/types.h"
#include "framework/ddim.h"

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

class TensorBase {
 public:
  virtual inline TensorBase &Resize(const DDim &dims) = 0;

  inline bool IsInitialized() const { return holder_ != nullptr; }

  virtual inline void *mutable_data(std::type_index type) = 0;

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

  /*! Return the dimensions of the memory block. */
  inline const DDim &dims() const { return dims_; }

  /*! Return the numel of the memory block. */
  inline int64_t numel() const { return product(dims_); }

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

 protected:
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

  /**
   * @brief points to elements dimensions.
   *
   * @note dims_ do not indicate the memory block size.
   */

  DDim dims_;

  /*! holds the memory block if allocated. */
  std::shared_ptr<Placeholder> holder_;

  /**
   * @brief   A PlaceHolder may be shared by more than one tensor.
   *
   * @note    Some of them may be slices of the others. So the offset_
   *          is introduced here to indicate the byte offset between
   *          PlaceHolder::ptr_ and where the tensor data really
   * begins.
   */
  size_t offset_ = 0;
};

}  // namespace framework
}  // namespace paddle_mobile
