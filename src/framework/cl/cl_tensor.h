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

#include <memory>
#include <string>
#include <vector>

#include "CL/cl.h"
#include "framework/cl/cl_deleter.h"
#include "framework/cl/cl_engine.h"
#include "framework/tensor_base.h"

namespace paddle_mobile {
namespace framework {

class CLTensor : TensorBase {
 public:
  CLTensor(cl_context context, cl_command_queue command_queue)
      : context_(context), command_queue_(command_queue) {}

  CLTensor() = default;

  /*
   * if init method haven't set context and command_queue, need set
   * */
  void SetContextAndCommandQueue(cl_context context,
                                 cl_command_queue command_queue) {
    context_ = context;
    command_queue_ = command_queue;
  }

  /*! Resize the dimensions of the memory block. */
  inline CLTensor &Resize(const DDim &dims) {
    dims_ = dims;
    return *this;
  }

  template <typename T>
  inline cl_mem mutable_with_data(const T *data) {
    int64_t size = numel() * sizeof(T);

    holder_.reset(new PlaceholderImpl(
        size, reinterpret_cast<void *>(const_cast<T *>(data)), type_id<T>(),
        context_, command_queue_));
    return reinterpret_cast<cl_mem>(holder_->ptr());
  }

  inline cl_mem mutable_data(std::string type) {
    if (holder_ != nullptr) {
      holder_->set_type(type);
    }
    PADDLE_MOBILE_ENFORCE(numel() >= 0, "the Tensor's numel must >=0.")
    int64_t size = numel() * SizeOfType(type);
    if (holder_ == nullptr || holder_->size() < size + offset_) {
      holder_.reset(new PlaceholderImpl(size, type, context_, command_queue_));
      offset_ = 0;
    }
    return reinterpret_cast<cl_mem>(holder_->ptr());
  }

  /**
   * @brief   Return a pointer to cl buffer.
   * @note    If not exist, then allocation.
   */
  template <typename T>
  inline cl_mem mutable_data() {
    return reinterpret_cast<cl_mem>(mutable_data(type_id<T>()));
  }

  /**
   * @brief     Return a pointer to cl buffer.
   *
   * @param[in] dims    The dimensions of the memory block.
   * @param[in] place   The place of the memory block.
   *
   * @note      If not exist, then allocation.
   */
  template <typename T>
  inline cl_mem mutable_data(DDim dims) {
    Resize(dims);
    return mutable_data<T>();
  }

  inline cl_mem CLBuffer() {
    check_memory_size();
    return reinterpret_cast<cl_mem>(
        reinterpret_cast<uintptr_t>(holder_->ptr()));
  }

  template <typename T>
  inline T *Data() {
    if (host_ptr_) {
      delete (host_ptr_);
      host_ptr_ = nullptr;
    }
    cl_mem buffer = CLBuffer();
    host_ptr_ = new char[holder_->size()];
    cl_int status;
    status = clEnqueueReadBuffer(command_queue_, buffer, CL_TRUE, 0,
                                 holder_->size(), host_ptr_, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
    return reinterpret_cast<T *>(host_ptr_);
  }

  int memorySize() { return holder_->size(); }

  ~CLTensor() {
    DLOG << "~CLTensor";
    if (host_ptr_) {
      DLOG << " delete host ptr ";
      delete (host_ptr_);
      host_ptr_ = nullptr;
    }
  }

 private:
  cl_context context_;
  cl_command_queue command_queue_;
  void *host_ptr_ = nullptr;

  struct PlaceholderImpl : public Placeholder {
    PlaceholderImpl(size_t size, void *input, std::string type,
                    cl_context context, cl_command_queue command_queue)
        : ptr_(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              size, reinterpret_cast<void *>(input), NULL)),
          size_(size),
          capatity_(size),
          type_(type),
          context_(context),
          command_queue_(command_queue) {}

    PlaceholderImpl(size_t size, std::string type, cl_context context,
                    cl_command_queue command_queue)
        : ptr_(clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL)),
          size_(size),
          capatity_(size),
          type_(type),
          context_(context),
          command_queue_(command_queue) {}

    virtual size_t size() const { return size_; }

    virtual void *ptr() const { return static_cast<void *>(ptr_.get()); }

    virtual std::string type() const { return type_; }

    virtual void set_type(std::string type) { type_ = type; }

    virtual void resize(size_t size) {
      if (size > capatity_) {
        capatity_ = size;
        ptr_.reset(
            clCreateBuffer(context_, CL_MEM_READ_WRITE, capatity_, NULL, NULL));
      }
      size_ = size;
    }

    std::unique_ptr<_cl_mem, CLMemDeleter> ptr_;

    size_t size_;

    size_t capatity_;

    /* the current type of memory */
    std::string type_;

    cl_context context_;
    cl_command_queue command_queue_;
  };
};

}  // namespace framework
}  // namespace paddle_mobile
