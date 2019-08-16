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

#include <cstddef>
#include <type_traits>

namespace paddle_mobile {
namespace memory {

void Copy(void *dst, const void *src, size_t num);

void *Alloc(size_t size);

void Free(void *ptr);

/**
 * \brief   Free memory block in one place.
 *
 * \note    In some cases, custom deleter is used to
 *          deallocate the memory automatically for
 *          std::unique_ptr<T> in tensor.h.
 *          static_cast
 */
template <typename T>
class PODDeleter {
  static_assert(std::is_pod<T>::value, "T must be POD");

 public:
  explicit PODDeleter(){};

  void operator()(T *ptr) { Free(static_cast<void *>(ptr)); }
};

/**
 * \brief   Free memory block in one place does not meet POD
 *
 * \note    In some cases, custom deleter is used to
 *          deallocate the memory automatically for
 *          std::unique_ptr<T> in tensor.h.
 *          reinterpret_cast
 */
template <typename T>
class PlainDeleter {
 public:
  explicit PlainDeleter(){};

  void operator()(T *ptr) { Free(reinterpret_cast<void *>(ptr)); }
};
}  // namespace memory
}  // namespace paddle_mobile
