/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/
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
template<typename T>
class PODDeleter {
  static_assert(std::is_pod<T>::value, "T must be POD");

public:
  explicit PODDeleter() {};

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
template<typename T>
class PlainDeleter {
public:
  explicit PlainDeleter() {};

  void operator()(T *ptr) { Free(reinterpret_cast<void *>(ptr)); }

};
}  // namespace memory
}  // namespace paddle
