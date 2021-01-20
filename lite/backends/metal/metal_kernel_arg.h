// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef LITE_BACKENDS_METAL_METAL_KERNEL_ARG_H_
#define LITE_BACKENDS_METAL_METAL_KERNEL_ARG_H_

#include <memory>
#include <type_traits>
#include <vector>
#include "lite/backends/metal/metal_buffer.h"
#include "lite/backends/metal/metal_image.h"
#include "lite/utils/varient.h"

namespace paddle {
namespace lite {

struct metal_kernel_arg {
  explicit metal_kernel_arg(const metal_buffer* buf) noexcept {
    var_.set<const metal_buffer*>(buf);
  }
  explicit metal_kernel_arg(const metal_buffer& buf) noexcept {
    var_.set<const metal_buffer*>(&buf);
  }

  explicit metal_kernel_arg(const std::shared_ptr<metal_buffer>& buf) noexcept {
    var_.set<const metal_buffer*>(buf.get());
  }
  explicit metal_kernel_arg(const std::unique_ptr<metal_buffer>& buf) noexcept {
    var_.set<const metal_buffer*>(buf.get());
  }

  explicit metal_kernel_arg(const std::vector<metal_buffer*>* bufs) noexcept {
    var_.set<const std::vector<metal_buffer*>*>(bufs);
  }

  explicit metal_kernel_arg(const std::vector<metal_buffer*>& bufs) noexcept {
    var_.set<const std::vector<metal_buffer*>*>(&bufs);
  }

  explicit metal_kernel_arg(
      const std::vector<std::shared_ptr<metal_buffer>>* bufs) noexcept {
    var_.set<const std::vector<std::shared_ptr<metal_buffer>>*>(bufs);
  }

  explicit metal_kernel_arg(
      const std::vector<std::shared_ptr<metal_buffer>>& bufs) noexcept {
    var_.set<const std::vector<std::shared_ptr<metal_buffer>>*>(&bufs);
  }

  explicit metal_kernel_arg(const metal_image* img) noexcept {
    var_.set<const metal_image*>(img);
  }

  explicit metal_kernel_arg(const metal_image& img) noexcept {
    var_.set<const metal_image*>(&img);
  }

  explicit metal_kernel_arg(const std::shared_ptr<metal_image>& img) noexcept {
    var_.set<const metal_image*>(img.get());
  }

  explicit metal_kernel_arg(const std::unique_ptr<metal_image>& img) noexcept {
    var_.set<const metal_image*>(img.get());
  }

  explicit metal_kernel_arg(const std::vector<metal_image*>* imgs) noexcept {
    var_.set<const std::vector<metal_image*>*>(imgs);
  }

  explicit metal_kernel_arg(const std::vector<metal_image*>& imgs) noexcept {
    var_.set<const std::vector<metal_image*>*>(&imgs);
  }

  explicit metal_kernel_arg(
      const std::vector<std::shared_ptr<metal_image>>* imgs) noexcept {
    var_.set<const std::vector<std::shared_ptr<metal_image>>*>(imgs);
  }

  explicit metal_kernel_arg(
      const std::vector<std::shared_ptr<metal_image>>& imgs) noexcept {
    var_.set<const std::vector<std::shared_ptr<metal_image>>*>(&imgs);
  }

  template <typename T>
  explicit metal_kernel_arg(const std::shared_ptr<T>& arg) noexcept
      : metal_kernel_arg(*arg) {}

  template <typename T>
  explicit metal_kernel_arg(const std::unique_ptr<T>& arg) noexcept
      : metal_kernel_arg(*arg) {}

  template <
      typename T,
      typename std::enable_if<(
          !std::is_convertible<
              typename std::decay<typename std::remove_pointer<T>::type>::type*,
              metal_buffer*>::value &&
          !std::is_convertible<
              typename std::decay<typename std::remove_pointer<T>::type>::type*,
              metal_image*>::value)>::type* = nullptr>
  explicit metal_kernel_arg(const T& generic_arg) noexcept : size_(sizeof(T)) {
    var_.set<const void*>(&generic_arg);
  }

  variant<const void*,
          const metal_buffer*,
          const std::vector<metal_buffer*>*,
          const std::vector<std::shared_ptr<metal_buffer>>*,
          const metal_image*,
          const std::vector<metal_image*>*,
          const std::vector<std::shared_ptr<metal_image>>*>
      var_;

  size_t size_{0};
};
}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_KERNEL_ARG_H_
