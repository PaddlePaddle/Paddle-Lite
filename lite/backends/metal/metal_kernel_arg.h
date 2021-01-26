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

struct MetalKernelArgument {
  explicit MetalKernelArgument(const MetalBuffer* buf) noexcept {
    var_.set<const MetalBuffer*>(buf);
  }
  explicit MetalKernelArgument(const MetalBuffer& buf) noexcept {
    var_.set<const MetalBuffer*>(&buf);
  }

  explicit MetalKernelArgument(
      const std::shared_ptr<MetalBuffer>& buf) noexcept {
    var_.set<const MetalBuffer*>(buf.get());
  }
  explicit MetalKernelArgument(
      const std::unique_ptr<MetalBuffer>& buf) noexcept {
    var_.set<const MetalBuffer*>(buf.get());
  }

  explicit MetalKernelArgument(const std::vector<MetalBuffer*>* bufs) noexcept {
    var_.set<const std::vector<MetalBuffer*>*>(bufs);
  }

  explicit MetalKernelArgument(const std::vector<MetalBuffer*>& bufs) noexcept {
    var_.set<const std::vector<MetalBuffer*>*>(&bufs);
  }

  explicit MetalKernelArgument(
      const std::vector<std::shared_ptr<MetalBuffer>>* bufs) noexcept {
    var_.set<const std::vector<std::shared_ptr<MetalBuffer>>*>(bufs);
  }

  explicit MetalKernelArgument(
      const std::vector<std::shared_ptr<MetalBuffer>>& bufs) noexcept {
    var_.set<const std::vector<std::shared_ptr<MetalBuffer>>*>(&bufs);
  }

  explicit MetalKernelArgument(const MetalImage* img) noexcept {
    var_.set<const MetalImage*>(img);
  }

  explicit MetalKernelArgument(const MetalImage& img) noexcept {
    var_.set<const MetalImage*>(&img);
  }

  explicit MetalKernelArgument(
      const std::shared_ptr<MetalImage>& img) noexcept {
    var_.set<const MetalImage*>(img.get());
  }

  explicit MetalKernelArgument(
      const std::unique_ptr<MetalImage>& img) noexcept {
    var_.set<const MetalImage*>(img.get());
  }

  explicit MetalKernelArgument(const std::vector<MetalImage*>* imgs) noexcept {
    var_.set<const std::vector<MetalImage*>*>(imgs);
  }

  explicit MetalKernelArgument(const std::vector<MetalImage*>& imgs) noexcept {
    var_.set<const std::vector<MetalImage*>*>(&imgs);
  }

  explicit MetalKernelArgument(
      const std::vector<std::shared_ptr<MetalImage>>* imgs) noexcept {
    var_.set<const std::vector<std::shared_ptr<MetalImage>>*>(imgs);
  }

  explicit MetalKernelArgument(
      const std::vector<std::shared_ptr<MetalImage>>& imgs) noexcept {
    var_.set<const std::vector<std::shared_ptr<MetalImage>>*>(&imgs);
  }

  template <typename T>
  explicit MetalKernelArgument(const std::shared_ptr<T>& arg) noexcept
      : MetalKernelArgument(*arg) {}

  template <typename T>
  explicit MetalKernelArgument(const std::unique_ptr<T>& arg) noexcept
      : MetalKernelArgument(*arg) {}

  template <
      typename T,
      typename std::enable_if<(
          !std::is_convertible<
              typename std::decay<typename std::remove_pointer<T>::type>::type*,
              MetalBuffer*>::value &&
          !std::is_convertible<
              typename std::decay<typename std::remove_pointer<T>::type>::type*,
              MetalImage*>::value)>::type* = nullptr>
  explicit MetalKernelArgument(const T& generic_arg) noexcept
      : size_(sizeof(T)) {
    var_.set<const void*>(&generic_arg);
  }

  variant<const void*,
          const MetalBuffer*,
          const std::vector<MetalBuffer*>*,
          const std::vector<std::shared_ptr<MetalBuffer>>*,
          const MetalImage*,
          const std::vector<MetalImage*>*,
          const std::vector<std::shared_ptr<MetalImage>>*>
      var_;

  size_t size_{0};
};
}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_KERNEL_ARG_H_
