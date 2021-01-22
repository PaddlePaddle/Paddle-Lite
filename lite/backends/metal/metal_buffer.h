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

#ifndef LITE_BACKENDS_METAL_METAL_BUFFER_H_
#define LITE_BACKENDS_METAL_METAL_BUFFER_H_

#if defined(__OBJC__)
#include <Metal/Metal.h>
#endif

#include <algorithm>

// #include <mutex>
#include "lite/backends/metal/metal_common.h"
#include "lite/backends/metal/metal_converter.h"

namespace paddle {
namespace lite {

class metal_device;
class metal_queue;

class metal_buffer {
 public:
  metal_buffer(const metal_device& device,
               size_t size,
               METAL_ACCESS_FLAG flag = METAL_ACCESS_FLAG::CPUReadWrite);

  metal_buffer(const metal_device& device,
               void* data,
               size_t size,
               METAL_ACCESS_FLAG flag = METAL_ACCESS_FLAG::CPUReadWrite);

  metal_buffer(const metal_device& device, void* mtl_buffer);

  metal_buffer(const metal_device& device,
               DDim inDim,
               METAL_PRECISION_TYPE precision = METAL_PRECISION_TYPE::FLOAT,
               bool padWhenOneC = false,
               bool convertToNHWC = true,
               bool withTranspose = false,
               METAL_ACCESS_FLAG flag = METAL_ACCESS_FLAG::CPUReadWrite);

  metal_buffer() = delete;

  ~metal_buffer();

  template <typename P>
  P* convert(data_converter<P>* converter);

  void read(void* data, size_t size, size_t offset) const;

  void write(const void* src,
             size_t size,
             size_t offset,
             const metal_queue& queue);
  void write(const void* src, size_t size, size_t offset) const;

  void read(void* data, size_t size, size_t offset, const metal_queue& queue);

  void copy(const metal_queue& queue,
            const metal_buffer& src,
            size_t size_,
            size_t src_offset,
            size_t dst_offset);

  void copy(const metal_buffer& src,
            size_t size_,
            size_t src_offset,
            size_t dst_offset) const;

  template <typename P>
  void from_nchw(const P* src) {
    static_assert(
        std::is_same<float, P>::value || std::is_same<metal_half, P>::value,
        "can support float and half");
  }

  template <typename P>
  void to_nchw(P* dst) {
    static_assert(
        std::is_same<float, P>::value || std::is_same<metal_half, P>::value,
        "can support float and half");
  }

#if defined(__OBJC__)
  id<MTLBuffer> get_buffer() const;
  id<MTLBuffer> mtl_buffer_{nil};
#else
  void* get_buffer() const;
  void* mtl_buffer_{nullptr};
#endif
  int get_offset() const;
  void set_offset(int offset);

 private:
  template <typename P>
  void expand_nhwc();
  void convert();

  metal_device* mtl_device_;

  //  std::recursive_mutex buffer_lock_;
  size_t size_;
  int offset_ = 0;

  DDim tensor_dim_;
  __unused DDim padToFourDim_;
  DDim dim_;
  void* data_ = nullptr;

  int precision_size_;
  size_t data_length_;

  int c_;
  int c_slices_;
  int padded_c_;
  int count_;
  bool pad_when_one_channel_;
  bool convert_to_nhwc_;
  bool with_transpose_;
  METAL_PRECISION_TYPE precision_;
  METAL_ACCESS_FLAG flags_;
};

}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_BUFFER_H_
