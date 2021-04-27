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

#include "lite/backends/metal/metal_common.h"
#include "lite/backends/metal/metal_converter.h"
#include "lite/backends/metal/metal_device.h"
#include "lite/backends/metal/metal_queue.h"

namespace paddle {
namespace lite {

class MetalBufferDescriptor;

class MetalBuffer {
 public:
  enum class TYPE { kCommonBuffer, kTensorBuffer, kUnknownBuffer };

  MetalBuffer(const MetalDevice& device, const MetalBufferDescriptor& desc);

#ifdef __OBJC__
  MetalBuffer(const MetalDevice& device, id<MTLBuffer> buffer);
#else
  MetalBuffer(const MetalDevice& device, void* buffer);
#endif

  MetalBuffer(const MetalDevice& device,
              size_t size,
              METAL_ACCESS_FLAG flag = METAL_ACCESS_FLAG::CPUReadWrite);

  MetalBuffer(const MetalDevice& device,
              void* data,
              size_t size,
              METAL_ACCESS_FLAG flag = METAL_ACCESS_FLAG::CPUReadWrite);

  MetalBuffer(const MetalDevice& device,
              const DDim& in_dim,
              METAL_PRECISION_TYPE precision = METAL_PRECISION_TYPE::FLOAT,
              bool pad_when_one_c = false,
              bool convert_to_NHWC = true,
              bool with_transpose = false,
              METAL_ACCESS_FLAG flag = METAL_ACCESS_FLAG::CPUReadWrite);

 public:
  MetalBuffer() = delete;
  ~MetalBuffer();

  template <typename P>
  P* Convert(DataConverter<P>* converter);

  void Read(void* data, size_t size, size_t offset) const;
  void Read(void* data,
            size_t size,
            size_t offset,
            const MetalQueue& queue) const;

  void Write(const void* src,
             size_t size,
             size_t offset,
             const MetalQueue& queue) const;
  void Write(const void* src, size_t size, size_t offset) const;

  void Copy(const MetalQueue& queue,
            const MetalBuffer& src,
            size_t size,
            size_t src_offset,
            size_t dst_offset) const;
  void Copy(const MetalBuffer& src,
            size_t size,
            size_t src_offset,
            size_t dst_offset) const;

  template <typename SP>
  void CopyFromNCHW(const SP* src);

  template <typename DP>
  void CopyToNCHW(DP* dst);

#if defined(__OBJC__)
  id<MTLBuffer> buffer() const;
  id<MTLBuffer> buffer_{nil};
#else
  void* buffer() const;
  void* buffer_{nullptr};
#endif

  int offset() const;
  void set_offset(int offset);

  MetalBuffer::TYPE type() { return type_; }
  DDim tensor_dim() { return tensor_dim_; }
  size_t data_length() { return data_length_; }

 private:
  template <typename P>
  void ExpandNHWC();
  void Convert();

  size_t size_;
  int offset_ = 0;

  DDim tensor_dim_;
  __unused DDim pad_to_four_dim_;
  DDim dim_;
  void* data_ = nullptr;

  int precision_size_;
  size_t data_length_;
  MetalDevice* mtl_device_;
  int c_;
  int c_slices_;
  int padded_c_;
  int count_;
  bool pad_when_one_channel_;
  bool convert_to_nhwc_;
  bool with_transpose_;
  METAL_PRECISION_TYPE precision_;
  METAL_ACCESS_FLAG flags_;
  TYPE type_;
};

struct MetalBufferDescriptor {
  DDim dim_;
  MetalBuffer::TYPE type_ = MetalBuffer::TYPE::kTensorBuffer;
  METAL_PRECISION_TYPE precision_ = METAL_PRECISION_TYPE::FLOAT;
  bool pad_when_one_c_ = false;
  bool convert_to_NHWC_ = true;
  bool with_transpose_ = false;
  METAL_ACCESS_FLAG flag_ = METAL_ACCESS_FLAG::CPUReadWrite;
};

}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_METAL_METAL_BUFFER_H_
