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

#pragma once

#if defined(__OBJC__)
#include <Metal/Metal.h>
#endif

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "lite/backends/metal/metal_buffer.h"
#include "lite/backends/metal/metal_common.h"
#include "lite/backends/metal/metal_image.h"
#include "lite/backends/metal/metal_kernel_arg.h"

namespace paddle {
namespace lite {

namespace metal_argument_helper {
using metal_encoder_t = id<MTLComputeCommandEncoder>;

struct arguments_stats {
  uint32_t total_argument_num_{0};
  uint32_t buffer_arguments_num_{0};
  uint32_t image_arguments_num_{0};
};

static void parse_argument(const arguments_stats &stats,
                           metal_encoder_t encoder,
                           const void *ptr,
                           const size_t &size) {
  if (ptr == nullptr) return;
  if (size < 0) return;
  [encoder setBytes:ptr length:size atIndex:stats.buffer_arguments_num_];
}

static void parse_argument(const arguments_stats &stats,
                           metal_encoder_t encoder,
                           const metal_buffer *arg) {
  if (arg == nullptr) return;
  [encoder setBuffer:arg->mtl_buffer_
              offset:static_cast<NSUInteger>(arg->get_offset())
             atIndex:stats.buffer_arguments_num_];
}

static void parse_argument(const arguments_stats &stats,
                           metal_encoder_t encoder,
                           const metal_image *arg) {
  if (arg == nullptr) return;
  [encoder setTexture:arg->mtl_image_ atIndex:stats.image_arguments_num_];
}

static void parse_argument(
    const arguments_stats &stats,
    metal_encoder_t encoder,
    const std::vector<std::shared_ptr<metal_buffer>> &args) {
  const auto count = args.size();
  if (count < 1) return;

  std::vector<id<MTLBuffer>> mtl_buf_array(count, nil);
  std::vector<NSUInteger> offsets(count, 0);
  for (size_t i = 0; i < count; ++i) {
    mtl_buf_array[i] = args[i].get()->get_buffer();
    offsets[i] = static_cast<NSUInteger>(args[i].get()->get_offset());
  }

  [encoder setBuffers:mtl_buf_array.data()
              offsets:offsets.data()
            withRange:NSRange{stats.image_arguments_num_, count}];
}

static void parse_argument(
    const arguments_stats &stats,
    metal_encoder_t encoder,
    const std::vector<std::shared_ptr<metal_image>> &args) {
  const auto count = args.size();
  if (count < 1) return;

  std::vector<id<MTLTexture>> mtl_buf_array(count, nil);
  for (size_t i = 0; i < count; ++i) {
    mtl_buf_array[i] = args[i].get()->get_image();
  }

  [encoder setTextures:mtl_buf_array.data()
             withRange:NSRange{stats.image_arguments_num_, count}];
}

static void parse_argument(const arguments_stats &stats,
                           metal_encoder_t encoder,
                           const std::vector<metal_buffer *> &args) {
  const auto count = args.size();
  if (count < 1) return;

  std::vector<id<MTLBuffer>> mtl_buf_array(count, nil);
  std::vector<NSUInteger> offsets(count, 0);
  for (size_t i = 0; i < count; ++i) {
    mtl_buf_array[i] =
        (reinterpret_cast<metal_buffer *>(args[i]))->get_buffer();
  }

  [encoder setBuffers:mtl_buf_array.data()
              offsets:offsets.data()
            withRange:NSRange{stats.buffer_arguments_num_, count}];
}

static void parse_argument(const arguments_stats &stats,
                           metal_encoder_t encoder,
                           const std::vector<metal_image *> &args) {
  const auto count = args.size();
  if (count < 1) return;

  std::vector<id<MTLTexture>> mtl_buf_array(count, nil);
  std::vector<NSUInteger> offsets(count, 0);
  for (size_t i = 0; i < count; ++i) {
    mtl_buf_array[i] = args[i]->get_image();
  }

  [encoder setTextures:mtl_buf_array.data()
             withRange:NSRange{stats.image_arguments_num_, count}];
}

static void parse_argument(const arguments_stats &stats,
                           metal_encoder_t encoder,
                           int offset,
                           const metal_buffer *arg) {
  if (arg == nullptr) return;
  [encoder setBuffer:arg->get_buffer()
              offset:static_cast<NSUInteger>(offset)
             atIndex:stats.buffer_arguments_num_];
}

__unused static void parse_argument(
    const arguments_stats &stats,
    metal_encoder_t encoder,
    std::vector<int> offsets_int,
    const std::vector<std::shared_ptr<metal_buffer>> &args) {
  const auto count = args.size();
  if (count < 1) return;

  std::vector<id<MTLBuffer>> mtl_buf_array(count, nil);
  std::vector<NSUInteger> offsets(count, 0);
  for (size_t i = 0; i < count; ++i) {
    mtl_buf_array[i] = args[i].get()->get_buffer();
    offsets[i] = static_cast<NSUInteger>(offsets_int[i]);
  }

  [encoder setBuffers:mtl_buf_array.data()
              offsets:offsets.data()
            withRange:NSRange{stats.buffer_arguments_num_, count}];
}

static void parse_argument(const arguments_stats &idx,
                           metal_encoder_t encoder,
                           std::vector<int> offsets_int,
                           const std::vector<metal_buffer *> &arg) {
  const auto count = arg.size();
  if (count < 1) return;

  std::vector<id<MTLBuffer>> mtl_buf_array(count, nil);
  std::vector<NSUInteger> offsets(count, 0);
  for (size_t i = 0; i < count; ++i) {
    mtl_buf_array[i] = (reinterpret_cast<metal_buffer *>(arg[i]))->get_buffer();
    offsets[i] = static_cast<NSUInteger>(offsets_int[i]);
  }

  [encoder setBuffers:mtl_buf_array.data()
              offsets:offsets.data()
            withRange:NSRange{idx.buffer_arguments_num_, count}];
}

bool parse_arguments(metal_encoder_t encoder,
                     const std::vector<metal_kernel_arg> &args) {
  const size_t argc = args.size();
  if (argc < 1) return false;
  arguments_stats stats = arguments_stats();
  for (size_t i = 0; i < argc; i++) {
    const auto &arg = args[i];
    if (auto buf_ptr = arg.var_.get_if<const metal_buffer *>()) {
      parse_argument(stats, encoder, buf_ptr);
      stats.buffer_arguments_num_++;
      stats.total_argument_num_++;
    } else if (auto vec_buf_ptrs =
                   arg.var_.get_if<const std::vector<metal_buffer *> *>()) {
      parse_argument(stats, encoder, *vec_buf_ptrs);
      stats.buffer_arguments_num_ += vec_buf_ptrs->size();
      stats.total_argument_num_ += vec_buf_ptrs->size();
    } else if (auto vec_buf_sptrs =
                   arg.var_.get_if<
                       const std::vector<std::shared_ptr<metal_buffer>> *>()) {
      parse_argument(stats, encoder, *vec_buf_sptrs);
      stats.buffer_arguments_num_ += vec_buf_sptrs->size();
      stats.total_argument_num_ += vec_buf_sptrs->size();
    } else if (auto img_ptr = arg.var_.get_if<const metal_image *>()) {
      parse_argument(stats, encoder, img_ptr);
      stats.image_arguments_num_++;
      stats.total_argument_num_++;
    } else if (auto vec_img_ptrs =
                   arg.var_.get_if<const std::vector<metal_image *> *>()) {
      parse_argument(stats, encoder, *vec_img_ptrs);
      stats.image_arguments_num_ += vec_img_ptrs->size();
      stats.total_argument_num_ += vec_img_ptrs->size();
    } else if (auto vec_img_sptrs =
                   arg.var_.get_if<
                       const std::vector<std::shared_ptr<metal_image>> *>()) {
      parse_argument(stats, encoder, *vec_img_sptrs);
      stats.image_arguments_num_ += vec_img_sptrs->size();
      stats.total_argument_num_ += vec_img_sptrs->size();
    } else {
      // TODO(lzy): Stuff.
      return false;
    }
  }
  return true;
}

bool parse_arguments(metal_encoder_t encoder,
                     std::vector<int> offset,
                     const std::vector<metal_kernel_arg> &args) {
  const size_t argc = args.size();
  if (argc < 1) return false;

  arguments_stats stats = arguments_stats();
  for (size_t i = 0; i < argc; i++) {
    const auto &arg = args[i];

    if (auto buf_ptr = arg.var_.get_if<const metal_buffer *>()) {
      parse_argument(stats, encoder, offset[i], buf_ptr);
      stats.buffer_arguments_num_++;
      stats.total_argument_num_++;
    } else if (auto vec_buf_ptrs =
                   arg.var_.get_if<const std::vector<metal_buffer *> *>()) {
      parse_argument(stats, encoder, (*vec_buf_ptrs));
      stats.buffer_arguments_num_ += vec_buf_ptrs->size();
      stats.total_argument_num_ += vec_buf_ptrs->size();
    } else if (auto vec_buf_sptrs =
                   arg.var_.get_if<
                       const std::vector<std::shared_ptr<metal_buffer>> *>()) {
      parse_argument(stats, encoder, *vec_buf_sptrs);
      stats.buffer_arguments_num_ += vec_buf_sptrs->size();
      stats.total_argument_num_ += vec_buf_sptrs->size();
    } else if (auto img_ptr = arg.var_.get_if<const metal_image *>()) {
      parse_argument(stats, encoder, img_ptr);
      stats.image_arguments_num_++;
      stats.total_argument_num_++;
    } else if (auto vec_img_ptrs =
                   arg.var_.get_if<const std::vector<metal_image *> *>()) {
      parse_argument(stats, encoder, (*vec_img_ptrs));
      stats.image_arguments_num_ += vec_img_ptrs->size();
      stats.total_argument_num_ += vec_img_ptrs->size();
    } else if (auto vec_img_sptrs =
                   arg.var_.get_if<
                       const std::vector<std::shared_ptr<metal_image>> *>()) {
      parse_argument(stats, encoder, *vec_img_sptrs);
      stats.image_arguments_num_ += vec_img_sptrs->size();
      stats.total_argument_num_ += vec_img_sptrs->size();
    } else {
      return false;
    }
  }
  return true;
}

bool parse_arguments(
    metal_encoder_t encoder,
    const std::vector<std::pair<metal_kernel_arg, int>> &args) {
  const size_t argc = args.size();
  if (argc < 1) return false;

  arguments_stats stats = arguments_stats();
  for (size_t i = 0; i < argc; i++) {
    const auto &arg = args[i].first;
    auto offset = args[i].second;

    if (auto buf_ptr = arg.var_.get_if<const metal_buffer *>()) {
      parse_argument(stats, encoder, offset, buf_ptr);
      stats.buffer_arguments_num_++;
      stats.total_argument_num_++;
    } else if (auto vec_buf_ptrs =
                   arg.var_.get_if<const std::vector<metal_buffer *> *>()) {
      parse_argument(stats, encoder, *vec_buf_ptrs);
      stats.buffer_arguments_num_ += vec_buf_ptrs->size();
      stats.total_argument_num_ += vec_buf_ptrs->size();
    } else if (auto vec_buf_sptrs =
                   arg.var_.get_if<
                       const std::vector<std::shared_ptr<metal_buffer>> *>()) {
      parse_argument(stats, encoder, *vec_buf_sptrs);
      stats.buffer_arguments_num_ += vec_buf_sptrs->size();
      stats.total_argument_num_ += vec_buf_sptrs->size();
    } else if (auto img_ptr = arg.var_.get_if<const metal_image *>()) {
      parse_argument(stats, encoder, img_ptr);
      stats.image_arguments_num_++;
      stats.total_argument_num_++;
    } else if (auto vec_img_ptrs =
                   arg.var_.get_if<const std::vector<metal_image *> *>()) {
      parse_argument(stats, encoder, *vec_img_ptrs);
      stats.image_arguments_num_ += vec_img_ptrs->size();
      stats.total_argument_num_ += vec_img_ptrs->size();
    } else if (auto vec_img_sptrs =
                   arg.var_.get_if<
                       const std::vector<std::shared_ptr<metal_image>> *>()) {
      parse_argument(stats, encoder, *vec_img_sptrs);
      stats.image_arguments_num_ += vec_img_sptrs->size();
      stats.total_argument_num_ += vec_img_sptrs->size();
    } else if (auto generic_ptr = arg.var_.get_if<const void *>()) {
      parse_argument(stats, encoder, generic_ptr, offset);
      stats.image_arguments_num_ += 1;
      stats.total_argument_num_ += 1;
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace metal_argument_helper
}  // namespace lite
}  // namespace paddle
