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

struct ArgumentsStats {
  uint32_t total_argument_num_{0};
  uint32_t buffer_arguments_num_{0};
  uint32_t image_arguments_num_{0};
};

static void ParseArgument(const ArgumentsStats &stats,
                          metal_encoder_t encoder,
                          const void *ptr,
                          const size_t &size) {
  if (ptr == nullptr) return;
  if (size < 0) return;
  [encoder setBytes:ptr length:size atIndex:stats.buffer_arguments_num_];
}

static void ParseArgument(const ArgumentsStats &stats,
                          metal_encoder_t encoder,
                          const MetalBuffer *arg) {
  if (arg == nullptr) return;
  [encoder setBuffer:arg->buffer_
              offset:static_cast<NSUInteger>(arg->offset())
             atIndex:stats.buffer_arguments_num_];
}

static void ParseArgument(const ArgumentsStats &stats,
                          metal_encoder_t encoder,
                          const MetalImage *arg) {
  if (arg == nullptr) return;
  [encoder setTexture:arg->image() atIndex:stats.image_arguments_num_];
}

static void ParseArgument(
    const ArgumentsStats &stats,
    metal_encoder_t encoder,
    const std::vector<std::shared_ptr<MetalBuffer>> &args) {
  const auto count = args.size();
  if (count < 1) return;

  std::vector<id<MTLBuffer>> mtl_buf_array(count, nil);
  std::vector<NSUInteger> offsets(count, 0);
  for (size_t i = 0; i < count; ++i) {
    mtl_buf_array[i] = args[i].get()->buffer();
    offsets[i] = static_cast<NSUInteger>(args[i].get()->offset());
  }

  [encoder setBuffers:mtl_buf_array.data()
              offsets:offsets.data()
            withRange:NSRange{stats.image_arguments_num_, count}];
}

static void ParseArgument(
    const ArgumentsStats &stats,
    metal_encoder_t encoder,
    const std::vector<std::shared_ptr<MetalImage>> &args) {
  const auto count = args.size();
  if (count < 1) return;

  std::vector<id<MTLTexture>> mtl_buf_array(count, nil);
  for (size_t i = 0; i < count; ++i) {
    mtl_buf_array[i] = args[i].get()->image();
  }

  [encoder setTextures:mtl_buf_array.data()
             withRange:NSRange{stats.image_arguments_num_, count}];
}

static void ParseArgument(const ArgumentsStats &stats,
                          metal_encoder_t encoder,
                          const std::vector<MetalBuffer *> &args) {
  const auto count = args.size();
  if (count < 1) return;

  std::vector<id<MTLBuffer>> mtl_buf_array(count, nil);
  std::vector<NSUInteger> offsets(count, 0);
  for (size_t i = 0; i < count; ++i) {
    mtl_buf_array[i] = (reinterpret_cast<MetalBuffer *>(args[i]))->buffer();
  }

  [encoder setBuffers:mtl_buf_array.data()
              offsets:offsets.data()
            withRange:NSRange{stats.buffer_arguments_num_, count}];
}

static void ParseArgument(const ArgumentsStats &stats,
                          metal_encoder_t encoder,
                          const std::vector<MetalImage *> &args) {
  const auto count = args.size();
  if (count < 1) return;

  std::vector<id<MTLTexture>> mtl_buf_array(count, nil);
  std::vector<NSUInteger> offsets(count, 0);
  for (size_t i = 0; i < count; ++i) {
    mtl_buf_array[i] = args[i]->image();
  }

  [encoder setTextures:mtl_buf_array.data()
             withRange:NSRange{stats.image_arguments_num_, count}];
}

static void ParseArgument(const ArgumentsStats &stats,
                          metal_encoder_t encoder,
                          int offset,
                          const MetalBuffer *arg) {
  if (arg == nullptr) return;
  [encoder setBuffer:arg->buffer()
              offset:static_cast<NSUInteger>(offset)
             atIndex:stats.buffer_arguments_num_];
}

__unused static void ParseArgument(
    const ArgumentsStats &stats,
    metal_encoder_t encoder,
    std::vector<int> offsets_int,
    const std::vector<std::shared_ptr<MetalBuffer>> &args) {
  const auto count = args.size();
  if (count < 1) return;

  std::vector<id<MTLBuffer>> mtl_buf_array(count, nil);
  std::vector<NSUInteger> offsets(count, 0);
  for (size_t i = 0; i < count; ++i) {
    mtl_buf_array[i] = args[i].get()->buffer();
    offsets[i] = static_cast<NSUInteger>(offsets_int[i]);
  }

  [encoder setBuffers:mtl_buf_array.data()
              offsets:offsets.data()
            withRange:NSRange{stats.buffer_arguments_num_, count}];
}

static void ParseArgument(const ArgumentsStats &idx,
                          metal_encoder_t encoder,
                          std::vector<int> offsets_int,
                          const std::vector<MetalBuffer *> &arg) {
  const auto count = arg.size();
  if (count < 1) return;

  std::vector<id<MTLBuffer>> mtl_buf_array(count, nil);
  std::vector<NSUInteger> offsets(count, 0);
  for (size_t i = 0; i < count; ++i) {
    mtl_buf_array[i] = (reinterpret_cast<MetalBuffer *>(arg[i]))->buffer();
    offsets[i] = static_cast<NSUInteger>(offsets_int[i]);
  }

  [encoder setBuffers:mtl_buf_array.data()
              offsets:offsets.data()
            withRange:NSRange{idx.buffer_arguments_num_, count}];
}

bool ParseArguments(metal_encoder_t encoder,
                    const std::vector<MetalKernelArgument> &args) {
  const size_t argc = args.size();
  if (argc < 1) return false;
  ArgumentsStats stats = ArgumentsStats();
  for (size_t i = 0; i < argc; i++) {
    const auto &arg = args[i];
    if (auto buf_ptr = arg.var_.get_if<const MetalBuffer *>()) {
      ParseArgument(stats, encoder, buf_ptr);
      stats.buffer_arguments_num_++;
      stats.total_argument_num_++;
    } else if (auto vec_buf_ptrs =
                   arg.var_.get_if<const std::vector<MetalBuffer *> *>()) {
      ParseArgument(stats, encoder, *vec_buf_ptrs);
      stats.buffer_arguments_num_ += vec_buf_ptrs->size();
      stats.total_argument_num_ += vec_buf_ptrs->size();
    } else if (auto vec_buf_sptrs =
                   arg.var_.get_if<
                       const std::vector<std::shared_ptr<MetalBuffer>> *>()) {
      ParseArgument(stats, encoder, *vec_buf_sptrs);
      stats.buffer_arguments_num_ += vec_buf_sptrs->size();
      stats.total_argument_num_ += vec_buf_sptrs->size();
    } else if (auto img_ptr = arg.var_.get_if<const MetalImage *>()) {
      ParseArgument(stats, encoder, img_ptr);
      stats.image_arguments_num_++;
      stats.total_argument_num_++;
    } else if (auto vec_img_ptrs =
                   arg.var_.get_if<const std::vector<MetalImage *> *>()) {
      ParseArgument(stats, encoder, *vec_img_ptrs);
      stats.image_arguments_num_ += vec_img_ptrs->size();
      stats.total_argument_num_ += vec_img_ptrs->size();
    } else if (auto vec_img_sptrs =
                   arg.var_.get_if<
                       const std::vector<std::shared_ptr<MetalImage>> *>()) {
      ParseArgument(stats, encoder, *vec_img_sptrs);
      stats.image_arguments_num_ += vec_img_sptrs->size();
      stats.total_argument_num_ += vec_img_sptrs->size();
    } else {
      // TODO(lzy): Stuff.
      return false;
    }
  }
  return true;
}

bool ParseArguments(metal_encoder_t encoder,
                    std::vector<int> offset,
                    const std::vector<MetalKernelArgument> &args) {
  const size_t argc = args.size();
  if (argc < 1) return false;

  ArgumentsStats stats = ArgumentsStats();
  for (size_t i = 0; i < argc; i++) {
    const auto &arg = args[i];

    if (auto buf_ptr = arg.var_.get_if<const MetalBuffer *>()) {
      ParseArgument(stats, encoder, offset[i], buf_ptr);
      stats.buffer_arguments_num_++;
      stats.total_argument_num_++;
    } else if (auto vec_buf_ptrs =
                   arg.var_.get_if<const std::vector<MetalBuffer *> *>()) {
      ParseArgument(stats, encoder, (*vec_buf_ptrs));
      stats.buffer_arguments_num_ += vec_buf_ptrs->size();
      stats.total_argument_num_ += vec_buf_ptrs->size();
    } else if (auto vec_buf_sptrs =
                   arg.var_.get_if<
                       const std::vector<std::shared_ptr<MetalBuffer>> *>()) {
      ParseArgument(stats, encoder, *vec_buf_sptrs);
      stats.buffer_arguments_num_ += vec_buf_sptrs->size();
      stats.total_argument_num_ += vec_buf_sptrs->size();
    } else if (auto img_ptr = arg.var_.get_if<const MetalImage *>()) {
      ParseArgument(stats, encoder, img_ptr);
      stats.image_arguments_num_++;
      stats.total_argument_num_++;
    } else if (auto vec_img_ptrs =
                   arg.var_.get_if<const std::vector<MetalImage *> *>()) {
      ParseArgument(stats, encoder, (*vec_img_ptrs));
      stats.image_arguments_num_ += vec_img_ptrs->size();
      stats.total_argument_num_ += vec_img_ptrs->size();
    } else if (auto vec_img_sptrs =
                   arg.var_.get_if<
                       const std::vector<std::shared_ptr<MetalImage>> *>()) {
      ParseArgument(stats, encoder, *vec_img_sptrs);
      stats.image_arguments_num_ += vec_img_sptrs->size();
      stats.total_argument_num_ += vec_img_sptrs->size();
    } else {
      return false;
    }
  }
  return true;
}

bool ParseArguments(
    metal_encoder_t encoder,
    const std::vector<std::pair<MetalKernelArgument, int>> &args) {
  const size_t argc = args.size();
  if (argc < 1) return false;

  ArgumentsStats stats = ArgumentsStats();
  for (size_t i = 0; i < argc; i++) {
    const auto &arg = args[i].first;
    auto offset = args[i].second;

    if (auto buf_ptr = arg.var_.get_if<const MetalBuffer *>()) {
      ParseArgument(stats, encoder, offset, buf_ptr);
      stats.buffer_arguments_num_++;
      stats.total_argument_num_++;
    } else if (auto vec_buf_ptrs =
                   arg.var_.get_if<const std::vector<MetalBuffer *> *>()) {
      ParseArgument(stats, encoder, *vec_buf_ptrs);
      stats.buffer_arguments_num_ += vec_buf_ptrs->size();
      stats.total_argument_num_ += vec_buf_ptrs->size();
    } else if (auto vec_buf_sptrs =
                   arg.var_.get_if<
                       const std::vector<std::shared_ptr<MetalBuffer>> *>()) {
      ParseArgument(stats, encoder, *vec_buf_sptrs);
      stats.buffer_arguments_num_ += vec_buf_sptrs->size();
      stats.total_argument_num_ += vec_buf_sptrs->size();
    } else if (auto img_ptr = arg.var_.get_if<const MetalImage *>()) {
      ParseArgument(stats, encoder, img_ptr);
      stats.image_arguments_num_++;
      stats.total_argument_num_++;
    } else if (auto vec_img_ptrs =
                   arg.var_.get_if<const std::vector<MetalImage *> *>()) {
      ParseArgument(stats, encoder, *vec_img_ptrs);
      stats.image_arguments_num_ += vec_img_ptrs->size();
      stats.total_argument_num_ += vec_img_ptrs->size();
    } else if (auto vec_img_sptrs =
                   arg.var_.get_if<
                       const std::vector<std::shared_ptr<MetalImage>> *>()) {
      ParseArgument(stats, encoder, *vec_img_sptrs);
      stats.image_arguments_num_ += vec_img_sptrs->size();
      stats.total_argument_num_ += vec_img_sptrs->size();
    } else if (auto generic_ptr = arg.var_.get_if<const void *>()) {
      ParseArgument(stats, encoder, generic_ptr, offset);
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
